import torch
import torch.nn as nn
import torch.optim as optim
from modules.generator import Unet
from modules.discriminator import PatchDiscriminator
from modules.helpers import init_model, GANLoss, ColorHistogramLoss

class MainModel(nn.Module):
    def __init__(self, net_G=None, lr_G=2e-4, lr_D=2e-4, 
                 beta1=0.5, beta2=0.999, lambda_L1=100., 
                 skip_connections=[True,True,True,True,True,True,True],
                 attentions=[True,True,True,True,True,True,True],
                 style_net=None):
        super().__init__()
        
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.lambda_L1 = lambda_L1
        self.style_net = style_net
        
        if net_G is None:
            self.net_G = init_model(Unet(skip_connections=skip_connections, 
                                         attentions=attentions), self.device)
        else:
            self.net_G = net_G.to(self.device)
        self.net_D = init_model(PatchDiscriminator(input_c=3, n_down=3, num_filters=64), self.device)
        self.GANcriterion = GANLoss(gan_mode='vanilla').to(self.device)
        self.L1criterion = nn.L1Loss()
        self.opt_G = optim.Adam(self.net_G.parameters(), lr=lr_G, betas=(beta1, beta2))
        self.opt_D = optim.Adam(self.net_D.parameters(), lr=lr_D, betas=(beta1, beta2))
        self.hist_loss = ColorHistogramLoss()
    
    def set_requires_grad(self, model, requires_grad=True):
        for p in model.parameters():
            p.requires_grad = requires_grad
        
    def setup_input(self, data):
        self.grayscale = data['grayscale'].to(self.device)
        self.rgb = data['rgb'].to(self.device)
        
    def forward(self):
        x = self.style_net(self.rgb)
        self.fake_color = self.net_G(self.grayscale, x)
        #print(self.fake_color)
    
    def backward_D(self):
        fake_image = self.fake_color
        fake_preds = self.net_D(fake_image.detach())
        self.loss_D_fake = self.GANcriterion(fake_preds, False)
        real_image = self.rgb
        real_preds = self.net_D(real_image)
        self.loss_D_real = self.GANcriterion(real_preds, True)
        self.loss_D = (self.loss_D_fake + self.loss_D_real) * 0.5
        self.loss_D.backward()
    
    def backward_G(self):
        fake_image = self.fake_color
        fake_preds = self.net_D(fake_image)
        self.loss_G_GAN = self.GANcriterion(fake_preds, True)
        self.loss_G_L1 = self.L1criterion(self.fake_color, self.rgb) * self.lambda_L1
        self.loss_color = self.hist_loss(self.rgb, self.fake_color)
        self.loss_G = self.loss_G_GAN + self.loss_G_L1 + self.loss_color
        self.loss_G.backward()
    
    def optimize(self):
        self.forward()
        self.net_D.train()
        self.set_requires_grad(self.net_D, True)
        self.opt_D.zero_grad()
        self.backward_D()
        self.opt_D.step()
        
        self.net_G.train()
        self.set_requires_grad(self.net_D, False)
        self.opt_G.zero_grad()
        self.backward_G()
        self.opt_G.step()