import math
import torch
import torch.nn as nn
from histogram_helpers.HistogramLoss import color_histogram_of_training_image

# write a  adaptive instance normalization layer
class AdaIN(nn.Module):
    def __init__(self, eps=1e-5):
        super(AdaIN, self).__init__()
        self.eps = eps

    def forward(self, x, y):
        #print(f"x shape: {x.shape}, y shape: {y.shape}")
        #print(f"y mean: {y.mean()}, y std: {y.std()}")
        y_mean = y.mean(dim=[2, 3], keepdim=True)
        y_std = y.std(dim=[2, 3], keepdim=True)
        #print(f"y_mean shape: {y_mean.shape}, y_std shape: {y_std.shape}")
        #print(f"y_mean: {y_mean}, y_std: {y_std}")
        x_normalized = (x - x.mean(dim=[2, 3], keepdim=True)) / (x.std(dim=[2, 3], keepdim=True) + self.eps)
        #print(f"x_normalized shape: {x_normalized.shape}")
        #print(f"x_normalized mean: {x_normalized.mean()}, x_normalized std: {x_normalized.std()}")
        return x_normalized * y_std + y_mean

# write a style transfer network
class StyleTransfer1(nn.Module):
    def __init__(self, in_channel):
        super(StyleTransfer1, self).__init__()
        self.in_channel = in_channel
        self.mlp = nn.Sequential(
            nn.Flatten(),
            nn.Linear(1, 1),  # Adjusted input size
            nn.ReLU(),
            nn.Linear(1, 1),
            nn.ReLU(),
            nn.Linear(1, 1),
            nn.ReLU(),
            nn.Linear(1, 1),
            nn.ReLU(),
            nn.Linear(1, 1),
            nn.ReLU(),
            nn.Linear(1, 1),
            nn.ReLU(),
            nn.Linear(1, 1),
            nn.ReLU(),
            nn.Linear(1, 1),  
        )
        self.conv1 = nn.Conv2d(in_channel, in_channel, kernel_size=(4, 4), stride=(2, 2), padding=(1, 1), bias=False)
        self.conv2 = nn.Conv2d(in_channel, in_channel*2, kernel_size=(4, 4), stride=(2, 2), padding=(1, 1), bias=False)
        self.conv3 = nn.Conv2d(in_channel*2, in_channel*4, kernel_size=(4, 4), stride=(2, 2), padding=(1, 1), bias=False)

        self.LeakyReLU = nn.LeakyReLU(negative_slope=0.2, inplace=True)
        self.BatchNorm2d1 = nn.BatchNorm2d(in_channel, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.BatchNorm2d2 = nn.BatchNorm2d(in_channel*2, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.BatchNorm2d3 = nn.BatchNorm2d(in_channel*4, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)



    def forward(self, x):
        x = self.conv1(x)
        x = self.LeakyReLU(x)
        x = self.BatchNorm2d1(x)
        x = self.conv2(x)
        x = self.LeakyReLU(x)
        x = self.BatchNorm2d2(x)
        x = self.conv3(x)
        x = self.LeakyReLU(x)
        x = self.BatchNorm2d3(x)
        x = x.view(-1, self.in_channel*4)
        #print(f"before mlp: {x.shape}")
        x = self.mlp(x)
        #print(f"after mlp: {x.shape}")
        return x 

    

# write a a network that takes a latent vector that is taken from gaussian distribution takes it as input and outputs information from adaptive instance normalization layer
class StyleTransfer2(nn.Module):
    def __init__(self):
        super(StyleTransfer2, self).__init__()
        self.initial_latent = nn.Parameter(torch.randn(3, 32, 32))
        self.net = nn.Sequential(
            nn.Flatten(),
            nn.Linear(1024, 1024),  # Adjusted input size
            nn.ReLU(),
            nn.Linear(1024, 1024),
            nn.ReLU(),
            nn.Linear(1024, 1024),
            nn.ReLU(),
            
            nn.Linear(1024, 1024),
            nn.ReLU(),
            
            nn.Linear(1024, 1024),
            nn.ReLU(),
            
            nn.Linear(1024, 1024),
            nn.ReLU(),
            
            nn.Linear(1024, 1024),
            nn.ReLU(),
            
            nn.Linear(1024, 1024),
            nn.ReLU()
        )
    
    def forward(self, x):
        x = self.net(self.initial_latent)
        #print(f"self.initial_latent: {self.initial_latent},\n x: {x}")
        return x

class StyleTransfer3(nn.Module):
    def __init__(self):
        super(StyleTransfer3, self).__init__()
        self.mlp = nn.Sequential(
            nn.Linear(256, 256),  # Adjusted input size
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
            
            nn.Linear(256, 256),
            nn.ReLU(),
            
            nn.Linear(256, 256),
            nn.ReLU(),
            
            nn.Linear(256, 256),
            nn.ReLU(),
            
            nn.Linear(256, 256),
            nn.ReLU(),
            
            nn.Linear(256, 256),
            nn.ReLU()
        )
        self.final_mlp = nn.Sequential(
            nn.Linear(256, 256),  # Adjusted input size
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256, 256),  
        )
        self.num_bins = 256
        
    def forward(self, style_img):
        r = style_img[:, 0, :, :]
        g = style_img[:, 1, :, :]
        b = style_img[:, 2, :, :]
        
        # calculate the histogram of each channel
        self.r = torch.histc(r, bins=256, min=0, max=1)
        self.g = torch.histc(g, bins=256, min=0, max=1)
        self.b = torch.histc(b, bins=256, min=0, max=1)
        
        r = self.mlp(self.r)
        g = self.mlp(self.g)
        b = self.mlp(self.b)
        
        #print(f"r shape: {r.shape}")

        rgb = torch.cat((r, g, b), dim=0)
        rgb = rgb.reshape(3, 256)
        #print(f"rgb shape = {rgb.shape}")
        rgb = self.final_mlp(rgb)
        return rgb

class HistogramLoss(nn.Module):
    def __init__(self):
        super(HistogramLoss, self).__init__()

    def forward(self, x,y):
        r1, g1, b1 = color_histogram_of_training_image(y)
        r2, g2, b2 = color_histogram_of_training_image(x)

        # calculate KL divergence of two histograms
        r = torch.sum(r1 * torch.log(r1 / r2))
        g = torch.sum(g1 * torch.log(g1 / g2))
        b = torch.sum(b1 * torch.log(b1 / b2))

        return r + g + b

class HistogramLoss(nn.Module):
    def __init__(self):
        super(HistogramLoss, self).__init__()

    def forward(self, x,y):
        r1, g1, b1 = color_histogram_of_training_image(y)
        r2, g2, b2 = color_histogram_of_training_image(x)

        # calculate KL divergence of two histograms
        r = torch.sum(r1 * torch.log(r1 / r2))
        g = torch.sum(g1 * torch.log(g1 / g2))
        b = torch.sum(b1 * torch.log(b1 / b2))

        return r + g + b

    def compute_histogram(self, image):
        # Clamp the image values to the range [0, 1]
        image = torch.clamp(image, 0, 1)

        # Multiply the image values by (num_bins - 1) and round them to integers
        bins = (self.num_bins - 1) * image

        # Flatten the tensor along the channel and spatial dimensions
        bins = bins.view(bins.size(0), -1).long()

        # Compute the histogram using the bincount function
        histogram = torch.stack([torch.bincount(b, minlength=self.num_bins) for b in bins])
        return histogram
    
class GANLoss(nn.Module):
    def __init__(self, gan_mode='vanilla', real_label=1.0, fake_label=0.0):
        super().__init__()
        self.register_buffer('real_label', torch.tensor(real_label))
        self.register_buffer('fake_label', torch.tensor(fake_label))
        if gan_mode == 'vanilla':
            self.loss = nn.BCEWithLogitsLoss()
        elif gan_mode == 'lsgan':
            self.loss = nn.MSELoss()
    
    def get_labels(self, preds, target_is_real):
        if target_is_real:
            labels = self.real_label
        else:
            labels = self.fake_label
        return labels.expand_as(preds)
    
    def __call__(self, preds, target_is_real):
        labels = self.get_labels(preds, target_is_real)
        loss = self.loss(preds, labels)
        return loss
    

def init_weights(net, init='norm', gain=0.02):
    
    def init_func(m):
        classname = m.__class__.__name__
        if hasattr(m, 'weight') and 'Conv' in classname:
            if init == 'norm':
                nn.init.normal_(m.weight.data, mean=0.0, std=gain)
            elif init == 'xavier':
                nn.init.xavier_normal_(m.weight.data, gain=gain)
            elif init == 'kaiming':
                nn.init.kaiming_normal_(m.weight.data, a=0, mode='fan_in')
            
            if hasattr(m, 'bias') and m.bias is not None:
                nn.init.constant_(m.bias.data, 0.0)
        elif 'BatchNorm2d' in classname:
            nn.init.normal_(m.weight.data, 1., gain)
            nn.init.constant_(m.bias.data, 0.)
            
    net.apply(init_func)
    print(f"model initialized with {init} initialization")
    return net

def init_model(model, device):
    model = model.to(device)
    model = init_weights(model)
    return model

