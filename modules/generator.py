import torch
import torch.nn as nn
import modules.helpers as helpers
from helpers import AdaIN

class UnetBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding, bias):
        super(UnetBlock, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, bias=bias)
        self.leaky_relu = nn.LeakyReLU(negative_slope=0.2, inplace=True)
        self.batch_norm = nn.BatchNorm2d(out_channels, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)

    def forward(self, x):
        x = self.conv(x)
        x = self.leaky_relu(x)
        x = self.batch_norm(x)
        return x

class SelfAttention(nn.Module):
    def __init__(self, in_dim):
        super(SelfAttention, self).__init__()
        self.chanel_in = in_dim
        
        self.query_conv = nn.Conv2d(in_channels = in_dim , out_channels = in_dim//8 , kernel_size= 1)
        self.key_conv = nn.Conv2d(in_channels = in_dim , out_channels = in_dim//8 , kernel_size= 1)
        self.value_conv = nn.Conv2d(in_channels = in_dim , out_channels = in_dim , kernel_size= 1)
        self.gamma = nn.Parameter(torch.zeros(1))
        self.softmax  = nn.Softmax(dim=-1) #
        
    def forward(self,x):
        m_batchsize,C,width ,height = x.size()
        proj_query  = self.query_conv(x).view(m_batchsize,-1,width*height).permute(0,2,1) # B X CX(N)
        proj_key =  self.key_conv(x).view(m_batchsize,-1,width*height) # B X C x (*W*H)
        energy =  torch.bmm(proj_query,proj_key) # transpose check
        attention = self.softmax(energy) # BX (N) X (N) 
        proj_value = self.value_conv(x).view(m_batchsize,-1,width*height) # B X C X N
        
        out = torch.bmm(proj_value,attention.permute(0,2,1) )
        out = out.view(m_batchsize,C,width,height)
        
        out = self.gamma*out + x
        return out,attention
    
class Unet(nn.Module):
    def __init__(self, skip_connections=[True,True,True,True,True,True], attentions=[True,True,True,True,True,True]):
        super(Unet, self).__init__()
        self.skip_connections = skip_connections
        self.attentions = attentions 
        self.conv1 = nn.Conv2d(1, 64, kernel_size=(4, 4), stride=(2, 2), padding=(1, 1), bias=False)
        self.block1 = UnetBlock(64, 128, kernel_size=(4, 4), stride=(2, 2), padding=(1, 1), bias=False)
        self.block2 = UnetBlock(128, 256, kernel_size=(4, 4), stride=(2, 2), padding=(1, 1), bias=False)
        self.block3 = UnetBlock(256, 512, kernel_size=(4, 4), stride=(2, 2), padding=(1, 1), bias=False)
        self.block4 = UnetBlock(512, 512, kernel_size=(4, 4), stride=(2, 2), padding=(1, 1), bias=False)
        self.block5 = UnetBlock(512, 512, kernel_size=(4, 4), stride=(2, 2), padding=(1, 1), bias=False)
        self.block6 = UnetBlock(512, 512, kernel_size=(4, 4), stride=(2, 2), padding=(1, 1), bias=False)
        self.block7 = UnetBlock(512, 512, kernel_size=(4, 4), stride=(2, 2), padding=(1, 1), bias=False)
        self.relu = nn.ReLU(inplace=True)
        self.deconv1 = nn.ConvTranspose2d(512, 512, kernel_size=(4, 4), stride=(2, 2), padding=(1, 1), bias=False)
        self.bn1 = nn.BatchNorm2d(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.adaIn1 = AdaIN()
        
        
        
        
        x = 512
        self.attention512 = SelfAttention(512)
        
        if skip_connections[0]:
            x = 1024
        self.deconv2 = nn.ConvTranspose2d(x, 512, kernel_size=(4, 4), stride=(2, 2), padding=(1, 1))
        
        self.bn2 = nn.BatchNorm2d(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.dropout1 = nn.Dropout(p=0.5, inplace=False)
        
        x = 512
        if skip_connections[1]:
            x = 1024
        self.deconv3 = nn.ConvTranspose2d(x, 512, kernel_size=(4, 4), stride=(2, 2), padding=(1, 1))
        self.bn3 = nn.BatchNorm2d(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.dropout2 = nn.Dropout(p=0.5, inplace=False)
        
        x = 512
        if skip_connections[2]:
            x = 1024
        self.deconv4 = nn.ConvTranspose2d(x, 512, kernel_size=(4, 4), stride=(2, 2), padding=(1, 1))
        self.bn4 = nn.BatchNorm2d(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.dropout3 = nn.Dropout(p=0.5, inplace=False)
        
        x = 512
        if skip_connections[3]:
            x = 1024
        self.deconv5 = nn.ConvTranspose2d(x, 256, kernel_size=(4, 4), stride=(2, 2), padding=(1, 1))
        self.bn5 = nn.BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        
        
        x = 256
        self.attention256 = SelfAttention(256)
        if skip_connections[4]:
            x = 512
        self.deconv6 = nn.ConvTranspose2d(x, 128, kernel_size=(4, 4), stride=(2, 2), padding=(1, 1))
        self.bn6 = nn.BatchNorm2d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        
        
        x = 128
        self.attention128 = SelfAttention(128)
        if skip_connections[5]:
            x = 256
        self.deconv7 = nn.ConvTranspose2d(x, 64, kernel_size=(4, 4), stride=(2, 2), padding=(1, 1))
        self.bn7 = nn.BatchNorm2d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.deconv8 = nn.ConvTranspose2d(64, 3, kernel_size=(4, 4), stride=(2, 2), padding=(1, 1))
        self.tanh = nn.Tanh()

    def forward(self, x, style_feature):
        x1 = self.conv1(x)
        x1 = self.relu(x1)
        x2 = self.adaIn1(self.block1(x1), style_feature)
        x3 = self.adaIn1(self.block2(x2), style_feature)
        x4 = self.adaIn1(self.block3(x3), style_feature)
        x5 = self.adaIn1(self.block4(x4), style_feature)
        x6 = self.adaIn1(self.block5(x5), style_feature)
        x7 = self.adaIn1(self.block6(x6), style_feature)
        x8 = self.block7(x7)

        x = self.relu(self.deconv1(x8))
        x = self.bn1(x)
        
        if self.skip_connections[0]:
            if self.attentions[0]:
                x = self.relu(self.deconv2(torch.cat([x, self.attention512(x7)[0]], dim=1)))
            else:
                x = self.relu(self.deconv2(torch.cat([x, x7], dim=1)))
        else:
            x = self.relu(self.deconv2(x))
            
        x = self.bn2(x)
        x = self.dropout1(x)
        
        if self.skip_connections[1]:
            if self.attentions[1]:
                x = self.relu(self.deconv3(torch.cat([x, self.attention512(x6)[0]], dim=1)))
            else:
                x = self.relu(self.deconv3(torch.cat([x, x6], dim=1)))
        else:
            x = self.relu(self.deconv3(x))
            
        x = self.bn3(x)
        x = self.dropout2(x)
        
        if self.skip_connections[2]:
            if self.attentions[2]:
                x = self.relu(self.deconv4(torch.cat([x, self.attention512(x5)[0]], dim=1)))
            else:
                x = self.relu(self.deconv4(torch.cat([x, x5], dim=1)))
        else:
            x = self.relu(self.deconv4(x))
        
        x = self.bn4(x)
        x = self.dropout3(x)
        
        if self.skip_connections[3]:
            if self.attentions[3]:
                x = self.relu(self.deconv5(torch.cat([x, self.attention512(x4)[0]], dim=1)))
            else:
                x = self.relu(self.deconv5(torch.cat([x, x4], dim=1)))
        else:
            x = self.relu(self.deconv5(x))
        
        x = self.bn5(x)
        
        if self.skip_connections[4]:
            if self.attentions[4]:
                x = self.relu(self.deconv6(torch.cat([x, self.attention256(x3)[0]], dim=1)))
            else:
                x = self.relu(self.deconv6(torch.cat([x, x3], dim=1)))
        else:
            x = self.relu(self.deconv6(x))
        
        x = self.bn6(x)
        
        if self.skip_connections[5]:
            if self.attentions[5]:
                x = self.relu(self.deconv7(torch.cat([x, self.attention128(x2)[0]], dim=1)))
            else:
                x = self.relu(self.deconv7(torch.cat([x, x2], dim=1)))
        else:
            x = self.relu(self.deconv7(x))
        
        x = self.bn7(x)
        x = self.deconv8(x)
        x = self.tanh(x)

        return x

     