import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms

class AdaIN2d(nn.Module):
    def __init__(self, style_dim, num_features):
        super().__init__()
        self.norm = nn.InstanceNorm2d(num_features, affine=False)
        self.fc = nn.Linear(style_dim, num_features*2)
    
    def forward(self, x, s): 
        h = self.fc(s)
        h = h.view(h.size(0), h.size(1), 1, 1)
        gamma, beta = torch.chunk(h, chunks=2, dim=1)
        return (1 + gamma) * self.norm(x) + beta


class RandMix(nn.Module):
    def __init__(self, noise_lv):
        super(RandMix, self).__init__()
        ############# Trainable Parameters
        self.zdim = zdim = 10
        self.noise_lv = noise_lv
        self.adain_1 = AdaIN2d(zdim, 3)
        self.adain_2 = AdaIN2d(zdim, 3)
        self.adain_3 = AdaIN2d(zdim, 3)
        self.adain_4 = AdaIN2d(zdim, 3)


        self.tran = transforms.Normalize([0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

    def forward(self, x, estimation=False, ratio=0):
        data = x
        
        spatial1 = nn.Conv2d(3, 3, 5).cuda()
        spatial_up1 = nn.ConvTranspose2d(3, 3, 5).cuda()

        spatial2 = nn.Conv2d(3, 3, 9).cuda()
        spatial_up2 = nn.ConvTranspose2d(3, 3, 9).cuda()

        spatial3 = nn.Conv2d(3, 3, 13).cuda()
        spatial_up3 = nn.ConvTranspose2d(3, 3, 13).cuda()

        spatial4 = nn.Conv2d(3, 3, 17).cuda()
        spatial_up4 = nn.ConvTranspose2d(3, 3, 17).cuda()

        color = nn.Conv2d(3, 3, 1).cuda()
        weight = torch.randn(6)

        x = x + torch.randn_like(x) * self.noise_lv * 0.001
        x_c = torch.tanh(F.dropout(color(x), p=.2))

        x_s1down = spatial1(x)
        s = torch.randn(len(x_s1down), self.zdim).cuda()
        x_s1down = self.adain_1(x_s1down, s)
        x_s1 = torch.tanh(spatial_up1(x_s1down))

        x_s2down = spatial2(x)
        s = torch.randn(len(x_s2down), self.zdim).cuda()
        x_s2down = self.adain_2(x_s2down, s)
        x_s2 = torch.tanh(spatial_up2(x_s2down))

        x_s3down = spatial3(x)
        s = torch.randn(len(x_s3down), self.zdim).cuda()
        x_s3down = self.adain_3(x_s3down, s)
        x_s3 = torch.tanh(spatial_up3(x_s3down))

        x_s4down = spatial4(x)
        s = torch.randn(len(x_s4down), self.zdim).cuda()
        x_s4down = self.adain_4(x_s4down, s)
        x_s4 = torch.tanh(spatial_up4(x_s4down))
        
        output = (weight[0] * x_c + weight[1] * x_s1 + weight[2] * x_s2 + weight[3] * x_s3 + weight[4] * x_s4 + weight[5] * data) / weight.sum()
        return output
