
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models

class SuperPointEncoder(nn.Module):
    """SuperPoint Encoder (VGG-style) for extracting dense descriptors."""
    def __init__(self, weight_path=None, freeze=True):
        super().__init__()
        self.relu = nn.ReLU(inplace=True)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        c1, c2, c3, c4 = 64, 64, 128, 128

        self.conv1a = nn.Conv2d(1, c1, kernel_size=3, stride=1, padding=1)
        self.conv1b = nn.Conv2d(c1, c1, kernel_size=3, stride=1, padding=1)
        self.conv2a = nn.Conv2d(c1, c2, kernel_size=3, stride=1, padding=1)
        self.conv2b = nn.Conv2d(c2, c2, kernel_size=3, stride=1, padding=1)
        self.conv3a = nn.Conv2d(c2, c3, kernel_size=3, stride=1, padding=1)
        self.conv3b = nn.Conv2d(c3, c3, kernel_size=3, stride=1, padding=1)
        self.conv4a = nn.Conv2d(c3, c4, kernel_size=3, stride=1, padding=1)
        self.conv4b = nn.Conv2d(c4, c4, kernel_size=3, stride=1, padding=1)
        
        self.convDa = nn.Conv2d(c4, 256, kernel_size=3, stride=1, padding=1)
        self.convDb = nn.Conv2d(256, 256, kernel_size=1, stride=1, padding=0)
        
        if weight_path and os.path.exists(weight_path):
            print(f"Loading SuperPoint weights from {weight_path}")
            self.load_state_dict(torch.load(weight_path), strict=False)
        else:
             print("Warning: No SuperPoint weights loaded or file not found!")

        if freeze:
            for param in self.parameters():
                param.requires_grad = False
            self.eval()

    def forward(self, x):
        x = self.relu(self.conv1b(self.relu(self.conv1a(x))))
        x = self.pool(x) # 1/2
        x = self.relu(self.conv2b(self.relu(self.conv2a(x))))
        x = self.pool(x) # 1/4
        x = self.relu(self.conv3b(self.relu(self.conv3a(x))))
        x = self.pool(x) # 1/8
        x = self.relu(self.conv4b(self.relu(self.conv4a(x))))
        cDa = self.relu(self.convDa(x))
        desc = self.convDb(cDa) 
        desc = torch.nn.functional.normalize(desc, p=2, dim=1)
        return desc

class FineEncoder(nn.Module):
    """精特征编码器: VGG19"""
    def __init__(self, pretrained=True):
        super().__init__()
        vgg19 = models.vgg19(pretrained=pretrained)
        self.features = vgg19.features
        self.coarse_layers = nn.Sequential(*list(self.features.children())[:18])
        self.fine_layers = nn.Sequential(*list(self.features.children())[:10])
    
    def forward(self, x, vessel_mask=None):
        if x.shape[1] == 1:
            x = x.repeat(1, 3, 1, 1)
        feat_fine = self.fine_layers(x)
        feat_coarse = self.coarse_layers(x)
        return feat_coarse, feat_fine

class RoMaBackbone(nn.Module):
    """
    RoMa 双流 Backbone (SuperPoint Edition):
    - 粗特征流: SuperPoint (Geometry)
    - 精特征流: VGG19 (Texture)
    """
    def __init__(self, config):
        super().__init__()
        self.config = config
        
        # SuperPoint Coarse Encoder
        spp_path = config.get('ROMA', {}).get('SUPERPOINT_PATH', 'weights/superpoint_v1.pth')
        self.coarse_encoder = SuperPointEncoder(weight_path=spp_path, freeze=True)
        
        # VGG Fine Encoder
        self.fine_encoder = FineEncoder(pretrained=True)
        
    def forward(self, x, vessel_mask=None):
        # 1. Fine Stream (VGG)
        feat_coarse_vgg, feat_fine = self.fine_encoder(x, None) # [B, 256, H/4, W/4], [B, 128, H/4, W/4]
        
        # 2. Coarse Stream (SuperPoint)
        feat_coarse_spp = self.coarse_encoder(x) # [B, 256, H/8, W/8]
        
        # 3. Align Resolutions (All to H/8)
        # Downsample VGG coarse feature from H/4 to H/8
        # Using Average Pooling or Strided Convolution would be better, but interpolate is simple
        target_h, target_w = feat_coarse_spp.shape[2:]
        if feat_coarse_vgg.shape[2:] != (target_h, target_w):
            feat_coarse_vgg = F.interpolate(feat_coarse_vgg, size=(target_h, target_w), 
                                            mode='bilinear', align_corners=False)
            
        # 4. Fusion
        feat_c = torch.cat([feat_coarse_vgg, feat_coarse_spp], dim=1) # [B, 512, H/8, W/8]
        
        return feat_c, feat_fine, None # x_adapted is None for SuperPoint
