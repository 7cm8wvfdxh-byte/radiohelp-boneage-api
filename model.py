"""
RadioHelp — Model Tanımlamaları
V1 (ConvNeXt, EfficientNet), V2 (CBAM), Ensemble
"""

import torch
import torch.nn as nn
from torchvision.models import (
    convnext_small, ConvNeXt_Small_Weights,
    efficientnet_b4, EfficientNet_B4_Weights
)


# ============================================================
#  CBAM (Convolutional Block Attention Module)
# ============================================================

class ChannelAttention(nn.Module):
    def __init__(self, channels, reduction=16):
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channels, channels // reduction, bias=False),
            nn.ReLU(),
            nn.Linear(channels // reduction, channels, bias=False)
        )
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        b, c, _, _ = x.size()
        avg_out = self.fc(self.avg_pool(x).view(b, c))
        max_out = self.fc(self.max_pool(x).view(b, c))
        return self.sigmoid(avg_out + max_out).view(b, c, 1, 1)


class SpatialAttention(nn.Module):
    def __init__(self, kernel_size=7):
        super().__init__()
        self.conv = nn.Conv2d(2, 1, kernel_size, padding=kernel_size // 2, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        combined = torch.cat([avg_out, max_out], dim=1)
        return self.sigmoid(self.conv(combined))


class CBAM(nn.Module):
    def __init__(self, channels, reduction=16, kernel_size=7):
        super().__init__()
        self.channel_att = ChannelAttention(channels, reduction)
        self.spatial_att = SpatialAttention(kernel_size)

    def forward(self, x):
        x = x * self.channel_att(x)
        x = x * self.spatial_att(x)
        return x


# ============================================================
#  V1: Temel Model
# ============================================================

class BoneAgeModel(nn.Module):
    def __init__(self, backbone_name="convnext_small"):
        super().__init__()
        self.backbone_name = backbone_name

        if "convnext" in backbone_name:
            self.backbone = convnext_small(weights=ConvNeXt_Small_Weights.IMAGENET1K_V1)
            num_features = self.backbone.classifier[2].in_features
            self.backbone.classifier = nn.Identity()
        else:
            self.backbone = efficientnet_b4(weights=EfficientNet_B4_Weights.IMAGENET1K_V1)
            num_features = self.backbone.classifier[1].in_features
            self.backbone.classifier = nn.Identity()

        self.gender_fc = nn.Sequential(
            nn.Linear(1, 32),
            nn.ReLU()
        )

        self.regressor = nn.Sequential(
            nn.Linear(num_features + 32, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(512, 128),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(128, 1),
            nn.Sigmoid()
        )

    def forward(self, image, gender):
        img_features = self.backbone(image)
        if len(img_features.shape) > 2:
            img_features = torch.flatten(img_features, 1)
        gender_features = self.gender_fc(gender.unsqueeze(1))
        combined = torch.cat([img_features, gender_features], dim=1)
        return self.regressor(combined).squeeze(1)


# ============================================================
#  V2: CBAM Model
# ============================================================

class BoneAgeModelV2(nn.Module):
    def __init__(self):
        super().__init__()
        backbone = convnext_small(weights=ConvNeXt_Small_Weights.IMAGENET1K_V1)

        self.stage0 = backbone.features[0]
        self.stage1 = backbone.features[1]
        self.stage2 = nn.Sequential(backbone.features[2], backbone.features[3])
        self.stage3 = nn.Sequential(backbone.features[4], backbone.features[5])
        self.stage4 = nn.Sequential(backbone.features[6], backbone.features[7])

        self.cbam_s2 = CBAM(192)
        self.cbam_s3 = CBAM(384)
        self.cbam_s4 = CBAM(768)

        num_features = 768

        self.gender_fc = nn.Sequential(
            nn.Linear(1, 32),
            nn.ReLU()
        )

        self.regressor = nn.Sequential(
            nn.Linear(num_features + 32, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(512, 128),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(128, 1),
            nn.Sigmoid()
        )

    def forward(self, image, gender):
        x = self.stage0(image)
        x = self.stage1(x)
        x = self.stage2(x)
        x = self.cbam_s2(x)
        x = self.stage3(x)
        x = self.cbam_s3(x)
        x = self.stage4(x)
        x = self.cbam_s4(x)

        x = nn.functional.adaptive_avg_pool2d(x, 1)
        img_features = torch.flatten(x, 1)

        gender_features = self.gender_fc(gender.unsqueeze(1))
        combined = torch.cat([img_features, gender_features], dim=1)
        return self.regressor(combined).squeeze(1)


# ============================================================
#  Ensemble
# ============================================================

class BoneAgeEnsemble(nn.Module):
    def __init__(self):
        super().__init__()
        self.model_convnext = BoneAgeModel(backbone_name="convnext_small")
        self.model_effnet = BoneAgeModel(backbone_name="efficientnet_b4")

        self.weight_layer = nn.Sequential(
            nn.Linear(2, 8),
            nn.ReLU(),
            nn.Linear(8, 2),
            nn.Softmax(dim=1)
        )

    def forward(self, image, gender):
        pred1 = self.model_convnext(image, gender).unsqueeze(1)
        pred2 = self.model_effnet(image, gender).unsqueeze(1)
        stacked = torch.cat([pred1, pred2], dim=1)
        weights = self.weight_layer(stacked.detach())
        weighted_pred = (stacked * weights).sum(dim=1)
        return weighted_pred

    def load_pretrained(self, convnext_path, effnet_path, device='cpu'):
        if convnext_path:
            ckpt = torch.load(convnext_path, map_location=device, weights_only=False)
            state = ckpt.get('model_state_dict', ckpt)
            self.model_convnext.load_state_dict(state, strict=False)
            print(f"✅ ConvNeXt yüklendi: {convnext_path}")
        if effnet_path:
            ckpt = torch.load(effnet_path, map_location=device, weights_only=False)
            state = ckpt.get('model_state_dict', ckpt)
            self.model_effnet.load_state_dict(state, strict=False)
            print(f"✅ EfficientNet yüklendi: {effnet_path}")
