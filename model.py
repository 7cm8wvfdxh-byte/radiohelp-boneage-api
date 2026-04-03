"""
RadioHelp — Model Tanimlari
ConvNeXt V1 Base (timm) + Gender — MAE 6.76 ay
"""

import torch
import torch.nn as nn
import timm


class BoneAgeModel(nn.Module):
    """
    ConvNeXt V1 Base + Gender embedding.
    Egitim: RSNA Bone Age, U-Net maskeli, 512px, z-score normalize.
    Cikti: z-score normalized (raw * 41.2 + 127.3 = ay)
    """

    def __init__(self):
        super().__init__()
        self.backbone = timm.create_model(
            'convnext_base.fb_in22k_ft_in1k_384',
            pretrained=False,
            num_classes=0,
        )
        feat_dim = self.backbone.num_features  # 1024

        self.gender_fc = nn.Sequential(
            nn.Linear(1, 32),
            nn.ReLU(),
        )

        self.head = nn.Sequential(
            nn.Linear(feat_dim + 32, 256),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(256, 1),
        )

    def forward(self, image, gender):
        features = self.backbone(image)
        gender_features = self.gender_fc(gender.unsqueeze(1))
        combined = torch.cat([features, gender_features], dim=1)
        return self.head(combined).squeeze(1)
