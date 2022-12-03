import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F
from models.build import MODELS
from timm.models.layers import trunc_normal_
from utils.logger import *

class Linear_ResBlock(nn.Module):
    def __init__(self, input_size=1024, output_size=1024):
        super(Linear_ResBlock, self).__init__()
        self.conv1 = nn.Linear(input_size, input_size)
        self.conv2 = nn.Linear(input_size, output_size)
        self.conv_res = nn.Linear(input_size, output_size)

        self.af = nn.ReLU()

    def forward(self, feature):
        return self.conv2(self.af(self.conv1(self.af(feature)))) + self.conv_res(feature)

class Encoder(nn.Module):
    def __init__(self, feat_dim):
        """
        PCN based encoder
        """
        super().__init__()

        self.first_conv = nn.Sequential(
            nn.Conv1d(3, 128, 1),
            nn.BatchNorm1d(128),
            nn.ReLU(inplace=True),
            nn.Conv1d(128, 256, 1)
        )
        self.second_conv = nn.Sequential(
            nn.Conv1d(512, 512, 1),
            nn.BatchNorm1d(512),
            nn.ReLU(inplace=True),
            nn.Conv1d(512, feat_dim, 1)
        )
        self.res_conv = Linear_ResBlock(feat_dim, feat_dim)
        self.fc1 = nn.Linear(feat_dim, feat_dim)
        self.fc2 = nn.Linear(feat_dim, 1)

    def forward(self, x):
        bs, n, _ = x.shape
        feature = self.first_conv(x.transpose(2, 1))  # B 256 n
        feature_global = torch.max(feature, dim=2, keepdim=True)[0]  # B 256 1
        feature = torch.cat([feature_global.expand(-1, -1, n), feature], dim=1)  # B 512 n
        feature = self.second_conv(feature)  # B 1024 n
        feature_global = torch.max(feature, dim=2, keepdim=False)[0]  # B 1024
        feature_global = self.res_conv(feature_global)
        feature_global = self.fc1(feature_global)
        vol = self.fc2(F.relu(feature_global))
        return vol


@MODELS.register_module()
class Predictor(nn.Module):
    def __init__(self, config, **kwargs):
        super().__init__()
        self.config = config
        self.feat_dim = config.feat_dim
        self.predictor = Encoder(self.feat_dim)

        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, nn.Conv1d):
            trunc_normal_(m.weight, std=.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)

    def forward(self, pts):
        return self.predictor(pts)
