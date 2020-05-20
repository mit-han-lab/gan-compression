import torch
from torch import nn
from torch.nn import functional as F

class SpatialTransform(nn.Module):
    def __init__(self, in_features,
                 mid_features=256):
        super(SpatialTransform, self).__init__()
        self.use_bias = True               # !!!
        self.in_features = in_features
        self.mid_features = mid_features
        
        self.localization = nn.Sequential(
            nn.Linear(in_features, mid_features, bias=self.use_bias),
            nn.ReLU(True),
            nn.Linear(mid_features, 6, bias=self.use_bias),
        )

    def forward(self, x):
        print('-' * 30)
        print(x.shape)
        xs = x.view(-1, self.in_features)
        theta = self.localization(xs)
        theta = theta.view(-1, 2, 3)
        print(theta)
        grid = F.affine_grid(theta, x.size())
        x = F.grid_sample(x, grid)
        print(x.shape)
        print('-' * 30)
        return x
