import torch
from torch import nn
from torch.nn import functional as F


class DenseMotion(nn.Module):
    def __init__(self, in_channels, mid_channels, norm_layer, use_bias):
        super(DenseMotion, self).__init__()
        
        self.grid_map = nn.Sequential(
            nn.Conv2d(in_channels, mid_channels, kernel_size=3, stride=1, padding=1, bias=use_bias),
            norm_layer(mid_channels),
            nn.ReLU(True),
            nn.Conv2d(mid_channels, 2, kernel_size=3, stride=1, padding=1, bias=use_bias),
            nn.Tanh()
        )

    def get_grid(self, x):
        '''
            Function for calculating grid
        '''
        grid = self.grid_map(x)
        grid = grid.permute(0, 2, 3, 1)
        return grid

    def forward(self, x):
        grid = self.get_grid(x)
        res = F.grid_sample(x, grid, align_corners=True)
        return x


class DenseMotionWithIdentity(DenseMotion):

    def get_identity(self, x):
        '''
            Function for getting identity transforming:
            x == F.grid_sample(x, grid)
            >>> True
        '''
        theta = torch.Tensor([1, 0, 0, 0, 1, 0])
        theta = theta.view(-1, 2, 3)
        grid = F.affine_grid(theta, x.size(), align_corners=True)
        grid.requires_grad_(False)
        return grid
        
    def get_grid(self, x):
        '''
            Function for calculating grid
            For this module add Identity transforming
        '''
        grid = self.grid_map(x)
        grid = grid.permute(0, 2, 3, 1)
        identity = self.get_identity(x)
        return identity + grid