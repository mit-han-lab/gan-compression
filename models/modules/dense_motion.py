import torch
from torch import nn
from torch.nn import functional as F


class DenseMotion(nn.Module):
    def __init__(self, in_channels, mid_channels, norm_layer, use_bias, use_tanh=False):
        super(DenseMotion, self).__init__()
        
        model = [
            nn.Conv2d(in_channels, mid_channels, kernel_size=3, stride=1, padding=1, bias=use_bias),
            norm_layer(mid_channels),
            nn.ReLU(True),
            nn.Conv2d(mid_channels, 2, kernel_size=3, stride=1, padding=1, bias=use_bias),
        ]
        if use_tanh:
            model += [nn.Tanh()]

        self.grid_map = nn.Sequential(*model)

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

    def __init__(self, in_channels, mid_channels, norm_layer, use_bias, use_tanh=False, h=64, w=64):
        super(DenseMotionWithIdentity, self).__init__(in_channels, mid_channels, norm_layer, use_bias, use_tanh)
        
        x_coords = 2.0 * torch.arange(h).unsqueeze(0).expand(h, w) / (h - 1.0) - 1.0
        y_coords = 2.0 * torch.arange(w).unsqueeze(1).expand(h, w) / (w - 1.0) - 1.0
        self.identity_grid = torch.stack((x_coords, y_coords), dim=0).permute(1, 2, 0).unsqueeze(0)
        self.identity_grid.requires_grad_(False)

    def get_identity(self, x):
        '''
            Function for getting identity transforming:
            x == F.grid_sample(x, grid)
            >>> True
        '''
        bs = x.size(0)
        grid = self.identity_grid.repeat(bs, 1, 1, 1)
        grid = grid.to(x.device)
        return grid
        
    def get_grid(self, x):
        '''
            Function for calculating grid
            for this module add Identity transforming
        '''
        grid = self.grid_map(x)
        grid = grid.permute(0, 2, 3, 1)
        identity = self.get_identity(x)
        return identity + grid


@torch.no_grad()
def get_only_grids(netG, input):
    """ Function for vizalization grid """
    input = input.clamp(-1, 1)
    for module in netG.model:
        if isinstance(module, DenseMotionWithIdentity):
            identity = module.get_identity(input)
            residual = module.get_grid(input) - identity
            return residual, identity
        else:
            input = module(input)
    return input