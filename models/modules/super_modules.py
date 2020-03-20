import torch.nn as nn
from torch.nn import functional as F


class SuperConv2d(nn.Conv2d):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1,
                 padding=0, dilation=1, groups=1, bias=True, padding_mode='zeros'):
        super(SuperConv2d, self).__init__(in_channels, out_channels, kernel_size,
                                          stride, padding, dilation, groups, bias, padding_mode)

    def forward(self, x, config):
        in_nc = x.size(1)
        out_nc = config['channel']
        weight = self.weight[:out_nc, :in_nc]  # [oc, ic, H, W]
        if self.bias is not None:
            bias = self.bias[:out_nc]
        else:
            bias = None
        return F.conv2d(x, weight, bias, self.stride, self.padding, self.dilation, self.groups)


class SuperConvTranspose2d(nn.ConvTranspose2d):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1,
                 padding=0, output_padding=0, groups=1, bias=True,
                 dilation=1, padding_mode='zeros'):
        super(SuperConvTranspose2d, self).__init__(in_channels, out_channels,
                                                   kernel_size, stride, padding,
                                                   output_padding, groups, bias,
                                                   dilation, padding_mode)

    def forward(self, x, config, output_size=None):
        output_padding = self._output_padding(x, output_size, self.stride, self.padding, self.kernel_size)
        in_nc = x.size(1)
        out_nc = config['channel']
        weight = self.weight[:in_nc, :out_nc]  # [ic, oc, H, W]
        if self.bias is not None:
            bias = self.bias[:out_nc]
        else:
            bias = None
        return F.conv_transpose2d(x, weight, bias, self.stride, self.padding,
                                  output_padding, self.groups, self.dilation)


class SuperSeparableConv2d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, norm_layer=nn.InstanceNorm2d,
                 use_bias=True, scale_factor=1):
        super(SuperSeparableConv2d, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels=in_channels, out_channels=in_channels * scale_factor, kernel_size=kernel_size,
                      stride=stride, padding=padding, groups=in_channels, bias=use_bias),
            norm_layer(in_channels),
            nn.Conv2d(in_channels=in_channels * scale_factor, out_channels=out_channels,
                      kernel_size=1, stride=1, bias=use_bias),
        )

    def forward(self, x, config):
        in_nc = x.size(1)
        out_nc = config['channel']

        conv = self.conv[0]
        assert isinstance(conv, nn.Conv2d)
        weight = conv.weight[:in_nc]  # [oc, 1, H, W]
        # print(weight.shape)
        if conv.bias is not None:
            bias = conv.bias[:in_nc]
        else:
            bias = None
        x = F.conv2d(x, weight, bias, conv.stride, conv.padding, conv.dilation, in_nc)

        x = self.conv[1](x)

        conv = self.conv[2]
        assert isinstance(conv, nn.Conv2d)
        weight = conv.weight[:out_nc, :in_nc]  # [oc, ic, H, W]
        # print(weight.shape)
        if conv.bias is not None:
            bias = conv.bias[:out_nc]
        else:
            bias = None
        x = F.conv2d(x, weight, bias, conv.stride, conv.padding, conv.dilation, conv.groups)
        return x
