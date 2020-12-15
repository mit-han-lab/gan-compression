import torch
from torch import nn
from torch.nn import functional as F

from models.modules.sync_batchnorm import SynchronizedBatchNorm2d
from models.modules.sync_batchnorm.batchnorm import _ChildMessage, _sum_ft, _unsqueeze_ft


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
            norm_layer(in_channels * scale_factor),
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


class SuperSynchronizedBatchNorm2d(SynchronizedBatchNorm2d):
    def __init__(self, num_features, eps=1e-5, momentum=0.1, affine=True):
        super(SuperSynchronizedBatchNorm2d, self).__init__(num_features, eps, momentum, affine)

    def forward(self, x, config={'calibrate_bn': False}):
        # If it is not parallel computation or is in evaluation mode, use PyTorch's implementation.
        # return input
        input = x
        if x.shape[1] != self.num_features:
            padding = torch.zeros([x.shape[0], self.num_features - x.shape[1], x.shape[2], x.shape[3]], device=x.device)
            input = torch.cat([input, padding], dim=1)
        calibrate_bn = config['calibrate_bn']
        if not (self._is_parallel and self.training):
            if calibrate_bn:
                ret = F.batch_norm(
                    input, self.running_mean, self.running_var, self.weight, self.bias,
                    self.training, 1, self.eps)
            else:
                ret = F.batch_norm(
                    input, self.running_mean, self.running_var, self.weight, self.bias,
                    self.training, self.momentum, self.eps)
            return ret[:, :x.shape[1]]

        momentum = self.momentum
        if calibrate_bn:
            self.momentum = 1
        # print('another route')

        # Resize the input to (B, C, -1).
        input_shape = input.size()
        input = input.view(input.size(0), self.num_features, -1)

        # Compute the sum and square-sum.
        sum_size = input.size(0) * input.size(2)
        input_sum = _sum_ft(input)
        input_ssum = _sum_ft(input ** 2)

        # Reduce-and-broadcast the statistics.
        if self._parallel_id == 0:
            mean, inv_std = self._sync_master.run_master(_ChildMessage(input_sum, input_ssum, sum_size))
        else:
            mean, inv_std = self._slave_pipe.run_slave(_ChildMessage(input_sum, input_ssum, sum_size))

        # Compute the output.
        if self.affine:
            # MJY:: Fuse the multiplication for speed.
            output = (input - _unsqueeze_ft(mean)) * _unsqueeze_ft(inv_std * self.weight) + _unsqueeze_ft(self.bias)
        else:
            output = (input - _unsqueeze_ft(mean)) * _unsqueeze_ft(inv_std)

        # Reshape it.
        if calibrate_bn:
            self.momentum = momentum
        output = output.view(input_shape)
        return output[:, :x.shape[1]]
