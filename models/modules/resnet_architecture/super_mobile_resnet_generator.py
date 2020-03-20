import functools

from torch import nn

from models.modules.super_modules import SuperConvTranspose2d, SuperConv2d, SuperSeparableConv2d

scale_factor = 2


class SuperMobileResnetGenerator(nn.Module):
    def __init__(self, input_nc, output_nc, ngf, norm_layer=nn.BatchNorm2d, dropout_rate=0, n_blocks=6,
                 padding_type='reflect'):
        assert n_blocks >= 0
        super(SuperMobileResnetGenerator, self).__init__()
        if type(norm_layer) == functools.partial:
            use_bias = norm_layer.func == nn.InstanceNorm2d
        else:
            use_bias = norm_layer == nn.InstanceNorm2d

        model = [nn.ReflectionPad2d(3),
                 SuperConv2d(input_nc, ngf, kernel_size=7, padding=0, bias=use_bias),
                 norm_layer(ngf),
                 nn.ReLU(True)]

        n_downsampling = 2
        for i in range(n_downsampling):  # add downsampling layers
            mult = 2 ** i
            model += [SuperConv2d(ngf * mult, ngf * mult * 2, kernel_size=3, stride=2, padding=1, bias=use_bias),
                      norm_layer(ngf * mult * 2),
                      nn.ReLU(True)]

        mult = 2 ** n_downsampling

        n_blocks1 = n_blocks // 3
        n_blocks2 = n_blocks1
        n_blocks3 = n_blocks - n_blocks1 - n_blocks2

        for i in range(n_blocks1):
            model += [SuperMobileResnetBlock(ngf * mult, padding_type=padding_type, norm_layer=norm_layer,
                                             dropout_rate=dropout_rate,
                                             use_bias=use_bias)]

        for i in range(n_blocks2):
            model += [SuperMobileResnetBlock(ngf * mult, padding_type=padding_type, norm_layer=norm_layer,
                                             dropout_rate=dropout_rate,
                                             use_bias=use_bias)]

        for i in range(n_blocks3):
            model += [SuperMobileResnetBlock(ngf * mult, padding_type=padding_type, norm_layer=norm_layer,
                                             dropout_rate=dropout_rate,
                                             use_bias=use_bias)]

        for i in range(n_downsampling):  # add upsampling layers
            mult = 2 ** (n_downsampling - i)
            model += [SuperConvTranspose2d(ngf * mult, int(ngf * mult / 2),
                                           kernel_size=3, stride=2,
                                           padding=1, output_padding=1,
                                           bias=use_bias),
                      norm_layer(int(ngf * mult / 2)),
                      nn.ReLU(True)]
        model += [nn.ReflectionPad2d(3)]
        model += [SuperConv2d(ngf, output_nc, kernel_size=7, padding=0)]
        model += [nn.Tanh()]
        self.model = nn.Sequential(*model)

    def forward(self, input):
        """Standard forward"""
        configs = self.configs
        input = input.clamp(-1, 1)
        x = input
        cnt = 0
        for i in range(0, 10):
            module = self.model[i]
            if isinstance(module, SuperConv2d):
                channel = configs['channels'][cnt] * (2 ** cnt)
                config = {'channel': channel}
                x = module(x, config)
                cnt += 1
            else:
                x = module(x)
        for i in range(3):
            for j in range(10 + i * 3, 13 + i * 3):
                if len(configs['channels']) == 6:
                    channel = configs['channels'][3] * 4
                else:
                    channel = configs['channels'][i + 3] * 4
                config = {'channel': channel}
                module = self.model[j]
                x = module(x, config)
        cnt = 2
        for i in range(19, 28):
            module = self.model[i]
            if isinstance(module, SuperConvTranspose2d):
                cnt -= 1
                if len(configs['channels']) == 6:
                    channel = configs['channels'][5 - cnt] * (2 ** cnt)
                else:
                    channel = configs['channels'][7 - cnt] * (2 ** cnt)
                config = {'channel': channel}
                x = module(x, config)
            elif isinstance(module, SuperConv2d):
                config = {'channel': module.out_channels}
                x = module(x, config)
            else:
                x = module(x)
        return x


class SuperMobileResnetBlock(nn.Module):
    def __init__(self, dim, padding_type, norm_layer, dropout_rate, use_bias):
        super(SuperMobileResnetBlock, self).__init__()
        self.conv_block = self.build_conv_block(dim, padding_type, norm_layer, dropout_rate, use_bias)

    def build_conv_block(self, dim, padding_type, norm_layer, dropout_rate, use_bias):
        conv_block = []
        p = 0
        if padding_type == 'reflect':
            conv_block += [nn.ReflectionPad2d(1)]
        elif padding_type == 'replicate':
            conv_block += [nn.ReplicationPad2d(1)]
        elif padding_type == 'zero':
            p = 1
        else:
            raise NotImplementedError('padding [%s] is not implemented' % padding_type)

        conv_block += [
            SuperSeparableConv2d(in_channels=dim, out_channels=dim,
                                 kernel_size=3, padding=p, stride=1),
            norm_layer(dim),
            nn.ReLU(True)
        ]
        conv_block += [nn.Dropout(dropout_rate)]

        p = 0
        if padding_type == 'reflect':
            conv_block += [nn.ReflectionPad2d(1)]
        elif padding_type == 'replicate':
            conv_block += [nn.ReplicationPad2d(1)]
        elif padding_type == 'zero':
            p = 1
        else:
            raise NotImplementedError('padding [%s] is not implemented' % padding_type)

        conv_block += [
            SuperSeparableConv2d(in_channels=dim, out_channels=dim,
                                 kernel_size=3, padding=p, stride=1),
            norm_layer(dim)
        ]

        return nn.Sequential(*conv_block)

    def forward(self, input, config):
        x = input
        cnt = 0
        for module in self.conv_block:
            if isinstance(module, SuperSeparableConv2d):
                if cnt == 1:
                    config['channel'] = input.size(1)
                x = module(x, config)
                cnt += 1
            else:
                x = module(x)
        out = input + x
        return out
