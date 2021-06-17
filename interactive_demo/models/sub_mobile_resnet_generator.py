import functools

from torch import nn

from models.mobile_modules import SeparableConv2d


class MobileResnetBlock(nn.Module):
    def __init__(self, ic, oc, padding_type, norm_layer, dropout_rate, use_bias):
        super(MobileResnetBlock, self).__init__()
        self.conv_block = self.build_conv_block(ic, oc, padding_type, norm_layer, dropout_rate, use_bias)

    def build_conv_block(self, ic, oc, padding_type, norm_layer, dropout_rate, use_bias):
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
            SeparableConv2d(in_channels=ic, out_channels=oc,
                            kernel_size=3, padding=p, stride=1),
            norm_layer(oc),
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
            SeparableConv2d(in_channels=oc, out_channels=ic,
                            kernel_size=3, padding=p, stride=1),
            norm_layer(oc)
        ]

        return nn.Sequential(*conv_block)

    def forward(self, x):
        out = x + self.conv_block(x)
        return out


class SubMobileResnetGenerator(nn.Module):
    def __init__(self, input_nc, output_nc, config, norm_layer=nn.BatchNorm2d,
                 dropout_rate=0, n_blocks=9, padding_type='reflect'):
        assert n_blocks >= 0
        super(SubMobileResnetGenerator, self).__init__()
        if type(norm_layer) == functools.partial:
            use_bias = norm_layer.func == nn.InstanceNorm2d
        else:
            use_bias = norm_layer == nn.InstanceNorm2d

        model = [nn.ReflectionPad2d(3),
                 nn.Conv2d(input_nc, config['channels'][0], kernel_size=7, padding=0, bias=use_bias),
                 norm_layer(config['channels'][0]),
                 nn.ReLU(True)]

        n_downsampling = 2
        for i in range(n_downsampling):  # add downsampling layers

            mult = 2 ** i
            ic = config['channels'][i]
            oc = config['channels'][i + 1]
            model += [nn.Conv2d(ic * mult, oc * mult * 2, kernel_size=3, stride=2, padding=1, bias=use_bias),
                      norm_layer(ic * mult * 2),
                      nn.ReLU(True)]

        mult = 2 ** n_downsampling

        ic = config['channels'][2]
        # oc = config['channels'][3]
        # for i in range(n_blocks):
        #     oc = config['channels'][i + 3]
        #     model += [MobileResnetBlock(ic * mult, oc * mult, padding_type=padding_type, norm_layer=norm_layer,
        #                                 dropout_rate=dropout_rate, use_bias=use_bias)]
        for i in range(n_blocks):
            if len(config['channels']) == 6:
                offset = 0
            else:
                offset = i // 3
            oc = config['channels'][offset + 3]
            model += [MobileResnetBlock(ic * mult, oc * mult, padding_type=padding_type, norm_layer=norm_layer,
                                        dropout_rate=dropout_rate, use_bias=use_bias)]

        if len(config['channels']) == 6:
            offset = 4
        else:
            offset = 6
        for i in range(n_downsampling):  # add upsampling layers
            oc = config['channels'][offset + i]
            # print(oc)
            mult = 2 ** (n_downsampling - i)
            model += [nn.ConvTranspose2d(ic * mult, int(oc * mult / 2),
                                         kernel_size=3, stride=2,
                                         padding=1, output_padding=1,
                                         bias=use_bias),
                      norm_layer(int(oc * mult / 2)),
                      nn.ReLU(True)]
            ic = oc
        model += [nn.ReflectionPad2d(3)]
        model += [nn.Conv2d(ic, output_nc, kernel_size=7, padding=0)]
        model += [nn.Tanh()]
        self.model = nn.Sequential(*model)

    def forward(self, input):
        input = input.clamp(-1, 1)
        return self.model(input)
