from torch import nn

from models.modules.mobile_modules import SeparableConv2d
from models.modules.munit_architecture.munit_generator import AdaptiveInstanceNorm2d, \
    Conv2dBlock, LayerNorm, MLP, StyleEncoder
from models.networks import BaseNetwork


class MobileConv2dBlock(nn.Module):
    def __init__(self, input_dim, output_dim, kernel_size, stride,
                 padding=0, norm='none', activation='relu', pad_type='zero'):
        super(MobileConv2dBlock, self).__init__()
        self.use_bias = True
        # initialize padding
        if pad_type == 'reflect':
            self.pad = nn.ReflectionPad2d(padding)
        elif pad_type == 'replicate':
            self.pad = nn.ReplicationPad2d(padding)
        elif pad_type == 'zero':
            self.pad = nn.ZeroPad2d(padding)
        else:
            assert 0, "Unsupported padding type: {}".format(pad_type)

        # initialize normalization
        norm_dim = output_dim
        if norm == 'bn':
            raise NotImplementedError
        elif norm == 'in':
            self.norm = nn.InstanceNorm2d(norm_dim)
        elif norm == 'ln':
            self.norm = LayerNorm(norm_dim)
        elif norm == 'adain':
            self.norm = AdaptiveInstanceNorm2d(norm_dim)
        elif norm == 'none' or norm == 'sn':
            self.norm = None
        else:
            assert 0, "Unsupported normalization: {}".format(norm)

        # initialize activation
        if activation == 'relu':
            self.activation = nn.ReLU(inplace=True)
        elif activation == 'lrelu':
            self.activation = nn.LeakyReLU(0.2, inplace=True)
        elif activation == 'prelu':
            self.activation = nn.PReLU()
        elif activation == 'selu':
            self.activation = nn.SELU(inplace=True)
        elif activation == 'tanh':
            self.activation = nn.Tanh()
        elif activation == 'none':
            self.activation = None
        else:
            assert 0, "Unsupported activation: {}".format(activation)

        # initialize convolution
        if norm == 'sn':
            raise NotImplementedError
        else:
            self.conv = SeparableConv2d(input_dim, output_dim, kernel_size, stride, use_bias=self.use_bias)

    def forward(self, x):
        x = self.conv(self.pad(x))
        if self.norm:
            x = self.norm(x)
        if self.activation:
            x = self.activation(x)
        return x


class SubMobileResBlock(nn.Module):
    def __init__(self, dim, mc, norm='in', activation='relu', pad_type='zero'):
        super(SubMobileResBlock, self).__init__()
        model = []
        model += [MobileConv2dBlock(dim, mc, 3, 1, 1, norm=norm, activation=activation, pad_type=pad_type)]
        model += [MobileConv2dBlock(mc, dim, 3, 1, 1, norm=norm, activation='none', pad_type=pad_type)]
        self.model = nn.Sequential(*model)

    def forward(self, x):
        residual = x
        out = self.model(x)
        out += residual
        return out


class SubMobileResBlocks(nn.Module):
    def __init__(self, num_blocks, dim, norm='in', activation='relu', pad_type='zero', configs=None):
        super(SubMobileResBlocks, self).__init__()
        self.num_blocks = num_blocks
        self.model = []
        for i in range(num_blocks):
            self.model += [SubMobileResBlock(dim, mc=configs['channels'][i // 2], norm=norm,
                                             activation=activation, pad_type=pad_type)]
        self.model = nn.Sequential(*self.model)

    def forward(self, x):
        return self.model(x)


class SubMobileContentEncoder(nn.Module):
    def __init__(self, n_downsample, n_res, input_dim, dim, norm, activ, pad_type, configs):
        super(SubMobileContentEncoder, self).__init__()
        self.n_downsample = n_downsample
        self.n_res = n_res
        self.model = []
        self.model += [Conv2dBlock(input_dim, configs['channels'][0], 7, 1, 3,
                                   norm=norm, activation=activ, pad_type=pad_type)]
        ic = configs['channels'][0]
        # downsampling blocks
        for i in range(n_downsample):
            oc = configs['channels'][i + 1] * (2 ** (i + 1))
            self.model += [Conv2dBlock(ic, oc, 4, 2, 1, norm=norm, activation=activ, pad_type=pad_type)]
            ic = oc

        # residual blocks
        resblock_channels = []
        for c in configs['channels'][-(self.n_res // 2):]:
            resblock_channels.append(c * (2 ** n_downsample))
        self.model += [SubMobileResBlocks(n_res, ic, norm=norm, activation=activ, pad_type=pad_type,
                                          configs={'channels': resblock_channels})]
        self.model = nn.Sequential(*self.model)
        self.output_dim = ic

    def forward(self, x):
        return self.model(x)


class SubMobileDecoder(nn.Module):
    def __init__(self, n_upsample, n_res, dim, output_dim, res_norm='adain',
                 activ='relu', pad_type='zero', configs=None):
        super(SubMobileDecoder, self).__init__()
        self.n_upsample = n_upsample
        self.n_res = n_res
        self.output_dim = output_dim
        self.model = []
        # AdaIN residual blocks
        self.model += [SubMobileResBlocks(n_res, dim, res_norm, activ, pad_type=pad_type,
                                          configs={'channels': configs['channels'][:self.n_res // 2]})]
        ic = dim
        # upsampling blocks
        for i in range(n_upsample):
            oc = configs['channels'][self.n_res // 2 + i] // (2 ** (i + 1))
            self.model += [nn.Upsample(scale_factor=2),
                           MobileConv2dBlock(ic, oc, 5, 1, 2, norm='ln',
                                             activation=activ, pad_type=pad_type)]
            ic = oc
        # use reflection padding in the last conv layer
        self.model += [Conv2dBlock(ic, output_dim, 7, 1, 3, norm='none', activation='tanh', pad_type=pad_type)]
        self.model = nn.Sequential(*self.model)

    def forward(self, x):
        return self.model(x)


class SubMobileAdaINGenerator(BaseNetwork):
    # AdaIN auto-encoder architecture
    def __init__(self, opt, configs):
        super(SubMobileAdaINGenerator, self).__init__()
        self.opt = opt
        input_dim = opt.input_nc
        dim = opt.ngf
        style_dim = opt.style_dim
        n_downsample = opt.n_downsample
        n_res = opt.n_res
        activ = opt.activ
        pad_type = opt.pad_type
        mlp_dim = opt.mlp_dim
        no_style_encoder = opt.no_style_encoder

        if not no_style_encoder:
            self.enc_style = StyleEncoder(4, input_dim, configs['channels'][0], style_dim,
                                          norm='none', activ=activ, pad_type=pad_type)

        # content encoder
        content_encoder_channels = []
        for i in range(1, self.opt.n_downsample + (self.opt.n_res // 2) + 2):
            content_encoder_channels.append(configs['channels'][i])
        self.enc_content = SubMobileContentEncoder(n_downsample, n_res, input_dim, dim, 'in', activ,
                                                   pad_type=pad_type, configs={'channels': content_encoder_channels})
        decoder_channels = []
        offset = self.opt.n_downsample + (self.opt.n_res // 2) + 2
        base = 2 ** self.opt.n_downsample
        for i in range(offset, offset + self.opt.n_downsample + (self.opt.n_res // 2)):
            decoder_channels.append(base * configs['channels'][i])
        self.dec = SubMobileDecoder(n_downsample, n_res, self.enc_content.output_dim,
                                    input_dim, res_norm='adain', activ=activ, pad_type=pad_type,
                                    configs={'channels': decoder_channels})

        # MLP to generate AdaIN parameters
        self.mlp = MLP(style_dim, self.get_num_adain_params(self.dec), mlp_dim, 3, norm='none', activ=activ)

    def forward(self, images, z=None):
        # reconstruct an image
        if z is None:
            content, style_fake = self.encode(images)
        else:
            content, _ = self.encode(images, need_style=False)
            style_fake = z
        images_recon = self.decode(content, style_fake)
        return images_recon

    def encode(self, images, need_content=True, need_style=True):
        # encode an image to its content and style codes
        if need_style:
            assert hasattr(self, 'enc_style')
            style_fake = self.enc_style(images)
        else:
            style_fake = None
        if need_content:
            content = self.enc_content(images)
        else:
            content = None
        return content, style_fake

    def decode(self, content, style):
        # decode content and style codes to an image
        adain_params = self.mlp(style)
        self.assign_adain_params(adain_params, self.dec)
        images = self.dec(content)
        return images

    def assign_adain_params(self, adain_params, model):
        # assign the adain_params to the AdaIN layers in model
        for m in model.modules():
            if "AdaptiveInstanceNorm2d" in m.__class__.__name__:
                mean = adain_params[:, :m.num_features]
                std = adain_params[:, m.num_features:2 * m.num_features]
                m.bias = mean.contiguous().view(-1)
                m.weight = std.contiguous().view(-1)
                if adain_params.size(1) > 2 * m.num_features:
                    adain_params = adain_params[:, 2 * m.num_features:]

    def get_num_adain_params(self, model):
        # return the number of AdaIN parameters needed by the model
        num_adain_params = 0
        for m in model.modules():
            if "AdaptiveInstanceNorm2d" in m.__class__.__name__:
                num_adain_params += 2 * m.num_features
        return num_adain_params
