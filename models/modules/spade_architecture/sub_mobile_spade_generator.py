from torch import nn
from torch.nn import functional as F

from models.networks import BaseNetwork
from .normalization import SubMobileSPADE


class SubMobileSPADEResnetBlock(nn.Module):
    def __init__(self, fin, fout, ic, opt, config):
        super(SubMobileSPADEResnetBlock, self).__init__()
        # Attributes
        self.learned_shortcut = (fin != fout)
        self.ic = ic
        self.config = config
        channel, hidden = config['channel'], config['hidden']

        fmiddle = min(fin, fout)

        # create conv layers
        self.conv_0 = nn.Conv2d(ic, channel, kernel_size=3, padding=1)
        if self.learned_shortcut:
            self.conv_1 = nn.Conv2d(channel, channel, kernel_size=3, padding=1)
        else:
            self.conv_1 = nn.Conv2d(channel, ic, kernel_size=3, padding=1)

        if self.learned_shortcut:
            self.conv_s = nn.Conv2d(ic, channel, kernel_size=1, bias=False)

        # apply spectral norm if specified

        # define normalization layers
        spade_config_str = opt.norm_G
        self.norm_0 = SubMobileSPADE(spade_config_str, fin, opt.semantic_nc,
                                     nhidden=hidden, oc=ic)
        self.norm_1 = SubMobileSPADE(spade_config_str, fmiddle, opt.semantic_nc,
                                     nhidden=hidden, oc=channel)
        if self.learned_shortcut:
            self.norm_s = SubMobileSPADE(spade_config_str, fin, opt.semantic_nc,
                                         nhidden=hidden, oc=ic)

    # note the resnet block with SPADE also takes in |seg|,
    # the semantic segmentation map as input
    def forward(self, x, seg):
        x_s = self.shortcut(x, seg)

        dx = self.conv_0(self.actvn(self.norm_0(x, seg)))
        dx = self.conv_1(self.actvn(self.norm_1(dx, seg)))

        out = x_s + dx

        return out

    def shortcut(self, x, seg):
        if self.learned_shortcut:
            x_s = self.conv_s(self.norm_s(x, seg))
        else:
            x_s = x
        return x_s

    def actvn(self, x):
        return F.leaky_relu(x, 2e-1)


class SubMobileSPADEGenerator(BaseNetwork):
    @staticmethod
    def modify_commandline_options(parser, is_train):
        return parser

    def __init__(self, opt, config):
        super(SubMobileSPADEGenerator, self).__init__()
        self.opt = opt
        self.config = config
        nf = opt.ngf

        self.sw, self.sh = self.compute_latent_vector_size(opt)

        # downsampled segmentation map instead of random z
        channel = config['channels'][0]
        self.fc = nn.Conv2d(self.opt.semantic_nc, 16 * channel, 3, padding=1)

        ic = channel * 16
        channel = config['channels'][1]
        self.head_0 = SubMobileSPADEResnetBlock(16 * nf, 16 * nf, ic, opt,
                                                {'channel': channel * 16,
                                                 'hidden': channel * 2})

        channel = config['channels'][2]
        self.G_middle_0 = SubMobileSPADEResnetBlock(16 * nf, 16 * nf, ic, opt,
                                                    {'channel': channel * 16,
                                                     'hidden': channel * 2})

        channel = config['channels'][3]
        self.G_middle_1 = SubMobileSPADEResnetBlock(16 * nf, 16 * nf, ic, opt,
                                                    {'channel': channel * 16,
                                                     'hidden': channel * 2})

        channel = config['channels'][4]
        self.up_0 = SubMobileSPADEResnetBlock(16 * nf, 8 * nf, ic, opt,
                                              {'channel': channel * 8,
                                               'hidden': channel * 2})

        ic = channel * 8
        channel = config['channels'][5]
        self.up_1 = SubMobileSPADEResnetBlock(8 * nf, 4 * nf, ic, opt,
                                              {'channel': channel * 4,
                                               'hidden': channel * 2})
        ic = channel * 4
        channel = config['channels'][6]
        self.up_2 = SubMobileSPADEResnetBlock(4 * nf, 2 * nf, ic, opt,
                                              {'channel': channel * 2,
                                               'hidden': channel * 2})
        ic = channel * 2
        channel = config['channels'][7]
        self.up_3 = SubMobileSPADEResnetBlock(2 * nf, 1 * nf, ic, opt,
                                              {'channel': channel,
                                               'hidden': channel * 2})

        final_nc = channel

        if opt.num_upsampling_layers == 'most':
            raise NotImplementedError
        self.conv_img = nn.Conv2d(final_nc, 3, 3, padding=1)

        self.up = nn.Upsample(scale_factor=2)

    def compute_latent_vector_size(self, opt):
        if opt.num_upsampling_layers == 'normal':
            num_up_layers = 5
        elif opt.num_upsampling_layers == 'more':
            num_up_layers = 6
        elif opt.num_upsampling_layers == 'most':
            num_up_layers = 7
        else:
            raise ValueError('opt.num_upsampling_layers [%s] not recognized' %
                             opt.num_upsampling_layers)

        sw = opt.crop_size // (2 ** num_up_layers)
        sh = round(sw / opt.aspect_ratio)

        return sw, sh

    def forward(self, input, z=None):
        seg = input

        # we downsample segmap and run convolution
        x = F.interpolate(seg, size=(self.sh, self.sw))
        x = self.fc(x)

        x = self.head_0(x, seg)

        x = self.up(x)
        x = self.G_middle_0(x, seg)

        if self.opt.num_upsampling_layers == 'more' or \
                self.opt.num_upsampling_layers == 'most':
            x = self.up(x)

        x = self.G_middle_1(x, seg)

        x = self.up(x)
        x = self.up_0(x, seg)
        x = self.up(x)
        x = self.up_1(x, seg)
        x = self.up(x)
        x = self.up_2(x, seg)
        x = self.up(x)
        x = self.up_3(x, seg)

        if self.opt.num_upsampling_layers == 'most':
            x = self.up(x)
            x = self.up_4(x, seg)

        x = self.conv_img(F.leaky_relu(x, 2e-1))
        x = F.tanh(x)

        return x
