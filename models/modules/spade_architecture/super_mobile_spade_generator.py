from torch import nn
from torch.nn import functional as F

from models.modules.super_modules import SuperConv2d
from models.networks import BaseNetwork
from .normalization import SuperMobileSPADE


class SuperMobileSPADEResnetBlock(nn.Module):
    def __init__(self, fin, fout, opt):
        super(SuperMobileSPADEResnetBlock, self).__init__()
        # Attributes
        self.learned_shortcut = (fin != fout)
        fmiddle = min(fin, fout)

        # create conv layers
        self.conv_0 = SuperConv2d(fin, fmiddle, kernel_size=3, padding=1)
        self.conv_1 = SuperConv2d(fmiddle, fout, kernel_size=3, padding=1)
        if self.learned_shortcut:
            self.conv_s = SuperConv2d(fin, fout, kernel_size=1, bias=False)

        # define normalization layers
        spade_config_str = opt.norm_G
        self.norm_0 = SuperMobileSPADE(spade_config_str, fin, opt.semantic_nc, nhidden=opt.ngf * 2)
        self.norm_1 = SuperMobileSPADE(spade_config_str, fmiddle, opt.semantic_nc, nhidden=opt.ngf * 2)
        if self.learned_shortcut:
            self.norm_s = SuperMobileSPADE(spade_config_str, fin, opt.semantic_nc, nhidden=opt.ngf * 2)

    # note the resnet block with SPADE also takes in |seg|,
    # the semantic segmentation map as input
    def forward(self, x, seg, config, verbose=False):
        # if verbose:
        #     print(config)
        #     print(self.learned_shortcut)
        #     print(x.shape)
        x_s = self.shortcut(x, seg, config)

        dx = self.conv_0(self.actvn(self.norm_0(x, seg, config, verbose=verbose)), config)
        if self.learned_shortcut:
            dx = self.conv_1(self.actvn(self.norm_1(dx, seg, config)), config)
        else:
            dx = self.conv_1(self.actvn(self.norm_1(dx, seg, config)), {'channel': x.shape[1]})

        out = x_s + dx

        return out

    def shortcut(self, x, seg, config):
        if self.learned_shortcut:
            x_s = self.conv_s(self.norm_s(x, seg, config), config)
        else:
            x_s = x
        return x_s

    def actvn(self, x):
        return F.leaky_relu(x, 2e-1)


class SuperMobileSPADEGenerator(BaseNetwork):
    @staticmethod
    def modify_commandline_options(parser, is_train):
        parser.add_argument('--norm_G', type=str, default='spadesyncbatch3x3',
                            help='instance normalization or batch normalization')
        parser.add_argument('--num_upsampling_layers',
                            choices=('normal', 'more', 'most'), default='more',
                            help="If 'more', adds upsampling layer between the two middle resnet blocks. "
                                 "If 'most', also add one more upsampling + resnet layer at the end of the generator")

        return parser

    def __init__(self, opt):
        super(SuperMobileSPADEGenerator, self).__init__()
        self.opt = opt
        nf = opt.ngf

        self.sw, self.sh = self.compute_latent_vector_size(opt)

        # downsampled segmentation map instead of random z
        self.fc = SuperConv2d(self.opt.semantic_nc, 16 * nf, 3, padding=1)

        self.head_0 = SuperMobileSPADEResnetBlock(16 * nf, 16 * nf, opt)

        self.G_middle_0 = SuperMobileSPADEResnetBlock(16 * nf, 16 * nf, opt)
        self.G_middle_1 = SuperMobileSPADEResnetBlock(16 * nf, 16 * nf, opt)

        self.up_0 = SuperMobileSPADEResnetBlock(16 * nf, 8 * nf, opt)
        self.up_1 = SuperMobileSPADEResnetBlock(8 * nf, 4 * nf, opt)
        self.up_2 = SuperMobileSPADEResnetBlock(4 * nf, 2 * nf, opt)
        self.up_3 = SuperMobileSPADEResnetBlock(2 * nf, 1 * nf, opt)

        final_nc = nf

        if opt.num_upsampling_layers == 'most':
            self.up_4 = SuperMobileSPADEResnetBlock(1 * nf, nf // 2, opt)
            final_nc = nf // 2

        self.conv_img = SuperConv2d(final_nc, 3, 3, padding=1)

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

    def forward(self, input, acts=None, z=None):
        seg = input
        if acts is None:
            acts = []
        ret_acts = {}
        # we downsample segmap and run convolution
        x = F.interpolate(seg, size=(self.sh, self.sw))
        channel = self.config['channels'][0]
        x = self.fc(x, {'channel': channel * 16})
        if 'fc' in acts:
            ret_acts['fc'] = x

        channel = self.config['channels'][1]
        x = self.head_0(x, seg, {'channel': channel * 16, 'hidden': channel * 2,
                                 'calibrate_bn': self.config.get('calibrate_bn', False)})
        if 'head_0' in acts:
            ret_acts['head_0'] = x

        x = self.up(x)
        channel = self.config['channels'][2]
        x = self.G_middle_0(x, seg, {'channel': channel * 16, 'hidden': channel * 2,
                                     'calibrate_bn': self.config.get('calibrate_bn', False)})
        if 'G_middle_0' in acts:
            ret_acts['G_middle_0'] = x
        # print(x.shape)
        if self.opt.num_upsampling_layers == 'more' or \
                self.opt.num_upsampling_layers == 'most':
            x = self.up(x)

        channel = self.config['channels'][3]
        x = self.G_middle_1(x, seg, {'channel': channel * 16, 'hidden': channel * 2,
                                     'calibrate_bn': self.config.get('calibrate_bn', False)})
        if 'G_middle_1' in acts:
            ret_acts['G_middle_1'] = x
        x = self.up(x)
        channel = self.config['channels'][4]
        x = self.up_0(x, seg, {'channel': channel * 8, 'hidden': channel * 2,
                               'calibrate_bn': self.config.get('calibrate_bn', False)})
        if 'up_0' in acts:
            ret_acts['up_0'] = x
        x = self.up(x)
        channel = self.config['channels'][5]
        x = self.up_1(x, seg, {'channel': channel * 4, 'hidden': channel * 2,
                               'calibrate_bn': self.config.get('calibrate_bn', False)})
        if 'up_1' in acts:
            ret_acts['up_1'] = x
        x = self.up(x)
        channel = self.config['channels'][6]
        x = self.up_2(x, seg, {'channel': channel * 2, 'hidden': channel * 2,
                               'calibrate_bn': self.config.get('calibrate_bn', False)})
        if 'up_2' in acts:
            ret_acts['up_2'] = x
        x = self.up(x)
        channel = self.config['channels'][7]
        x = self.up_3(x, seg, {'channel': channel, 'hidden': channel * 2,
                               'calibrate_bn': self.config.get('calibrate_bn', False)})
        if 'up_3' in acts:
            ret_acts['up_3'] = x
        if self.opt.num_upsampling_layers == 'most':
            raise NotImplementedError
        x = self.conv_img(F.leaky_relu(x, 2e-1), {'channel': self.conv_img.out_channels})
        x = F.tanh(x)

        if len(acts) == 0:
            return x
        else:
            return x, ret_acts
