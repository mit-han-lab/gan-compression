import argparse

from torch import nn
from torch.nn import functional as F
from torch.nn.utils import remove_spectral_norm
from torch.nn.utils import spectral_norm

from models.networks import BaseNetwork
from .normalization import MobileSPADE


class MobileSPADEResnetBlock(nn.Module):
    def __init__(self, fin, fout, opt):
        super(MobileSPADEResnetBlock, self).__init__()
        # Attributes
        self.learned_shortcut = (fin != fout)
        fmiddle = min(fin, fout)

        # create conv layers
        self.conv_0 = nn.Conv2d(fin, fmiddle, kernel_size=3, padding=1)
        self.conv_1 = nn.Conv2d(fmiddle, fout, kernel_size=3, padding=1)
        if self.learned_shortcut:
            self.conv_s = nn.Conv2d(fin, fout, kernel_size=1, bias=False)
        if 'spectral' in opt.norm_G:
            self.conv_0 = spectral_norm(self.conv_0)
            self.conv_1 = spectral_norm(self.conv_1)
            if self.learned_shortcut:
                self.conv_s = spectral_norm(self.conv_s)
        # apply spectral norm if specified

        # define normalization layers
        spade_config_str = opt.norm_G.replace('spectral', '')
        self.norm_0 = MobileSPADE(spade_config_str, fin, opt.semantic_nc,
                                  nhidden=opt.ngf * 2, separable_conv_norm=opt.separable_conv_norm)
        self.norm_1 = MobileSPADE(spade_config_str, fmiddle, opt.semantic_nc,
                                  nhidden=opt.ngf * 2, separable_conv_norm=opt.separable_conv_norm)
        if self.learned_shortcut:
            self.norm_s = MobileSPADE(spade_config_str, fin, opt.semantic_nc,
                                      nhidden=opt.ngf * 2, separable_conv_norm=opt.separable_conv_norm)

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

    def remove_spectral_norm(self):
        self.conv_0 = remove_spectral_norm(self.conv_0)
        self.conv_1 = remove_spectral_norm(self.conv_1)
        if self.learned_shortcut:
            self.conv_s = remove_spectral_norm(self.conv_s)


class MobileSPADEGenerator(BaseNetwork):
    @staticmethod
    def modify_commandline_options(parser, is_train):
        assert isinstance(parser, argparse.ArgumentParser)
        return parser

    def __init__(self, opt):
        super(MobileSPADEGenerator, self).__init__()
        self.opt = opt
        nf = opt.ngf

        self.sw, self.sh = self.compute_latent_vector_size(opt)

        # downsampled segmentation map instead of random z
        self.fc = nn.Conv2d(self.opt.semantic_nc, 16 * nf, 3, padding=1)

        self.head_0 = MobileSPADEResnetBlock(16 * nf, 16 * nf, opt)

        self.G_middle_0 = MobileSPADEResnetBlock(16 * nf, 16 * nf, opt)
        self.G_middle_1 = MobileSPADEResnetBlock(16 * nf, 16 * nf, opt)

        self.up_0 = MobileSPADEResnetBlock(16 * nf, 8 * nf, opt)
        self.up_1 = MobileSPADEResnetBlock(8 * nf, 4 * nf, opt)
        self.up_2 = MobileSPADEResnetBlock(4 * nf, 2 * nf, opt)
        self.up_3 = MobileSPADEResnetBlock(2 * nf, 1 * nf, opt)

        final_nc = nf

        if opt.num_upsampling_layers == 'most':
            self.up_4 = MobileSPADEResnetBlock(1 * nf, nf // 2, opt)
            final_nc = nf // 2

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

    def forward(self, input, mapping_layers=[]):
        seg = input

        ret_acts = {}

        # we downsample segmap and run convolution
        x = F.interpolate(seg, size=(self.sh, self.sw))
        x = self.fc(x)

        if 'fc' in mapping_layers:
            ret_acts['fc'] = x

        x = self.head_0(x, seg)
        if 'head_0' in mapping_layers:
            ret_acts['head_0'] = x

        x = self.up(x)
        x = self.G_middle_0(x, seg)

        if 'G_middle_0' in mapping_layers:
            ret_acts['G_middle_0'] = x

        if self.opt.num_upsampling_layers == 'more' or \
                self.opt.num_upsampling_layers == 'most':
            x = self.up(x)

        x = self.G_middle_1(x, seg)
        if 'G_middle_1' in mapping_layers:
            ret_acts['G_middle_1'] = x

        x = self.up(x)
        x = self.up_0(x, seg)
        if 'up_0' in mapping_layers:
            ret_acts['up_0'] = x

        x = self.up(x)
        x = self.up_1(x, seg)
        if 'up_1' in mapping_layers:
            ret_acts['up_1'] = x

        x = self.up(x)
        x = self.up_2(x, seg)
        if 'up_2' in mapping_layers:
            ret_acts['up_2'] = x

        x = self.up(x)
        x = self.up_3(x, seg)
        if 'up_3' in mapping_layers:
            ret_acts['up_3'] = x

        if self.opt.num_upsampling_layers == 'most':
            x = self.up(x)
            x = self.up_4(x, seg)
            if 'up_4' in mapping_layers:
                ret_acts['up_4'] = x

        x = self.conv_img(F.leaky_relu(x, 2e-1))
        x = F.tanh(x)
        if len(mapping_layers) == 0:
            return x
        else:
            return x, ret_acts

    def remove_spectral_norm(self):
        self.head_0.remove_spectral_norm()
        self.G_middle_0.remove_spectral_norm()
        self.G_middle_1.remove_spectral_norm()

        self.up_0.remove_spectral_norm()
        self.up_1.remove_spectral_norm()
        self.up_2.remove_spectral_norm()
        self.up_3.remove_spectral_norm()

        if self.opt.num_upsampling_layers == 'most':
            self.up_4.remove_spectral_norm()
