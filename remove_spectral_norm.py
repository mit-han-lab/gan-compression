import argparse
import os

import torch

from models import networks
from utils import util


def main(opt):
    # define the generator with spectral normalization. Only the last argument counts
    netG = networks.define_G(None, None, None, opt.netG, opt=opt)
    util.load_network(netG, opt.restore_G_path, True)
    print(netG)
    netG.remove_spectral_norm()
    dirname = os.path.dirname(opt.output_path)
    os.makedirs(dirname, exist_ok=True)
    torch.save(netG.cpu().state_dict(), opt.output_path)
    print('Successfully export the model at [%s]!' % opt.output_path)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--netG', type=str, default='mobile_spade',
                        choices=['spade', 'mobile_spade'],
                        help='specify generator architecture')
    parser.add_argument('--ngf', type=int, default=64, help='the base number of filters of the student generator')
    parser.add_argument('--restore_G_path', type=str, required=True, help='the path to restore the generator')
    parser.add_argument('--output_path', type=str, required=True, help='the path to export the generator')
    parser.add_argument('--separable_conv_norm', type=str, default='instance',
                        choices=('none', 'instance', 'batch'),
                        help='whether to use instance norm for the separable convolutions')
    parser.add_argument('--num_upsampling_layers',
                        choices=('normal', 'more', 'most'), default='more',
                        help="If 'more', adds upsampling layer between the two middle resnet blocks. "
                             "If 'most', also add one more upsampling + resnet layer at the end of the generator")
    parser.add_argument('--crop_size', type=int, default=512, help='then crop to this size')
    parser.add_argument('--aspect_ratio', type=float, default=2.0,
                        help='The ratio width/height. The final height of the load image will be crop_size/aspect_ratio')
    parser.add_argument('--input_nc', type=int, default=35,
                        help='# of input image channels: 3 for RGB and 1 for grayscale')
    parser.add_argument('--output_nc', type=int, default=3,
                        help='# of output image channels: 3 for RGB and 1 for grayscale')
    parser.add_argument('--no_instance', action='store_true',
                        help='if specified, do *not* add instance map as input')
    parser.add_argument('--contain_dontcare_label', action='store_true',
                        help='if the label map contains dontcare label (dontcare=255)')
    opt = parser.parse_args()
    opt.norm_G = 'spectralspadesyncbatch3x3'
    opt.semantic_nc = opt.input_nc + (1 if opt.contain_dontcare_label else 0) + (0 if opt.no_instance else 1)

    main(opt)
