import argparse
import copy
import os

import torch

from models import networks
from utils.util import load_network


def create_opt(old_opt, role):
    assert role in ['A', 'B']
    new_opt = copy.deepcopy(old_opt)
    new_opt.ngf = getattr(old_opt, 'ngf_%s' % role)
    new_opt.no_style_encoder = getattr(old_opt, 'no_style_encoder_%s' % role)
    return new_opt


def main(opt):
    netG_A = networks.define_G(opt.netG_A, opt=create_opt(opt, 'A'))
    netG_B = networks.define_G(opt.netG_B, opt=create_opt(opt, 'B'))
    if opt.direction == 'AtoB':
        load_network(netG_A, opt.restore_G_A_path)
        load_network(netG_B, opt.restore_G_B_path)
    else:
        load_network(netG_A, opt.restore_G_B_path)
        load_network(netG_B, opt.restore_G_A_path)
    netG = copy.deepcopy(netG_B)
    netG.enc_content = netG_A.enc_content
    if opt.decoder_from_A:
        netG.dec = netG_A.dec
        netG.mlp = netG_A.mlp
    os.makedirs(os.path.dirname(opt.output_path), exist_ok=True)
    torch.save(netG.state_dict(), opt.output_path)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--netG_A', type=str, default='munit',
                        help='the architecture of the generator A')
    parser.add_argument('--netG_B', type=str, default='munit',
                        help='the architecture of the generator B')
    parser.add_argument('--restore_G_A_path', type=str, required=True,
                        help='path to the restore the generator A')
    parser.add_argument('--restore_G_B_path', type=str, required=True,
                        help='path to the restore the generator B')
    parser.add_argument('--input_nc', type=int, default=3,
                        help='# of input image channels: 3 for RGB and 1 for grayscale')
    parser.add_argument('--output_nc', type=int, default=3,
                        help='# of output image channels: 3 for RGB and 1 for grayscale')
    parser.add_argument('--ngf_A', type=int, default=64, help='the base number of generator A filters')
    parser.add_argument('--ngf_B', type=int, default=64, help='the base number of generator B filters')
    parser.add_argument('--direction', type=str, default='AtoB',
                        help='the direction of the generation')
    parser.add_argument('--output_path', type=str, required=True,
                        help='path to export the assembled model')
    parser.add_argument('--style_dim', type=int, default=8,
                        help='the dimension of the style vector')
    parser.add_argument('--n_downsample', type=int, default=2,
                        help='the number of downsample layer in the generator')
    parser.add_argument('--n_res', type=int, default=4,
                        help='the number of the ResBlock in the generator')
    parser.add_argument('--activ', type=str, default='relu',
                        help='the activation type of the generator')
    parser.add_argument('--pad_type', type=str, default='reflect',
                        help='the padding type of the generator and the discriminator')
    parser.add_argument('--mlp_dim', type=int, default=256,
                        help='the dimension of the mlp layer in the generator')
    parser.add_argument('--no_style_encoder_A', action='store_true',
                        help='whether to have the style encoder in the generator')
    parser.add_argument('--no_style_encoder_B', action='store_true',
                        help='whether to have the style encoder in the generator')
    parser.add_argument('--decoder_from_A', action='store_true',
                        help='is the decoder in the exported model from A')
    opt = parser.parse_args()
    main(opt)
