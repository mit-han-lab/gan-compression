import argparse
import os

import torch
from torch import nn

from configs import decode_config
from utils.util import load_network


def transfer(mA, mB):
    weight = mA.weight.data
    a, b, c, d = mB.weight.data.shape
    mB.weight.data = weight[:a, :b, :c, :d]
    if mB.bias is not None:
        a = mB.bias.data.shape[0]
        mB.bias.data = mA.bias.data[:a]


def transfer_weight(netA, netB):  # netA -> netB
    for nA, mA in netA.named_modules():
        if isinstance(mA, (nn.ConvTranspose2d, nn.Conv2d)):
            for nB, mB in netB.named_modules():
                if nA == nB:
                    transfer(mA, mB)


def main(opt):
    if opt.model == 'mobile_resnet':
        from models.modules.resnet_architecture.mobile_resnet_generator import MobileResnetGenerator as SuperModel
        from models.modules.resnet_architecture.sub_mobile_resnet_generator import SubMobileResnetGenerator as SubModel
    elif opt.model == 'mobile_spade':
        # TODO
        raise NotImplementedError
    else:
        raise NotImplementedError('Unknown architecture [%s]!' % opt.model)

    config = decode_config(opt.config_str)

    input_nc, output_nc = opt.input_nc, opt.output_nc
    super_model = SuperModel(input_nc, output_nc, ngf=opt.ngf, norm_layer=nn.InstanceNorm2d, n_blocks=9)
    sub_model = SubModel(input_nc, output_nc, config=config, norm_layer=nn.InstanceNorm2d, n_blocks=9)

    load_network(super_model, opt.input_path)
    transfer_weight(super_model, sub_model)

    output_dir = os.path.dirname(opt.output_path)
    os.makedirs(output_dir, exist_ok=True)
    torch.save(sub_model.state_dict(), opt.output_path)
    print('Successfully export the subnet at [%s].' % opt.output_path)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Export a specific subnet from a supernet')
    parser.add_argument('--model', type=str, default='mobile_resnet', choices=['mobile_resnet', 'mobile_spade'],
                        help='specify the model type you want to export')
    parser.add_argument('--ngf', type=int, default=48, help='the base number of filters of the generator')
    parser.add_argument('--input_path', type=str, required=True, help='the input model path')
    parser.add_argument('--output_path', type=str, required=True, help='the path to the exported model')
    parser.add_argument('--config_str', type=str, default=None,
                        help='the configuration string for a specific subnet in the supernet')
    parser.add_argument('--input_nc', type=int, default=3,
                        help='# of input image channels: 3 for RGB and 1 for grayscale')
    parser.add_argument('--output_nc', type=int, default=3,
                        help='# of output image channels: 3 for RGB and 1 for grayscale')
    opt = parser.parse_args()
    main(opt)
