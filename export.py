import argparse
import os
from typing import Union

import torch
from torch import nn

from configs import decode_config
from models.modules.munit_architecture.munit_generator import LayerNorm
from models.modules.munit_architecture.sub_mobile_munit_generator import SubMobileAdaINGenerator
from models.modules.munit_architecture.super_mobile_munit_generator import SuperMobileAdaINGenerator
from models.modules.munit_architecture.super_munit_generator import SuperLayerNorm
from models.modules.resnet_architecture.mobile_resnet_generator import MobileResnetGenerator
from models.modules.resnet_architecture.sub_mobile_resnet_generator import SubMobileResnetGenerator
from models.modules.spade_architecture.mobile_spade_generator import MobileSPADEGenerator
from models.modules.spade_architecture.sub_mobile_spade_generator import SubMobileSPADEGenerator
from models.modules.sync_batchnorm import SynchronizedBatchNorm2d
from utils.util import load_network


def transfer_conv(mA: Union[nn.ConvTranspose2d, nn.Conv2d], mB: Union[nn.ConvTranspose2d, nn.Conv2d]):
    weight = mA.weight.data
    a, b, c, d = mB.weight.data.shape
    mB.weight.data = weight[:a, :b, :c, :d]
    if mB.bias is not None:
        a = mB.bias.data.shape[0]
        mB.bias.data = mA.bias.data[:a]


def transfer_bn(mA: SynchronizedBatchNorm2d, mB: SynchronizedBatchNorm2d):
    if mA.weight is not None:
        assert mB.weight is not None
        assert mA.weight.shape == mB.weight.shape
        mB.weight.data = mA.weight.data
    if mA.bias is not None:
        assert mB.bias is not None
        assert mA.bias.shape == mB.bias.shape
        mB.bias.data = mA.bias.data
    if mA.running_mean is not None:
        assert mB.running_mean is not None
        assert mA.running_mean.shape == mB.running_mean.shape
        mB.running_mean.data = mA.running_mean.data
    if mA.running_var is not None:
        assert mB.running_var is not None
        assert mA.running_var.shape == mB.running_var.shape
        mB.running_var.data = mA.running_var.data
    if mA.num_batches_tracked is not None:
        assert mB.num_batches_tracked is not None
        mB.num_batches_tracked.data = mA.num_batches_tracked.data


def transfer_ln(mA: Union[LayerNorm, SuperLayerNorm], mB: Union[LayerNorm, SuperLayerNorm]):
    assert mA.num_features >= mB.num_features
    assert mA.affine == mB.affine
    if mA.affine:
        c = mB.num_features
        mB.gamma.data = mA.gamma.data[:c]
        mB.beta.data = mA.beta.data[:c]


def transfer_linear(mA: nn.Linear, mB: nn.Linear, indices=None):
    if indices is None:
        indices = list(range(mB.out_features))
    oc, ic = mB.weight.data.shape
    assert len(indices) == oc
    mB.weight.data = mA.weight.data[indices, :ic]
    assert (mA.bias is not None) == (mB.bias is not None)
    if mA.bias is not None:
        mB.bias.data = mA.bias.data[indices]


def transfer_MLP(netA: SuperMobileAdaINGenerator, netB: SubMobileAdaINGenerator):
    offset = 0
    indices = []
    for nA, mA in netA.named_modules():
        if "AdaptiveInstanceNorm2d" in mA.__class__.__name__:
            for nB, mB in netB.named_modules():
                if nA == nB:
                    for i in range(offset, offset + mB.num_features):
                        indices.append(i)
                    offset += mA.num_features
                    for i in range(offset, offset + mB.num_features):
                        indices.append(i)
                    offset += mA.num_features
    for nA, mA in netA.named_modules():
        if isinstance(mA, nn.Linear):
            for nB, mB in netB.named_modules():
                if nA == nB:
                    assert isinstance(mB, nn.Linear)
                    if mA.out_features == mB.out_features:
                        transfer_linear(mA, mB)
                    else:
                        transfer_linear(mA, mB, indices)


def transfer_weight(netA: nn.Module, netB: nn.Module):  # netA -> netB
    for nA, mA in netA.named_modules():
        if isinstance(mA, (nn.ConvTranspose2d, nn.Conv2d)):
            for nB, mB in netB.named_modules():
                if nA == nB:
                    transfer_conv(mA, mB)
        elif isinstance(mA, SynchronizedBatchNorm2d):
            for nB, mB in netB.named_modules():
                if nA == nB:
                    transfer_bn(mA, mB)
        elif isinstance(mA, (LayerNorm, SuperLayerNorm)):
            for nB, mB in netB.named_modules():
                if nA == nB:
                    transfer_ln(mA, mB)
    if isinstance(netA, SuperMobileAdaINGenerator) and isinstance(netB, SubMobileAdaINGenerator):
        transfer_MLP(netA, netB)


def main(opt):
    config = decode_config(opt.config_str)
    if opt.model == 'mobile_resnet':

        input_nc, output_nc = opt.input_nc, opt.output_nc
        super_model = MobileResnetGenerator(input_nc, output_nc, ngf=opt.ngf,
                                            norm_layer=nn.InstanceNorm2d, n_blocks=9)
        sub_model = SubMobileResnetGenerator(input_nc, output_nc, config=config,
                                             norm_layer=nn.InstanceNorm2d, n_blocks=9)
    elif opt.model == 'mobile_spade':
        opt.norm_G = 'spadesyncbatch3x3'
        opt.semantic_nc = opt.input_nc + (1 if opt.contain_dontcare_label else 0) + (0 if opt.no_instance else 1)
        super_model = MobileSPADEGenerator(opt)
        sub_model = SubMobileSPADEGenerator(opt, config)
    elif opt.model == 'mobile_munit':
        super_model = SuperMobileAdaINGenerator(opt)
        sub_model = SubMobileAdaINGenerator(opt, config)
    else:
        raise NotImplementedError('Unknown architecture [%s]!' % opt.model)

    load_network(super_model, opt.input_path)
    transfer_weight(super_model, sub_model)

    output_dir = os.path.dirname(opt.output_path)
    os.makedirs(output_dir, exist_ok=True)
    torch.save(sub_model.state_dict(), opt.output_path)
    print('Successfully export the subnet at [%s].' % opt.output_path)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Export a specific subnet from a supernet')
    parser.add_argument('--model', type=str, default='mobile_resnet',
                        choices=['mobile_resnet', 'mobile_spade', 'mobile_munit'],
                        help='specify the model type you want to export')
    parser.add_argument('--ngf', type=int, default=48, help='the base number of filters of the generator')
    parser.add_argument('--input_path', type=str, required=True, help='the input model path')
    parser.add_argument('--output_path', type=str, required=True, help='the path to the exported model')
    parser.add_argument('--config_str', type=str, required=True,
                        help='the configuration string for a specific subnet in the supernet')
    parser.add_argument('--input_nc', type=int, default=3,
                        help='# of input image channels: 3 for RGB and 1 for grayscale')
    parser.add_argument('--output_nc', type=int, default=3,
                        help='# of output image channels: 3 for RGB and 1 for grayscale')
    parser.add_argument('--no_instance', action='store_true',
                        help='if specified, do *not* add instance map as input')
    parser.add_argument('--separable_conv_norm', type=str, default='instance',
                        choices=('none', 'instance', 'batch'),
                        help='whether to use instance norm for the separable convolutions')
    parser.add_argument('--contain_dontcare_label', action='store_true',
                        help='if the label map contains dontcare label (dontcare=255)')
    parser.add_argument('--crop_size', type=int, default=512,
                        help='then crop to this size')
    parser.add_argument('--aspect_ratio', type=float, default=2.0,
                        help='The ratio width/height. The final height of the load image will be crop_size/aspect_ratio')

    parser.add_argument('--num_upsampling_layers',
                        choices=('normal', 'more', 'most'), default='more',
                        help="If 'more', adds upsampling layer between the two middle resnet blocks. "
                             "If 'most', also add one more upsampling + resnet layer at the end of the generator")

    parser.add_argument('--style_dim', type=int, default=8,
                        help='the dimension of the style vector for MUNIT model')
    parser.add_argument('--n_downsample', type=int, default=2,
                        help='the number of downsample layer in the MUNIT generator')
    parser.add_argument('--n_res', type=int, default=4,
                        help='the number of the ResBlock in the MUNIT generator')
    parser.add_argument('--activ', type=str, default='relu',
                        help='the activation type of the generator')
    parser.add_argument('--pad_type', type=str, default='reflect',
                        help='the padding type of the MUNIT generator and the MUNIT discriminator')
    parser.add_argument('--mlp_dim', type=int, default=256,
                        help='the dimension of the mlp layer in the MUNIT generator')
    parser.add_argument('--no_style_encoder', action='store_true',
                        help='whether to have the style encoder in the MUNIT generator')
    opt = parser.parse_args()
    main(opt)
