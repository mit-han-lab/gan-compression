from torch import nn

from models.modules.mobile_modules import SeparableConv2d
from models.modules.resnet_architecture.mobile_resnet_generator import MobileResnetBlock
from models.modules.resnet_architecture.resnet_generator import ResnetBlock
from models.modules.resnet_architecture.super_mobile_resnet_generator import SuperMobileResnetBlock
from models.modules.spade_architecture.mobile_spade_generator import MobileSPADEGenerator, MobileSPADEResnetBlock, \
    MobileSPADE
from models.modules.super_modules import SuperConv2d, SuperConvTranspose2d, SuperSeparableConv2d


def transfer_Conv2d(m1, m2, input_index=None, output_index=None):
    assert isinstance(m1, nn.Conv2d) and isinstance(m2, (nn.Conv2d, SuperConv2d))
    if m1.out_channels == 3:  # If this is the last convolution
        assert input_index is not None
        m2.weight.data = m1.weight.data[:, input_index].clone()
        if m2.bias is not None:
            m2.bias.data = m1.bias.data.clone()
        return None
    else:
        if m1.in_channels == 3:  # If this is the first convolution
            assert input_index is None
            input_index = [0, 1, 2]
        p = m1.weight.data

        if input_index is None:
            q = p.abs().sum([0, 2, 3])
            _, idxs = q.topk(m2.in_channels, largest=True)
            p = p[:, idxs]
        else:
            p = p[:, input_index]

        if output_index is None:
            q = p.abs().sum([1, 2, 3])
            _, idxs = q.topk(m2.out_channels, largest=True)
        else:
            idxs = output_index

        m2.weight.data = p[idxs].clone()
        if m2.bias is not None:
            m2.bias.data = m1.bias.data[idxs].clone()

        return idxs


def transfer_ConvTranspose2d(m1, m2, input_index=None, output_index=None):
    assert isinstance(m1, nn.ConvTranspose2d) and isinstance(m2, (nn.ConvTranspose2d, SuperConvTranspose2d))
    assert output_index is None
    p = m1.weight.data
    if input_index is None:
        q = p.abs().sum([1, 2, 3])
        _, idxs = q.topk(m2.in_channels, largest=True)
        p = p[idxs]
    else:
        p = p[input_index]
    q = p.abs().sum([0, 2, 3])
    _, idxs = q.topk(m2.out_channels, largest=True)
    m2.weight.data = p[:, idxs].clone()
    if m2.bias is not None:
        m2.bias.data = m1.bias.data[idxs].clone()
    return idxs


def transfer_SeparableConv2d(m1, m2, input_index=None, output_index=None):
    assert isinstance(m1, SeparableConv2d) and isinstance(m2, (SeparableConv2d, SuperSeparableConv2d))
    dw1, pw1 = m1.conv[0], m1.conv[2]
    dw2, pw2 = m2.conv[0], m2.conv[2]

    if input_index is None:
        p = dw1.weight.data
        q = p.abs().sum([1, 2, 3])
        _, input_index = q.topk(dw2.out_channels, largest=True)
    dw2.weight.data = dw1.weight.data[input_index].clone()
    if dw2.bias is not None:
        dw2.bias.data = dw1.bias.data[input_index].clone()

    idxs = transfer(pw1, pw2, input_index, output_index)
    return idxs


def transfer_MobileResnetBlock(m1, m2, input_index=None, output_index=None):
    assert isinstance(m1, MobileResnetBlock) and isinstance(m2, (MobileResnetBlock, SuperMobileResnetBlock))
    assert output_index is None
    idxs = transfer(m1.conv_block[1], m2.conv_block[1], input_index=input_index)
    idxs = transfer(m1.conv_block[6], m2.conv_block[6], input_index=idxs, output_index=input_index)
    return idxs


def transfer_ResnetBlock(m1, m2, input_index=None, output_index=None):
    assert isinstance(m1, ResnetBlock) and isinstance(m2, ResnetBlock)
    assert output_index is None
    idxs = transfer(m1.conv_block[1], m2.conv_block[1], input_index=input_index)
    idxs = transfer(m1.conv_block[6], m2.conv_block[6], input_index=idxs, output_index=input_index)
    return idxs


def transfer_MobileSPADE(m1, m2, input_index=None, output_index=None):
    assert isinstance(m1, MobileSPADE)
    assert isinstance(m2, MobileSPADE)
    m2.param_free_norm.running_mean = m1.param_free_norm.running_mean[input_index].clone()
    m2.param_free_norm.running_var = m1.param_free_norm.running_var[input_index].clone()
    idxs = transfer(m1.mlp_shared[0], m2.mlp_shared[0], list(range(m1.mlp_shared[0].in_channels)))
    transfer(m1.mlp_gamma, m2.mlp_gamma, idxs, input_index)
    transfer(m1.mlp_beta, m2.mlp_beta, idxs, input_index)
    return input_index


def transfer_MobileSPADEResnetBlock(m1, m2, input_index=None, output_index=None):
    assert isinstance(m1, MobileSPADEResnetBlock)
    assert isinstance(m2, MobileSPADEResnetBlock)
    if m1.learned_shortcut:
        assert m2.learned_shortcut
        idxs = transfer(m1.norm_0, m2.norm_0, input_index)
        idxs = transfer(m1.conv_0, m2.conv_0, idxs)
        idxs = transfer(m1.norm_1, m2.norm_1, idxs)
        output_index = transfer(m1.conv_1, m2.conv_1, idxs)
        idxs = transfer(m1.norm_s, m2.norm_s, input_index)
        transfer(m1.conv_s, m2.conv_s, idxs, output_index)
        return output_index
    else:
        assert not m2.learned_shortcut
        idxs = transfer(m1.norm_0, m2.norm_0, input_index)
        idxs = transfer(m1.conv_0, m2.conv_0, idxs)
        idxs = transfer(m1.norm_1, m2.norm_1, idxs)
        transfer(m1.conv_1, m2.conv_1, idxs, input_index)
        return input_index


def transfer(m1, m2, input_index=None, output_index=None):
    if isinstance(m1, nn.Conv2d):
        return transfer_Conv2d(m1, m2, input_index, output_index)
    elif isinstance(m1, nn.ConvTranspose2d):
        return transfer_ConvTranspose2d(m1, m2, input_index, output_index)
    elif isinstance(m1, SeparableConv2d):
        return transfer_SeparableConv2d(m1, m2, input_index, output_index)
    elif isinstance(m1, ResnetBlock):
        return transfer_ResnetBlock(m1, m2, input_index, output_index)
    elif isinstance(m1, MobileResnetBlock):
        return transfer_MobileResnetBlock(m1, m2, input_index, output_index)
    elif isinstance(m1, MobileSPADEResnetBlock):
        return transfer_MobileSPADEResnetBlock(m1, m2, input_index, output_index)
    elif isinstance(m1, MobileSPADE):
        return transfer_MobileSPADE(m1, m2, input_index, output_index)
    else:
        raise NotImplementedError('Unknown module [%s]!' % type(m1))


def load_pretrained_weight(model1, model2, netA, netB, ngf1, ngf2):
    assert ngf1 >= ngf2

    if isinstance(netA, nn.DataParallel):
        net1 = netA.module
    else:
        net1 = netA
    if isinstance(netB, nn.DataParallel):
        net2 = netB.module
    else:
        net2 = netB

    index = None
    if model1 == 'mobile_resnet_9blocks':
        assert len(net1.model) == len(net2.model)
        for i in range(28):
            m1, m2 = net1.model[i], net2.model[i]
            # assert type(m1) == type(m2)
            if isinstance(m1, (nn.Conv2d, nn.ConvTranspose2d, MobileResnetBlock)):
                index = transfer(m1, m2, index)
    elif model1 == 'resnet_9blocks':
        assert len(net1.model) == len(net2.model)
        for i in range(28):
            m1, m2 = net1.model[i], net2.model[i]
            assert type(m1) == type(m2)
            if isinstance(m1, (nn.Conv2d, nn.ConvTranspose2d, ResnetBlock)):
                index = transfer(m1, m2, index)
    elif model1 == 'mobile_spade':
        assert isinstance(net1, MobileSPADEGenerator)
        assert isinstance(net2, MobileSPADEGenerator)
        idxs = transfer(net1.fc, net2.fc, list(range(netA.fc.in_channels)))
        idxs = transfer(net1.head_0, net2.head_0, idxs)
        idxs = transfer(net1.G_middle_0, net2.G_middle_0, idxs)
        idxs = transfer(net1.G_middle_1, net2.G_middle_1, idxs)
        idxs = transfer(net1.up_0, net2.up_0, idxs)
        idxs = transfer(net1.up_1, net2.up_1, idxs)
        idxs = transfer(net1.up_2, net2.up_2, idxs)
        idxs = transfer(net1.up_3, net2.up_3, idxs)
        if hasattr(net1, 'up_4'):
            assert hasattr(netB, 'up_4')
            idxs = transfer(net1.up_4, net2.up_4, idxs)
        else:
            assert not hasattr(netB, 'up_4')
        idxs = transfer(netA.conv_img, net2.conv_img, idxs)
    else:
        raise NotImplementedError('Unknown model [%s]!' % model1)
