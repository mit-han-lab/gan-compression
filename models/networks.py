import functools

import torch
import torch.nn as nn
from torch.nn import init
from torch.optim import lr_scheduler

from configs import decode_config


###############################################################################
# Helper Functions
###############################################################################

class BaseNetwork(nn.Module):
    def __init__(self):
        super(BaseNetwork, self).__init__()

    @staticmethod
    def modify_commandline_options(parser, is_train):
        return parser


class Identity(nn.Module):
    def forward(self, x):
        return x


def get_norm_layer(norm_type='instance'):
    """Return a normalization layer

    Parameters:
        norm_type (str) -- the name of the normalization layer: batch | instance | none

    For BatchNorm, we use learnable affine parameters and track running statistics (mean/stddev).
    For InstanceNorm, we do not use learnable affine parameters. We do not track running statistics.
    """
    if norm_type == 'batch':
        norm_layer = functools.partial(nn.BatchNorm2d, affine=True, track_running_stats=True)
    elif norm_type == 'instance':
        norm_layer = functools.partial(nn.InstanceNorm2d, affine=False, track_running_stats=False)
    elif norm_type == 'none':
        def norm_layer(x):
            return Identity()
    else:
        raise NotImplementedError('normalization layer [%s] is not found' % norm_type)
    return norm_layer


def get_scheduler(optimizer, opt):
    """Return a learning rate scheduler

    Parameters:
        optimizer          -- the optimizer of the network
        opt (option class) -- stores all the experiment flags; needs to be a subclass of BaseOptions．　
                              opt.lr_policy is the name of learning rate policy: linear | step | plateau | cosine

    For 'linear', we keep the same learning rate for the first <opt.niter> epochs
    and linearly decay the rate to zero over the next <opt.niter_decay> epochs.
    For other schedulers (step, plateau, and cosine), we use the default PyTorch schedulers.
    See https://pytorch.org/docs/stable/optim.html for more details.
    """
    if opt.scheduler_counter == 'epoch':
        last_epoch = opt.epoch_base - 2
    else:
        last_epoch = opt.iter_base - 1
    if opt.lr_policy == 'linear':
        def lambda_rule(epoch):
            lr_l = 1.0 - max(0, epoch + 1 - opt.nepochs) / float(opt.nepochs_decay + 1)
            return lr_l

        scheduler = lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda_rule, last_epoch=last_epoch)
    elif opt.lr_policy == 'step':
        scheduler = lr_scheduler.StepLR(optimizer, step_size=opt.lr_decay_steps, gamma=opt.gamma, last_epoch=last_epoch)
    elif opt.lr_policy == 'plateau':
        scheduler = lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.2, threshold=0.01, patience=5)
    elif opt.lr_policy == 'cosine':
        scheduler = lr_scheduler.CosineAnnealingLR(optimizer, T_max=opt.niter, eta_min=0, last_epoch=last_epoch)
    else:
        return NotImplementedError('learning rate policy [%s] is not implemented', opt.lr_policy)
    return scheduler


def init_weights(net, init_type='normal', init_gain=0.02):
    """Initialize network weights.

    Parameters:
        net (network)   -- network to be initialized
        init_type (str) -- the name of an initialization method: normal | xavier | kaiming | orthogonal
        init_gain (float)    -- scaling factor for normal, xavier and orthogonal.

    We use 'normal' in the original pix2pix and CycleGAN paper. But xavier and kaiming might
    work better for some applications. Feel free to try yourself.
    """

    def init_func(m):  # define the initialization function
        classname = m.__class__.__name__
        if hasattr(m, 'weight') and (classname.find('Conv') != -1 or classname.find('Linear') != -1):
            if init_type == 'normal':
                init.normal_(m.weight.data, 0.0, init_gain)
            elif init_type == 'xavier':
                init.xavier_normal_(m.weight.data, gain=init_gain)
            elif init_type == 'kaiming':
                init.kaiming_normal_(m.weight.data, a=0, mode='fan_in')
            elif init_type == 'orthogonal':
                init.orthogonal_(m.weight.data, gain=init_gain)
            else:
                raise NotImplementedError('initialization method [%s] is not implemented' % init_type)
            if hasattr(m, 'bias') and m.bias is not None:
                init.constant_(m.bias.data, 0.0)
        elif classname.find(
                'BatchNorm2d') != -1:  # BatchNorm Layer's weight is not a matrix; only normal distribution applies.
            if hasattr(m, 'weight') and m.weight is not None:
                init.normal_(m.weight.data, 1.0, init_gain)
            if hasattr(m, 'bias') and m.weight is not None:
                init.constant_(m.bias.data, 0.0)

    print('initialize network with %s' % init_type)
    net.apply(init_func)  # apply the initialization function <init_func>


def init_net(net, init_type='normal', init_gain=0.02, gpu_ids=[]):
    """Initialize a network: 1. register CPU/GPU device (with multi-GPU support); 2. initialize the network weights
    Parameters:
        net (network)      -- the network to be initialized
        init_type (str)    -- the name of an initialization method: normal | xavier | kaiming | orthogonal
        gain (float)       -- scaling factor for normal, xavier and orthogonal.
        gpu_ids (int list) -- which GPUs the network runs on: e.g., 0,1,2

    Return an initialized network.
    """
    if len(gpu_ids) > 0:
        assert (torch.cuda.is_available())
        net.to(gpu_ids[0])
        if len(gpu_ids) > 1:
            net = torch.nn.DataParallel(net, gpu_ids)  # multi-GPUs
    init_weights(net, init_type, init_gain=init_gain)

    return net


def define_G(netG, **kwargs):
    Generator = get_netG_cls(netG)
    if netG in ['resnet_9blocks', 'mobile_resnet_9blocks', 'super_mobile_resnet_9blocks']:
        assert 'input_nc' in kwargs and 'output_nc' in kwargs and 'ngf' in kwargs
        input_nc = kwargs.get('input_nc')
        output_nc = kwargs.get('output_nc')
        ngf = kwargs.get('ngf')
        dropout_rate = kwargs.get('dropout_rate', 0)
        norm = kwargs.get('norm', 'batch')
        norm_layer = get_norm_layer(norm_type=norm)
        net = Generator(input_nc, output_nc, ngf=ngf, norm_layer=norm_layer,
                        dropout_rate=dropout_rate, n_blocks=9)
    elif netG in ['sub_mobile_resnet_9blocks', 'legacy_sub_mobile_resnet_9blocks']:
        assert 'input_nc' in kwargs and 'output_nc' in kwargs and 'opt' in kwargs
        input_nc = kwargs.get('input_nc')
        output_nc = kwargs.get('output_nc')
        dropout_rate = kwargs.get('dropout_rate', 0)
        norm = kwargs.get('norm', 'batch')
        opt = kwargs.get('opt')
        norm_layer = get_norm_layer(norm_type=norm)
        assert opt.config_str is not None
        config = decode_config(opt.config_str)
        net = Generator(input_nc, output_nc, config, norm_layer=norm_layer,
                        dropout_rate=dropout_rate, n_blocks=9)
    elif netG in ['spade', 'mobile_spade', 'super_mobile_spade', 'munit', 'super_munit', 'super_mobile_munit']:
        assert 'opt' in kwargs
        opt = kwargs.get('opt')
        net = Generator(opt)
    elif netG in ['sub_mobile_spade', 'sub_mobile_munit']:
        assert 'opt' in kwargs
        opt = kwargs.get('opt')
        assert opt.config_str is not None
        config = decode_config(opt.config_str)
        net = Generator(opt, config)
    else:
        raise NotImplementedError('Generator model name [%s] is not recognized' % netG)
    init_type = kwargs.get('init_type', 'normal')
    init_gain = kwargs.get('init_gain', 0.02)
    gpu_ids = kwargs.get('gpu_ids', [])
    return init_net(net, init_type, init_gain, gpu_ids)


def define_D(netD, **kwargs):
    Discriminator = get_netD_cls(netD)
    if netD == 'multi_scale':
        assert 'opt' in kwargs
        opt = kwargs.get('opt')
        net = Discriminator(opt)
    elif netD == 'n_layers':
        assert 'input_nc' in kwargs and 'ndf' in kwargs
        input_nc = kwargs.get('input_nc')
        ndf = kwargs.get('ndf')
        n_layers_D = kwargs.get('n_layers_D', 3)
        norm = kwargs.get('norm', 'batch')
        norm_layer = get_norm_layer(norm_type=norm)
        net = Discriminator(input_nc, ndf, n_layers_D, norm_layer=norm_layer)
    elif netD == 'pixel':  # classify if each pixel is real or fake
        assert 'input_nc' in kwargs and 'ndf' in kwargs
        input_nc = kwargs.get('input_nc')
        ndf = kwargs.get('ndf')
        norm = kwargs.get('norm', 'batch')
        norm_layer = get_norm_layer(norm_type=norm)
        net = Discriminator(input_nc, ndf, norm_layer=norm_layer)
    elif netD == 'ms_image':
        assert 'input_nc' in kwargs and 'opt' in kwargs
        input_nc = kwargs.get('input_nc')
        opt = kwargs.get('opt')
        net = Discriminator(input_nc, opt)
    else:
        raise NotImplementedError('Discriminator model name [%s] is not recognized' % netD)
    init_type = kwargs.get('init_type', 'normal')
    init_gain = kwargs.get('init_gain', '0.02')
    gpu_ids = kwargs.get('gpu_ids', [])
    return init_net(net, init_type, init_gain, gpu_ids)


def get_netG_cls(netG):
    if netG == 'resnet_9blocks':
        from models.modules.resnet_architecture.resnet_generator import ResnetGenerator
        return ResnetGenerator
    elif netG == 'mobile_resnet_9blocks':
        from models.modules.resnet_architecture.mobile_resnet_generator import MobileResnetGenerator
        return MobileResnetGenerator
    elif netG == 'super_mobile_resnet_9blocks':
        from models.modules.resnet_architecture.super_mobile_resnet_generator import SuperMobileResnetGenerator
        return SuperMobileResnetGenerator
    elif netG == 'sub_mobile_resnet_9blocks':
        from models.modules.resnet_architecture.sub_mobile_resnet_generator import SubMobileResnetGenerator
        return SubMobileResnetGenerator
    elif netG == 'legacy_sub_mobile_resnet_9blocks':
        from models.modules.resnet_architecture.legacy_sub_mobile_resnet_generator import LegacySubMobileResnetGenerator
        return LegacySubMobileResnetGenerator
    elif netG == 'spade':
        from models.modules.spade_architecture.spade_generator import SPADEGenerator
        return SPADEGenerator
    elif netG == 'mobile_spade':
        from models.modules.spade_architecture.mobile_spade_generator import MobileSPADEGenerator
        return MobileSPADEGenerator
    elif netG == 'super_mobile_spade':
        from models.modules.spade_architecture.super_mobile_spade_generator import SuperMobileSPADEGenerator
        return SuperMobileSPADEGenerator
    elif netG == 'sub_mobile_spade':
        from models.modules.spade_architecture.sub_mobile_spade_generator import SubMobileSPADEGenerator
        return SubMobileSPADEGenerator
    elif netG == 'munit':
        from models.modules.munit_architecture.munit_generator import AdaINGenerator
        return AdaINGenerator
    elif netG == 'super_munit':
        from models.modules.munit_architecture.super_munit_generator import SuperAdaINGenerator
        return SuperAdaINGenerator
    elif netG == 'super_mobile_munit':
        from models.modules.munit_architecture.super_mobile_munit_generator import SuperMobileAdaINGenerator
        return SuperMobileAdaINGenerator
    elif netG == 'sub_mobile_munit':
        from models.modules.munit_architecture.sub_mobile_munit_generator import SubMobileAdaINGenerator
        return SubMobileAdaINGenerator
    else:
        raise NotImplementedError('Generator model name [%s] is not recognized' % netG)


def get_netD_cls(netD):
    if netD == 'n_layers':
        from models.modules.discriminators import NLayerDiscriminator
        return NLayerDiscriminator
    elif netD == 'pixel':  # classify if each pixel is real or fake
        from models.modules.discriminators import PixelDiscriminator
        return PixelDiscriminator
    elif netD == 'multi_scale':
        from models.modules.discriminators import MultiscaleDiscriminator
        return MultiscaleDiscriminator
    elif netD == 'ms_image':
        from models.modules.discriminators import MsImageDiscriminator
        return MsImageDiscriminator
    else:
        raise NotImplementedError('Discriminator model name [%s] is not recognized' % netD)


def modify_commandline_options(parser, is_train):
    opt, _ = parser.parse_known_args()
    if hasattr(opt, 'netG'):
        netG_cls = get_netG_cls(opt.netG)
        parser = netG_cls.modify_commandline_options(parser, is_train)
    if is_train:
        netD_cls = get_netD_cls(opt.netD)
        parser = netD_cls.modify_commandline_options(parser, is_train)
    return parser
