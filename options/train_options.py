from argparse import ArgumentParser

from options.base_options import BaseOptions


class TrainOptions(BaseOptions):
    """This class includes training options.

    It also includes shared options defined in BaseOptions.
    """

    def __init__(self, isTrain=True):
        super(TrainOptions, self).__init__()
        self.isTrain = isTrain

    def initialize(self, parser):
        parser = BaseOptions.initialize(self, parser)
        # log parameters
        assert isinstance(parser, ArgumentParser)
        parser.add_argument('--log_dir', type=str, default='logs',
                            help='training logs are saved here')
        parser.add_argument('--tensorboard_dir', type=str, default=None,
                            help='tensorboard is saved here')
        parser.add_argument('--print_freq', type=int, default=100,
                            help='frequency of showing training results on console')
        parser.add_argument('--save_latest_freq', type=int, default=20000,
                            help='frequency of evaluating and save the latest model')
        parser.add_argument('--save_epoch_freq', type=int, default=5,
                            help='frequency of saving checkpoints at the end of epoch')
        parser.add_argument('--epoch_base', type=int, default=1,
                            help='the epoch base of the training (used for resuming)')
        parser.add_argument('--iter_base', type=int, default=0,
                            help='the iteration base of the training (used for resuming)')

        # model parameters
        parser.add_argument('--model', type=str, default='pix2pix', help='choose which model to use')
        parser.add_argument('--netD', type=str, default='n_layers',
                            help='specify discriminator architecture [n_layers | pixel]. '
                                 'The basic model is a 70x70 PatchGAN. '
                                 'n_layers allows you to specify the layers in the discriminator')
        parser.add_argument('--netG', type=str, default='mobile_resnet_9blocks',
                            help='specify the generator architecture')
        parser.add_argument('--ngf', type=int, default=64, help='the base number of generator filters')
        parser.add_argument('--ndf', type=int, default=128, help='the base number of discriminator filters')
        parser.add_argument('--n_layers_D', type=int, default=3, help='only used if netD==n_layers')
        parser.add_argument('--dropout_rate', type=float, default=0, help='the dropout rate of the generator')

        parser.add_argument('--restore_O_path', type=str, default=None,
                            help='the path to restore the optimizer')

        # training parameters
        parser.add_argument('--nepochs', type=int, default=5,
                            help='number of epochs with the initial learning rate')
        parser.add_argument('--nepochs_decay', type=int, default=15,
                            help='number of epochs to linearly decay learning rate to zero')
        parser.add_argument('--niters', type=int, default=1000000000,
                            help='max number of iteration of the training')
        parser.add_argument('--beta1', type=float, default=0.5, help='momentum term of adam')
        parser.add_argument('--lr', type=float, default=0.0002, help='initial learning rate for adam')
        parser.add_argument('--gan_mode', type=str, default='hinge',
                            help='the type of GAN objective. [vanilla| lsgan | wgangp | hinge]. '
                                 'vanilla GAN loss is the cross-entropy objective used in the original GAN paper.')
        parser.add_argument('--pool_size', type=int, default=50,
                            help='the size of image buffer that stores previously generated images')
        parser.add_argument('--lr_policy', type=str, default='linear',
                            help='learning rate policy. [linear | step | plateau | cosine]')
        parser.add_argument('--lr_decay_steps', type=int, default=100000,
                            help='multiply by a gamma every lr_decay_steps steps (only for step lr policy)')
        parser.add_argument('--scheduler_counter', type=str, default='epoch', choices=['epoch', 'iter'],
                            help='which counter to use for the scheduler')
        parser.add_argument('--gamma', type=float, default=0.5,
                            help='multiply by a gamma every lr_decay_epochs epochs (only for step lr policy)')

        # evaluation parameters
        parser.add_argument('--eval_batch_size', type=int, default=1, help='the evaluation batch size')
        parser.add_argument('--no_fid', action='store_true', help='No FID evaluation during training')
        parser.add_argument('--no_mIoU', action='store_true', help='No mIoU evaluation during training '
                                                                   '(sometimes because there are CUDA memory)')
        self.isTrain = True
        return parser
