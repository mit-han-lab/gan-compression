from .base_options import BaseOptions


class TestOptions(BaseOptions):
    """This class includes test options.

    It also includes shared options defined in BaseOptions.
    """

    def __init__(self, isTrain=False):
        super(TestOptions, self).__init__()
        self.isTrain = isTrain

    def initialize(self, parser):
        parser = BaseOptions.initialize(self, parser)  # define shared options
        parser.add_argument('--results_dir', type=str, default=None, required=True, help='saves results here.')
        parser.add_argument('--need_profile', action='store_true')
        parser.add_argument('--num_test', type=int, default=float('inf'), help='how many test images to run')
        parser.add_argument('--model', type=str, default='test', help='which model do you want test')
        parser.add_argument('--restore_G_path', type=str, required=True, help='the path to restore the generator')
        parser.add_argument('--netG', type=str, default='sub_mobile_resnet_9blocks',
                            choices=['resnet_9blocks',
                                     'mobile_resnet_9blocks',
                                     'super_mobile_resnet_9blocks',
                                     'sub_mobile_resnet_9blocks',
                                     'spade', 'sub_mobile_spade'],
                            help='specify generator architecture')
        parser.add_argument('--ngf', type=int, default=64, help='the base number of filters of the student generator')
        parser.add_argument('--dropout_rate', type=float, default=0, help='the dropout rate of the generator')
        # rewrite devalue values
        parser.add_argument('--no_fid', action='store_true')
        parser.add_argument('--real_stat_path', type=str, required=None,
                            help='the path to load the groud-truth images information to compute FID.')
        parser.add_argument('--no_mIoU', action='store_true')
        parser.add_argument('--times', type=int, default=100,
                            help='times of forwarding the data to test the latency')
        parser.set_defaults(phase='val', serial_batches=True, no_flip=True,
                            load_size=parser.get_default('crop_size'), batch_size=1)
        return parser
