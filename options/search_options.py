from options.base_options import BaseOptions


class SearchOptions(BaseOptions):
    """This class includes test options.

    It also includes shared options defined in BaseOptions.
    """

    def __init__(self, isTrain=False):
        super(SearchOptions, self).__init__()
        self.isTrain = isTrain

    def initialize(self, parser):
        parser = BaseOptions.initialize(self, parser)  # define shared options
        parser.add_argument('--output_path', type=str, default=None, required=True,
                            help='the path to save the evaluation result.')
        parser.add_argument('--num_test', type=int, default=float('inf'), help='how many test images to run')
        parser.add_argument('--model', type=str, default='test', help='which model do you want test')
        parser.add_argument('--no_fid', action='store_true',
                            help='whether you want to compute FID.')
        parser.add_argument('--no_mIoU', action='store_true',
                            help='whether you want to compute mIoU.')

        parser.add_argument('--netG', type=str, default='super_mobile_resnet_9blocks',
                            choices=['super_mobile_resnet_9blocks', 'super_mobile_spade'],
                            help='specify generator architecture')
        parser.add_argument('--ngf', type=int, default=48,
                            help='the base number of filters of the student generator')
        parser.add_argument('--dropout_rate', type=float, default=0,
                            help='the dropout rate of the generator')
        parser.add_argument('--budget', type=float, default=1e18,
                            help='the MAC budget')
        parser.add_argument('--real_stat_path', type=str, required=True,
                            help='the path to load the ground-truth images information to compute FID.')
        parser.add_argument('--num_splits', type=int, default=1,
                            help='the number of splits you would like to split searching space into')
        parser.add_argument('--split', type=int, default=0,
                            help='the specific split your are evaluating')

        parser.add_argument('--restore_pkl_path', type=str, default=None,
                            help='the checkpoint to restore searching')
        parser.add_argument('--save_freq', type=int, default=10,
                            help='the number of minutes to save the latest searching results')
        # rewrite devalue values
        parser.set_defaults(phase='val', serial_batches=True, no_flip=True,
                            load_size=parser.get_default('crop_size'), load_in_memory=True)

        return parser
