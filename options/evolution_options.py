import argparse

from options.base_options import BaseOptions


class EvolutionOptions(BaseOptions):
    """This class includes test options.

    It also includes shared options defined in BaseOptions.
    """

    def __init__(self, isTrain=False):
        super(EvolutionOptions, self).__init__()
        self.isTrain = isTrain

    def initialize(self, parser):
        parser = BaseOptions.initialize(self, parser)  # define shared options
        assert isinstance(parser, argparse.ArgumentParser)
        parser.add_argument('--output_dir', type=str, default=None, required=True,
                            help='the path to save the evaluation result.')
        parser.add_argument('--num_test', type=int, default=float('inf'), help='how many test images to run')
        parser.add_argument('--model', type=str, default='test', help='which model do you want test')
        parser.add_argument('--no_fid', action='store_true',
                            help='whether you want to compute FID.')
        parser.add_argument('--no_mIoU', action='store_true',
                            help='whether you want to compute mIoU.')

        parser.add_argument('--netG', type=str, default='super_mobile_resnet_9blocks',
                            help='specify generator architecture')
        parser.add_argument('--ngf', type=int, default=48,
                            help='the base number of filters of the student generator')
        parser.add_argument('--dropout_rate', type=float, default=0,
                            help='the dropout rate of the generator')
        parser.add_argument('--budget', type=float, default=1e18,
                            help='the MAC budget')
        parser.add_argument('--real_stat_path', type=str, default=None,
                            help='the path to load the ground-truth images information to compute FID.')

        parser.add_argument('--max_cache_size', type=int, default=10000000,
                            help='the cache size to store the results')
        parser.add_argument('--population_size', type=int, default=100)
        parser.add_argument('--mutate_prob', type=float, default=0.2,
                            help='the probability of mutation')
        parser.add_argument('--mutation_ratio', type=float, default=0.5,
                            help='the ratio of networks that are generated through mutation in generation n >= 2.')
        parser.add_argument('--parent_ratio', type=float, default=0.25,
                            help='the ratio of networks that are used as parents for next generation')
        parser.add_argument('--evolution_iters', type=int, default=500,
                            help='how many generations of population to be searched')
        parser.add_argument('--criterion', type=str, default='fid',
                            help='the criterion for the performance',
                            choices=['fid', 'mIoU', 'accu'])
        parser.add_argument('--weighted_sample', type=float, default=1,
                            help='a weight parameter to configure the sampling of the first generation'
                                 '(only affect the first generation)')
        parser.add_argument('--weight_strategy', type=str, default='arithmetic', choices=['arithmetic', 'geometric'],
                            help='use arithmetic weights or geometric weights to sample the first generation')
        parser.add_argument('--generation_base', type=int, default=1,
                            help='the generation base of the evolution (used for resuming)')
        parser.add_argument('--restore_pkl_path', type=str, default=None,
                            help='the checkpoint to restore searching')
        parser.add_argument('--only_restore_cache', action='store_true',
                            help='whether to only restore caches in the pkl file')
        parser.add_argument('--save_freq', type=int, default=60,
                            help='the number of minutes to save the latest searching results')

        # rewrite devalue values
        parser.set_defaults(phase='val', serial_batches=True, no_flip=True,
                            load_size=parser.get_default('crop_size'), load_in_memory=True)

        return parser
