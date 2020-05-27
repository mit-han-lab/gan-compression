import argparse
import os
import pickle

import torch

import data
import models


class BaseOptions:
    """This class defines options used during both training and test time.

    It also implements several helper functions such as parsing, printing, and saving the options.
    It also gathers additional options defined in <modify_commandline_options> functions in both dataset class and model class.
    """

    def __init__(self):
        """Reset the class; indicates the class hasn't been initailized"""
        self.isTrain = True

    def initialize(self, parser):
        """Define the common options that are used in both training and test."""
        # basic parameters
        parser.add_argument('--dataroot', required=True,
                            help='path to images (should have subfolders trainA, trainB, valA, valB, train, val, etc)')
        parser.add_argument('--gpu_ids', type=str, default='0', help='gpu ids: e.g. 0  0,1,2, 0,2. use -1 for CPU')
        parser.add_argument('--seed', type=int, default=233, help='random seed')

        # model parameters
        parser.add_argument('--input_nc', type=int, default=3,
                            help='# of input image channels: 3 for RGB and 1 for grayscale')
        parser.add_argument('--output_nc', type=int, default=3,
                            help='# of output image channels: 3 for RGB and 1 for grayscale')
        parser.add_argument('--norm', type=str, default='instance',
                            help='instance normalization or batch normalization [instance | batch | none]')
        parser.add_argument('--init_type', type=str, default='normal',
                            help='network initialization [normal | xavier | kaiming | orthogonal]')
        parser.add_argument('--init_gain', type=float, default=0.02,
                            help='scaling factor for normal, xavier and orthogonal.')
        parser.add_argument('--use_coord', dest='use_coord', action='store_true',
                            help='add CoordConv Layers instead Convolution layer or not')
        parser.add_argument('--use_motion', dest='use_motion', action='store_true',
                            help='add Dense Motion Layer into net')
        parser.add_argument('--use_motion_tanh', dest='motion_tanh', action='store_true',
                            help='add Dense Motion Layer into net')
        parser.set_defaults(use_coord=False, use_motion=False, motion_tanh=False)

        # dataset parameters
        parser.add_argument('--dataset_mode', type=str, default='aligned',
                            help='chooses how datasets are loaded. [unaligned | aligned | single]')
        parser.add_argument('--direction', type=str, default='AtoB', help='AtoB or BtoA')
        parser.add_argument('--serial_batches', action='store_true',
                            help='if true, takes images in order to make batches, otherwise takes them randomly')
        parser.add_argument('--num_threads', default=4, type=int, help='# threads for loading data')
        parser.add_argument('--batch_size', type=int, default=1, help='input batch size')
        parser.add_argument('--load_size', type=int, default=286, help='scale images to this size')
        parser.add_argument('--crop_size', type=int, default=256, help='then crop to this size')
        parser.add_argument('--aspect_ratio', type=float, default=1.0,
                            help='The ratio width/height. The final height of the load image will be crop_size/aspect_ratio')
        parser.add_argument('--max_dataset_size', type=int, default=-1,
                            help='Maximum number of samples allowed per dataset. If the dataset directory contains more than max_dataset_size, only a subset is loaded.')
        parser.add_argument('--preprocess', type=str, default='resize_and_crop',
                            help='scaling and cropping of images at load time [resize_and_crop | crop | scale_width | scale_width_and_crop | none]')
        parser.add_argument('--no_flip', action='store_true',
                            help='if specified, do not flip the images for data augmentation')
        parser.add_argument('--display_winsize', type=int, default=256,
                            help='display window size for both visdom and HTML')
        parser.add_argument('--load_in_memory', action='store_true',
                            help='whether you will load the data into the memory to bypass the IO.')
        parser.add_argument('--phase', type=str, default='train', help='train, val, test, etc')
        parser.add_argument('--config_set', type=str, default=None,
                            help='the name of the configuration set for the set of subnet configurations.')
        parser.add_argument('--config_str', type=str, default=None,
                            help='the configuration string for a specific subnet in the supernet')

        # evaluation parameters
        parser.add_argument('--drn_path', type=str, default='drn-d-105_ms_cityscapes.pth',
                            help='the path to the pretrained drn path to compute mAP')
        parser.add_argument('--cityscapes_path', type=str, default='database/cityscapes-origin',
                            help='the original cityscapes dataset path (not the pix2pix preprocessed one)')
        parser.add_argument('--table_path', type=str, default='datasets/table.txt',
                            help='the path to the mapping table (generated by datasets/prepare_cityscapes_dataset.py)')
        return parser

    def gather_options(self):
        # """Initialize our parser with basic options(only once).
        # Add additional model-specific and dataset-specific options.
        # These options are defined in the <modify_commandline_options> function
        # in model and dataset classes.
        # """
        parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
        parser = self.initialize(parser)

        # get the basic options
        opt, _ = parser.parse_known_args()

        # modify model-related parser options
        model_name = opt.model
        model_option_setter = models.get_option_setter(model_name)
        parser = model_option_setter(parser, self.isTrain)
        opt, _ = parser.parse_known_args()  # parse again with new defaults

        # modify dataset-related parser options
        dataset_name = opt.dataset_mode
        dataset_option_setter = data.get_option_setter(dataset_name)
        parser = dataset_option_setter(parser, self.isTrain)

        # save and return the parser
        self.parser = parser
        return parser.parse_args()

    def print_options(self, opt):
        message = ''
        message += '----------------- Options ---------------\n'
        for k, v in sorted(vars(opt).items()):
            comment = ''
            default = self.parser.get_default(k)
            if v != default:
                comment = '\t[default: %s]' % str(default)
            message += '{:>25}: {:<30}{}\n'.format(str(k), str(v), comment)
        message += '----------------- End -------------------'
        print(message)

        # save to the disk
        if self.isTrain:
            expr_dir = os.path.join(opt.log_dir)
            os.makedirs(expr_dir, exist_ok=True)
            file_name = os.path.join(expr_dir, 'opt.txt')
            with open(file_name, 'wt') as opt_file:
                opt_file.write(message)
                opt_file.write('\n')
            file_name = os.path.join(expr_dir, 'opt.pkl')
            with open(file_name, 'wb') as opt_file:
                pickle.dump(opt, opt_file)

    def parse(self, verbose=True):
        opt = self.gather_options()
        opt.isTrain = self.isTrain  # train or test

        if hasattr(opt, 'contain_dontcare_label') and hasattr(opt, 'no_instance'):
            opt.semantic_nc = opt.input_nc + (1 if opt.contain_dontcare_label else 0) + (0 if opt.no_instance else 1)

        if verbose:
            self.print_options(opt)

        # set gpu ids
        str_ids = opt.gpu_ids.split(',')
        opt.gpu_ids = []
        for str_id in str_ids:
            id = int(str_id)
            if id >= 0:
                opt.gpu_ids.append(id)
        if len(opt.gpu_ids) > 0:
            torch.cuda.set_device(opt.gpu_ids[0])

        self.opt = opt
        return self.opt
