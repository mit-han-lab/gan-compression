"""
Copyright (C) 2019 NVIDIA Corporation.  All rights reserved.
Licensed under the CC BY-NC-SA 4.0 license (https://creativecommons.org/licenses/by-nc-sa/4.0/legalcode).
"""

import argparse
import os.path

from data.image_folder import make_dataset
from data.spade_dataset import SPADEDataset


class CocoDataset(SPADEDataset):

    def __init__(self, opt):
        super(CocoDataset, self).__init__(opt)

    @staticmethod
    def modify_commandline_options(parser, is_train):
        parser = SPADEDataset.modify_commandline_options(parser, is_train)
        assert isinstance(parser, argparse.ArgumentParser)
        parser.add_argument('--coco_no_portraits', action='store_true')
        parser.set_defaults(preprocess_mode='resize_and_crop')
        if is_train:
            parser.set_defaults(load_size=286)
        else:
            parser.set_defaults(load_size=256)
        parser.set_defaults(crop_size=256)
        parser.set_defaults(input_nc=182)
        parser.set_defaults(contain_dontcare_label=True)
        parser.set_defaults(direction='BtoA')
        return parser

    def get_paths(self, opt):
        root = opt.dataroot
        phase = opt.phase

        label_dir = os.path.join(root, '%s_label' % phase)
        label_paths = make_dataset(label_dir, recursive=False, read_cache=True)

        if not opt.coco_no_portraits and opt.isTrain:
            label_portrait_dir = os.path.join(root, '%s_label_portrait' % phase)
            if os.path.isdir(label_portrait_dir):
                label_portrait_paths = make_dataset(label_portrait_dir, recursive=False, read_cache=True)
                label_paths += label_portrait_paths

        image_dir = os.path.join(root, '%s_img' % phase)
        image_paths = make_dataset(image_dir, recursive=False, read_cache=True)

        if not opt.coco_no_portraits and opt.isTrain:
            image_portrait_dir = os.path.join(root, '%s_img_portrait' % phase)
            if os.path.isdir(image_portrait_dir):
                image_portrait_paths = make_dataset(image_portrait_dir, recursive=False, read_cache=True)
                image_paths += image_portrait_paths

        if not opt.no_instance:
            instance_dir = os.path.join(root, '%s_inst' % phase)
            instance_paths = make_dataset(instance_dir, recursive=False, read_cache=True)

            if not opt.coco_no_portraits and opt.isTrain:
                instance_portrait_dir = os.path.join(root, '%s_inst_portrait' % phase)
                if os.path.isdir(instance_portrait_dir):
                    instance_portrait_paths = make_dataset(instance_portrait_dir, recursive=False, read_cache=True)
                    instance_paths += instance_portrait_paths

        else:
            instance_paths = []

        return label_paths, image_paths, instance_paths

    def paths_match(self, path1, path2):
        name1 = os.path.basename(path1)
        name2 = os.path.basename(path2)
        filename1 = os.path.splitext(name1)[0]
        filename2 = os.path.splitext(name2)[0]
        return filename1 == filename2
