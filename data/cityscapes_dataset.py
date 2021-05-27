"""
Copyright (C) 2019 NVIDIA Corporation.  All rights reserved.
Licensed under the CC BY-NC-SA 4.0 license (https://creativecommons.org/licenses/by-nc-sa/4.0/legalcode).
"""

import argparse
import os.path

from data.image_folder import make_dataset
from data.spade_dataset import SPADEDataset


class CityscapesDataset(SPADEDataset):

    def __init__(self, opt):
        super(CityscapesDataset, self).__init__(opt)

    @staticmethod
    def modify_commandline_options(parser, is_train):
        parser = SPADEDataset.modify_commandline_options(parser, is_train)
        assert isinstance(parser, argparse.ArgumentParser)
        parser.set_defaults(preprocess='scale_width', aspect_ratio=2,
                            load_size=512, crop_size=512, direction='BtoA',
                            display_winsize=512, input_nc=35, num_threads=0)
        return parser

    def get_paths(self, opt):
        root = opt.dataroot
        phase = opt.phase

        label_dir = os.path.join(root, 'gtFine', phase)
        label_paths_all = make_dataset(label_dir, recursive=True)
        label_paths = [p for p in label_paths_all if p.endswith('_labelIds.png')]

        image_dir = os.path.join(root, 'leftImg8bit', phase)
        image_paths = make_dataset(image_dir, recursive=True)

        if not opt.no_instance:
            instance_paths = [p for p in label_paths_all if p.endswith('_instanceIds.png')]
        else:
            instance_paths = []

        return label_paths, image_paths, instance_paths

    def paths_match(self, path1, path2):
        name1 = os.path.basename(path1)
        name2 = os.path.basename(path2)
        # compare the first 3 components, [city]_[id1]_[id2]
        return '_'.join(name1.split('_')[:3]) == '_'.join(name2.split('_')[:3])
