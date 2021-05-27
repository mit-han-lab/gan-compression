"""
Copyright (C) 2019 NVIDIA Corporation.  All rights reserved.
Licensed under the CC BY-NC-SA 4.0 license (https://creativecommons.org/licenses/by-nc-sa/4.0/legalcode).
"""

import os

from PIL import Image

from data.base_dataset import BaseDataset, get_params, get_transform
from utils import util


class SPADEDataset(BaseDataset):

    def __init__(self, opt):
        super(SPADEDataset, self).__init__(opt)
        self.initialize(opt)

    @staticmethod
    def modify_commandline_options(parser, is_train):
        parser.add_argument('--no_pairing_check', action='store_true',
                            help='If specified, skip sanity check of correct label-image file pairing')
        parser.add_argument('--no_instance', action='store_true',
                            help='if specified, do *not* add instance map as input')
        parser.add_argument('--contain_dontcare_label', action='store_true',
                            help='if the label map contains dontcare label (dontcare=255)')
        parser.add_argument('--meta_path', type=str, default=None,
                            help='the path to the meta file')
        return parser

    def paths_match(self, path1, path2):
        raise NotImplementedError

    def __getitem__(self, index):
        # Label Image
        label_path = self.label_paths[index]
        if not self.opt.load_in_memory or self.label_cache.get(index) is None:
            label = Image.open(label_path)
            if self.opt.load_in_memory:
                self.label_cache[index] = label
        else:
            label = self.label_cache[index]
        params = get_params(self.opt, label.size)
        transform_label = get_transform(self.opt, params, method=Image.NEAREST, normalized=False)
        label_tensor = transform_label(label) * 255.0
        label_tensor[label_tensor == 255] = self.opt.input_nc  # 'unknown' is opt.input_nc

        # input image (real images)
        image_path = self.image_paths[index]
        assert self.paths_match(label_path, image_path), \
            "The label_path %s and image_path %s don't match." % \
            (label_path, image_path)
        if not self.opt.load_in_memory or self.image_cache.get(index) is None:
            image = Image.open(image_path)
            if self.opt.load_in_memory:
                self.image_cache[index] = image
        else:
            image = self.image_cache[index]
        image = image.convert('RGB')

        transform_image = get_transform(self.opt, params)
        image_tensor = transform_image(image)

        # if using instance maps
        if self.opt.no_instance:
            instance_tensor = 0
        else:
            instance_path = self.instance_paths[index]
            if not self.opt.load_in_memory or self.instance_cache.get(index) is None:
                instance = Image.open(instance_path)
                if self.opt.load_in_memory:
                    self.instance_cache[index] = instance
            else:
                instance = self.instance_cache[index]
            if instance.mode == 'L':
                instance_tensor = transform_label(instance) * 255
                instance_tensor = instance_tensor.long()
            else:
                instance_tensor = transform_label(instance)

        input_dict = {
            'label': label_tensor,
            'instance': instance_tensor,
            'image': image_tensor,
            'path': image_path,
        }

        # Give subclasses a chance to modify the final output
        self.postprocess(input_dict)

        return input_dict

    def initialize(self, opt):
        self.opt = opt

        if opt.phase == 'train' and opt.meta_path is not None:
            label_paths, image_paths, instance_paths = [], [], []
            with open(opt.meta_path, 'r') as f:
                lines = f.readlines()
            for line in lines:
                splits = line.strip().split(' ')
                label_paths.append(os.path.join(opt.dataroot, splits[0]))
                image_paths.append(os.path.join(opt.dataroot, splits[1]))
                instance_paths.append(os.path.join(opt.dataroot, splits[2]))
        else:
            label_paths, image_paths, instance_paths = self.get_paths(opt)

        util.natural_sort(label_paths)
        util.natural_sort(image_paths)
        if not opt.no_instance:
            util.natural_sort(instance_paths)

        if opt.max_dataset_size > 0:
            label_paths = label_paths[:opt.max_dataset_size]
            image_paths = image_paths[:opt.max_dataset_size]
            instance_paths = instance_paths[:opt.max_dataset_size]

        if not opt.no_pairing_check:
            for path1, path2 in zip(label_paths, image_paths):
                assert self.paths_match(path1, path2), \
                    "The label-image pair (%s, %s) do not look like the right pair because the filenames are quite different. Are you sure about the pairing? Please see data/pix2pix_dataset.py to see what is going on, and use --no_pairing_check to bypass this." % (
                        path1, path2)

        self.label_paths = label_paths
        self.image_paths = image_paths
        self.instance_paths = instance_paths

        size = len(self.label_paths)
        self.dataset_size = size
        self.label_cache = {}
        self.image_cache = {}
        self.instance_cache = {}

    def postprocess(self, input_dict):
        return input_dict

    def __len__(self):
        if self.opt.max_dataset_size == -1:
            return self.dataset_size
        else:
            return self.opt.max_dataset_size

    def get_paths(self, opt):
        raise NotImplementedError
