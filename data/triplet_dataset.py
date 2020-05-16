import os.path

from PIL import Image

from data.base_dataset import BaseDataset, get_params, get_transform
from data.image_folder import make_dataset

import torch

class TripletDataset(BaseDataset):
    """A dataset class for paired image dataset.

    It assumes that the directory '/path/to/data/train' contains image triplet in the form of {A, D, B}.
    During test time, you need to prepare a directory '/path/to/data/test'.
    """

    def __init__(self, opt):
        """Initialize this dataset class.

        Parameters:
            opt (Option class) -- stores all the experiment flags; needs to be a subclass of BaseOptions
        """
        BaseDataset.__init__(self, opt)
        self.dir_ADB = os.path.join(opt.dataroot, opt.phase)  # get the image directory
        self.ADB_paths = sorted(make_dataset(self.dir_ADB))  # get image paths
        # print(self.ADB_paths)

        assert (self.opt.load_size >= self.opt.crop_size)  # crop_size should be smaller than the size of loaded image
        self.input_nc = self.opt.output_nc if self.opt.direction == 'BtoA' else self.opt.input_nc
        self.output_nc = self.opt.input_nc if self.opt.direction == 'BtoA' else self.opt.output_nc
        self.cache = {}

    @staticmethod
    def modify_commandline_options(parser, is_train):
        """Add new dataset-specific options, and rewrite default values for existing options.

        Parameters:
            parser          -- original option parser
            is_train (bool) -- whether training phase or test phase.
        Returns:
            the modified parser.
        """
        parser.set_defaults(input_nc=6, output_nc=3)  # specify dataset-specific default values
        return parser
        
    def __getitem__(self, index):
        ADB_path = self.ADB_paths[index]
        if not self.opt.load_in_memory or self.cache.get(index) is None:
            ADB = Image.open(ADB_path).convert('RGB')
            if self.opt.load_in_memory:
                self.cache[index] = ADB
        else:
            ADB = self.cache[index]

        # split ADB image into A , D and B
        w, h = ADB.size
        w3 = int(w / 3)
        A = ADB.crop((0, 0, w3, h))
        D = ADB.crop((w3, 0, 2 * w3, h))
        B = ADB.crop((2 * w3, 0, w, h))

        # apply the same transform to both A and B
        transform_params = get_params(self.opt, A.size)
        A_transform = get_transform(self.opt, transform_params, grayscale=(self.input_nc // 2 == 1))
        D_transform = get_transform(self.opt, transform_params, grayscale=(self.input_nc // 2 == 1))
        B_transform = get_transform(self.opt, transform_params, grayscale=(self.output_nc == 1))

        A = A_transform(A)
        D = D_transform(D)
        B = B_transform(B)
        
        #concatenate input
        A = torch.cat([A, D], dim=0)
        return {'A': A, 'B': B, 'A_paths': ADB_path, 'B_paths': ADB_path}

    def __len__(self):
        """Return the total number of images in the dataset."""
        if self.opt.max_dataset_size == -1:
            return len(self.ADB_paths)
        else:
            return self.opt.max_dataset_size
