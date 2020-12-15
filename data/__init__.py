import copy
import importlib
import os

import torch.utils.data

from data.base_dataset import BaseDataset


def find_dataset_using_name(dataset_name):
    dataset_filename = "data." + dataset_name + "_dataset"
    datasetlib = importlib.import_module(dataset_filename)

    dataset = None
    target_dataset_name = dataset_name.replace('_', '') + 'dataset'
    for name, cls in datasetlib.__dict__.items():
        if name.lower() == target_dataset_name.lower() \
                and issubclass(cls, BaseDataset):
            dataset = cls

    if dataset is None:
        raise NotImplementedError(
            "In %s.py, there should be a subclass of BaseDataset with class name that matches %s in lowercase." % (
                dataset_filename, target_dataset_name))

    return dataset


def get_option_setter(dataset_name):
    """Return the static method <modify_commandline_options> of the dataset class."""
    dataset_class = find_dataset_using_name(dataset_name)
    return dataset_class.modify_commandline_options


def create_dataloader(opt, verbose=True):
    """Create a dataset given the option.

    This function wraps the class CustomDatasetDataLoader.
        This is the main interface between this package and 'train.py'/'test.py'

    Example:
        >>> from data import create_dataloader
        >>> dataset = create_dataloader(opt)
    """
    dataloader = CustomDatasetDataLoader(opt, verbose)
    dataloader = dataloader.load_data()
    return dataloader


def create_eval_dataloader(opt, direction=None):
    opt = copy.deepcopy(opt)
    # Set some evaluation options
    # opt.prepocess = 'resize_and_crop'
    opt.load_size = opt.crop_size
    opt.no_flip = True
    opt.serial_batches = True
    opt.batch_size = opt.eval_batch_size
    opt.phase = 'val'
    if opt.dataset_mode == 'unaligned':
        assert direction is not None
        opt.dataset_mode = 'single'
        opt.dataroot = os.path.join(opt.dataroot, 'val%s' % (direction[0]))
    dataloader = CustomDatasetDataLoader(opt)
    dataloader = dataloader.load_data()
    return dataloader


def create_train_dataloader(opt):
    opt = copy.deepcopy(opt)
    # Set some evaluation options
    # opt.prepocess = 'resize_and_crop'
    opt.no_flip = False
    opt.serial_batches = False
    opt.phase = 'train'
    opt.load_in_memory = False
    opt.max_dataset_size = 512
    dataloader = CustomDatasetDataLoader(opt)
    dataloader = dataloader.load_data()
    return dataloader


class CustomDatasetDataLoader():
    """Wrapper class of Dataset class that performs multi-threaded data loading"""

    def __init__(self, opt, verbose=True):
        """Initialize this class

        Step 1: create a dataset instance given the name [dataset_mode]
        Step 2: create a multi-threaded data loader.
        """
        self.opt = opt
        dataset_class = find_dataset_using_name(opt.dataset_mode)
        self.dataset = dataset_class(opt)
        # print(len(self.dataset))
        if verbose:
            print("dataset [%s] was created" % type(self.dataset).__name__)
        self.dataloader = torch.utils.data.DataLoader(
            self.dataset,
            batch_size=opt.batch_size,
            shuffle=not opt.serial_batches,
            num_workers=opt.num_threads)

    def load_data(self):
        return self

    def __len__(self):
        """Return the number of data in the dataset"""
        return (len(self.dataset) + self.opt.batch_size - 1) // self.opt.batch_size

    def __iter__(self):
        """Return a batch of data"""
        # print(len(self.dataloader))
        for i, data in enumerate(self.dataloader):
            yield data
