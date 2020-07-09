import os
from abc import ABC, abstractmethod
from collections import OrderedDict

import torch

from utils import util
from . import networks
from torch.nn import DataParallel

class BaseModel(ABC):
    """This class is an abstract base class (ABC) for models.
    To create a subclass, you need to implement the following five functions:
        -- <__init__>:                      initialize the class; first call BaseModel.__init__(self, opt).
        -- <set_input>:                     unpack data from dataset and apply preprocessing.
        -- <forward>:                       produce intermediate results.
        -- <optimize_parameters>:           calculate losses, gradients, and update network weights.
        -- <modify_commandline_options>:    (optionally) add model-specific options and set default options.
    """

    def __init__(self, opt):
        """Initialize the BaseModel class.

        Parameters:
            opt (Option class)-- stores all the experiment flags; needs to be a subclass of BaseOptions

        When creating your custom class, you need to implement your own initialization.
        In this fucntion, you should first call <BaseModel.__init__(self, opt)>
        Then, you need to define four lists:
            -- self.loss_names (str list):          specify the training losses that you want to plot and save.
            -- self.model_names (str list):         specify the images that you want to display and save.
            -- self.visual_names (str list):        define networks used in our training.
            -- self.optimizers (optimizer list):    define and initialize optimizers. You can define one optimizer for each network. If two networks are updated at the same time, you can use itertools.chain to group them. See cycle_gan_model.py for an example.
        """
        self.opt = opt
        self.gpu_ids = opt.gpu_ids
        self.isTrain = opt.isTrain
        self.device = torch.device('cuda:{}'.format(self.gpu_ids[0])) if self.gpu_ids \
            else torch.device('cpu')  # get device name: CPU or GPU
        if opt.isTrain:
            self.save_dir = os.path.join(opt.log_dir, 'checkpoints')  # save all the checkpoints to save_dir
            os.makedirs(self.save_dir, exist_ok=True)
        if opt.preprocess != 'scale_width':  # with [scale_width], input images might have different sizes, which hurts the performance of cudnn.benchmark.
            torch.backends.cudnn.benchmark = True
        self.model_names = []
        self.visual_names = []
        self.image_paths = []
        self.metric = 0  # used for learning rate policy 'plateau'

    @staticmethod
    def modify_commandline_options(parser, is_train):
        """Add new model-specific options, and rewrite default values for existing options.

        Parameters:
            parser          -- original option parser
            is_train (bool) -- whether training phase or test phase. You can use this flag to add training-specific or test-specific options.

        Returns:
            the modified parser.
        """
        return parser

    @abstractmethod
    def set_input(self, input):
        """Unpack input data from the dataloader and perform necessary pre-processing steps.

        Parameters:
            input (dict): includes the data itself and its metadata information.
        """
        pass

    @abstractmethod
    def forward(self):
        """Run forward pass; called by both functions <optimize_parameters> and <test>."""
        pass

    @abstractmethod
    def optimize_parameters(self, steps):
        """Calculate losses, gradients, and update network weights; called in every training iteration"""
        pass

    def setup(self, opt, verbose=True):
        """Load and print networks; create schedulers

        Parameters:
            opt (Option class) -- stores all the experiment flags; needs to be a subclass of BaseOptions
        """
        if self.isTrain:
            self.schedulers = [networks.get_scheduler(optimizer, opt) for optimizer in self.optimizers]
        self.load_networks(verbose=verbose)
        if verbose:
            self.print_networks()

    def print_networks(self):
        print('---------- Networks initialized -------------')
        for name in self.model_names:
            if isinstance(name, str):
                net = getattr(self, 'net' + name)
                num_params = 0
                for param in net.parameters():
                    num_params += param.numel()
                print(net)
                print('[Network %s] Total number of parameters : %.3f M' % (name, num_params / 1e6))
                if hasattr(self.opt, 'log_dir'):
                    with open(os.path.join(self.opt.log_dir, 'net' + name + '.txt'), 'w') as f:
                        f.write(str(net) + '\n')
                        f.write('[Network %s] Total number of parameters : %.3f M\n' % (name, num_params / 1e6))
        print('-----------------------------------------------')

    def eval(self):
        """Make models eval mode during test time"""
        for name in self.model_names:
            if isinstance(name, str):
                net = getattr(self, 'net' + name)
                net.eval()

    def train(self):
        """Make models eval mode during test time"""
        for name in self.model_names:
            if isinstance(name, str):
                net = getattr(self, 'net' + name)
                net.train()

    def test(self):
        """Forward function used in test time.

        This function wraps <forward> function in no_grad() so we don't save intermediate steps for backprop
        It also calls <compute_visuals> to produce additional visualization results
        """
        with torch.no_grad():
            self.forward()

    def get_image_paths(self):
        """ Return image paths that are used to load current data"""
        return self.image_paths

    def update_learning_rate(self, logger=None):
        for scheduler in self.schedulers:
            if self.opt.lr_policy == 'plateau':
                scheduler.step(self.opt.metric)
            else:
                scheduler.step()
        lr = self.optimizers[0].param_groups[0]['lr']
        if logger is not None:
            logger.print_info('learning rate = %.7f\n' % lr)
        else:
            print('learning rate = %.7f' % lr)

    def get_current_visuals(self):
        """Return visualization images. train.py will display these images with visdom, and save the images to a HTML"""
        visual_ret = OrderedDict()
        for name in self.visual_names:
            if isinstance(name, str) and hasattr(self, name):
                visual_ret[name] = getattr(self, name)
        return visual_ret

    def get_current_losses(self):
        errors_set = OrderedDict()
        for name in self.loss_names:
            if not hasattr(self, 'loss_' + name):
                continue
            key = name

            def has_number(s):
                for i in range(10):
                    if str(i) in s:
                        return True
                return False

            if has_number(key):
                key = 'Specific_loss/' + key
            elif key.startswith('D_'):
                key = 'D_loss/' + key
            elif key.startswith('G_'):
                key = 'G_loss/' + key
            else:
                assert False
            errors_set[key] = float(getattr(self, 'loss_' + name))
        return errors_set

    def load_networks(self, verbose=True):
        for name in self.model_names:
            net = getattr(self, 'net' + name, None)
            path = getattr(self.opt, 'restore_%s_path' % name, None)
            if path is not None:
                util.load_network(net, path, verbose)

    def save_networks(self, epoch):
        for name in self.model_names:
            if isinstance(name, str):
                save_filename = '%s_net_%s.pth' % (epoch, name)
                save_path = os.path.join(self.save_dir, save_filename)
                net = getattr(self, 'net' + name)
                if len(self.gpu_ids) > 0 and torch.cuda.is_available():
                    if isinstance(net, DataParallel):
                        torch.save(net.module.cpu().state_dict(), save_path)
                    else:
                        torch.save(net.cpu().state_dict(), save_path)
                    net.cuda(self.gpu_ids[0])
                else:
                    torch.save(net.cpu().state_dict(), save_path)

    def set_requires_grad(self, nets, requires_grad=False):
        """Set requies_grad=Fasle for all the networks to avoid unnecessary computations
        Parameters:
            nets (network list)   -- a list of networks
            requires_grad (bool)  -- whether the networks require gradients or not
        """
        if not isinstance(nets, list):
            nets = [nets]
        for net in nets:
            if net is not None:
                for param in net.parameters():
                    param.requires_grad = requires_grad

    def evaluate_model(self, step):
        raise NotImplementedError

    def profile(self):
        raise NotImplementedError
