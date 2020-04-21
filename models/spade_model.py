import argparse

import torch
from torch import nn
from torchprofile import profile_macs

from models import networks
from .base_model import BaseModel


class SPADEModel(BaseModel):
    @staticmethod
    def modify_commandline_options(parser, is_train):
        assert isinstance(parser, argparse.ArgumentParser)
        parser.add_argument('--z_dim', type=int, default=256,
                            help="dimension of the latent z vector")
        parser.set_defaults(netG='sub_mobile_spade')
        parser = networks.modify_commandline_options(parser, is_train)
        return parser

    def __init__(self, opt):
        super(SPADEModel, self).__init__(opt)
        self.model_names = ['G']
        self.visual_names = ['labels', 'fake_B', 'real_B']
        self.netG = networks.define_G(opt.input_nc, opt.output_nc, opt.ngf, opt.netG,
                                      opt.norm, opt.dropout_rate, opt.init_type,
                                      opt.init_gain, self.gpu_ids, opt=opt)
        if opt.isTrain:
            raise NotImplementedError("Training mode of SPADE is currently not supported!!!")
        else:
            self.netG.eval()

    def set_input(self, input):
        self.data = input
        self.image_paths = input['path']
        self.labels = input['label'].to(self.device)
        self.input_semantics, self.real_B = self.preprocess_input(input)

    def test(self, config=None):
        with torch.no_grad():
            self.forward()

    def preprocess_input(self, data):
        # move to GPU and change data types
        data['label'] = data['label'].long()
        # if self.use_gpu():
        data['label'] = data['label'].to(self.device)
        data['instance'] = data['instance'].to(self.device)
        data['image'] = data['image'].to(self.device)

        # create one-hot label map
        label_map = data['label']
        bs, _, h, w = label_map.size()
        nc = self.opt.input_nc + 1 if self.opt.contain_dontcare_label \
            else self.opt.input_nc
        input_label = torch.zeros([bs, nc, h, w], device=self.device)
        input_semantics = input_label.scatter_(1, label_map, 1.0)

        # concatenate instance map if it exists
        if not self.opt.no_instance:
            inst_map = data['instance']
            instance_edge_map = self.get_edges(inst_map)
            input_semantics = torch.cat((input_semantics, instance_edge_map), dim=1)

        return input_semantics, data['image']

    def forward(self):
        self.fake_B = self.netG(self.input_semantics)

    def get_edges(self, t):
        edge = torch.zeros(t.size(), dtype=torch.uint8, device=self.device)
        edge[:, :, :, 1:] = edge[:, :, :, 1:] | ((t[:, :, :, 1:] != t[:, :, :, :-1]).byte())
        edge[:, :, :, :-1] = edge[:, :, :, :-1] | ((t[:, :, :, 1:] != t[:, :, :, :-1]).byte())
        edge[:, :, 1:, :] = edge[:, :, 1:, :] | ((t[:, :, 1:, :] != t[:, :, :-1, :]).byte())
        edge[:, :, :-1, :] = edge[:, :, :-1, :] | ((t[:, :, 1:, :] != t[:, :, :-1, :]).byte())
        return edge.float()

    def profile(self, config=None, verbose=True):
        netG = self.netG
        if isinstance(netG, nn.DataParallel):
            netG = netG.module
        if config is not None:
            netG.configs = config
        with torch.no_grad():
            macs = profile_macs(netG, (self.input_semantics,))
        params = 0
        for p in netG.parameters():
            params += p.numel()
        if verbose:
            print('MACs: %.3fG\tParams: %.3fM' % (macs / 1e9, params / 1e6), flush=True)
        return macs, params

    def optimize_parameters(self):
        raise NotImplementedError
