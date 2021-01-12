import argparse
import copy

import torch
from torch import nn
from torchprofile import profile_macs

from data import create_dataloader
from models import networks
from models.base_model import BaseModel


class MunitTestModel(BaseModel):

    @staticmethod
    def modify_commandline_options(parser, is_train=False):
        assert not is_train
        parser = super(MunitTestModel, MunitTestModel).modify_commandline_options(parser, is_train)
        assert isinstance(parser, argparse.ArgumentParser)
        parser.add_argument('--restore_G_A_path', type=str, required=True, help='the path to restore the generator A')
        parser.add_argument('--restore_G_B_path', type=str, default=None, help='the path to restore the generator B')
        parser.add_argument('--style_dim', type=int, default=8,
                            help='the dimension of the style vector')
        parser.add_argument('--n_downsample', type=int, default=2,
                            help='the number of downsample layer in the generator')
        parser.add_argument('--n_res', type=int, default=4,
                            help='the number of the ResBlock in the generator')
        parser.add_argument('--activ', type=str, default='relu',
                            help='the activation type of the generator')
        parser.add_argument('--pad_type', type=str, default='reflect',
                            help='the padding type of the generator and the discriminator')
        parser.add_argument('--mlp_dim', type=int, default=256,
                            help='the dimension of the mlp layer in the generator')
        parser.add_argument('--no_style_encoder', action='store_true',
                            help='whether to have the style encoder in the generator')
        parser.add_argument('--ref_root', type=str, default=None,
                            help='directory to load the style reference images')
        parser.set_defaults(dataset_mode='single', netG='munit')
        return parser

    def __init__(self, opt):
        valid_netGs = ['munit', 'super_munit', 'super_mobile_munit', 'sub_mobile_munit']
        assert opt.netG in valid_netGs
        super(MunitTestModel, self).__init__(opt)
        self.visual_names = ['real_A', 'fake_B', 'real_B']
        self.netG_A = networks.define_G(opt.netG, gpu_ids=self.gpu_ids, opt=opt)
        self.netG_A.eval()
        self.model_names = ['G_A']
        if opt.restore_G_B_path is not None:
            self.netG_B = networks.define_G(opt.netG, gpu_ids=self.gpu_ids, opt=opt)
            self.netG_B.eval()
            self.model_names.append('G_B')
        if opt.ref_root is not None:
            opt_ref = copy.deepcopy(opt)
            opt_ref.dataroot = opt.ref_root
            self.ref_loader = create_dataloader(opt_ref)

    def set_input(self, input):
        self.real_A = input['A'].to(self.device)
        if self.opt.dataset_mode != 'single':
            self.real_B = input['B'].to(self.device)
        self.image_paths = input['A_paths']

    def forward(self, config=None):
        if hasattr(self, 'netG_B'):
            netG_B = self.netG_B
        else:
            netG_B = self.netG_A
        if config is not None:
            self.netG_A.configs = config
            netG_B.configs = config
        content, _ = self.netG_A.encode(self.real_A, need_style=False)
        if hasattr(self, 'ref_loader'):
            ref_B = self.ref_loader.sample().to(self.device)
            _, style = netG_B.encode(ref_B, need_content=False)
        else:
            style = torch.randn(self.real_A.size(0), self.opt.style_dim, 1, 1, device=self.device)
        self.fake_B = netG_B.decode(content, style)

    def optimize_parameters(self, steps):
        assert False, 'This model is only for testing, you cannot optimize the parameters!!!'

    def save_networks(self, epoch):
        assert False, 'This model is only for testing!!!'

    def profile(self, config=None, verbose=True):
        netG = self.netG_A
        if isinstance(netG, nn.DataParallel):
            netG = netG.module
        if config is not None:
            netG.configs = config
        style = torch.randn(1, self.opt.style_dim, 1, 1, device=self.device)
        with torch.no_grad():
            macs_no_style_encoder = profile_macs(netG, (self.real_A[:1], style))
        if not self.opt.no_style_encoder:
            macs_with_style_encoder = profile_macs(netG, (self.real_A[:1]))
        params = 0
        for p in netG.parameters():
            params += p.numel()
        if verbose:
            if self.opt.no_style_encoder:
                print('MACs (no style encoder): %.3fG\tParams: %.3fM' %
                      (macs_no_style_encoder / 1e9, params / 1e6), flush=True)
            else:
                print('MACs (no style encoder): %.3fG\tMACs (with style encoder): %.3fG\tParams: %.3fM' %
                      (macs_no_style_encoder / 1e9, (macs_with_style_encoder) / 1e9, params / 1e6), flush=True)
        return macs_with_style_encoder, params

    def test(self, config=None):
        with torch.no_grad():
            self.forward(config)

    def get_current_losses(self):
        assert False, 'This model is only for testing!!!'

    def update_learning_rate(self, epoch, total_iter, logger=None):
        assert False, 'This model is only for testing!!!'
