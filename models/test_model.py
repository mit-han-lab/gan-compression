'''
This is the test model tailored for ResNet architecture.
'''
import torch
from torch import nn
from torchprofile import profile_macs

from models import networks
from models.base_model import BaseModel


class TestModel(BaseModel):
    @staticmethod
    def modify_commandline_options(parser, is_train):
        assert not is_train
        parser.add_argument('--restore_G_path', type=str, required=True,
                            help='the path to restore the generator')
        return parser

    def __init__(self, opt):
        super(TestModel, self).__init__(opt)
        self.visual_names = ['real_A', 'fake_B', 'real_B']
        self.model_names = ['G']
        self.netG = networks.define_G(opt.netG, input_nc=opt.input_nc, output_nc=opt.output_nc, ngf=opt.ngf,
                                      norm=opt.norm, dropout_rate=opt.dropout_rate, gpu_ids=self.gpu_ids, opt=opt)
        self.netG.eval()

    def set_input(self, input):
        AtoB = self.opt.dataset_mode == 'single' or self.opt.direction == 'AtoB'
        self.real_A = input['A' if AtoB else 'B'].to(self.device)
        if self.opt.dataset_mode != 'single':
            self.real_B = input['B' if AtoB else 'A'].to(self.device)
        self.image_paths = input['A_paths' if AtoB else 'B_paths']

    def forward(self, config=None):
        if config is not None:
            self.netG.configs = config
        self.fake_B = self.netG(self.real_A)

    def optimize_parameters(self, steps):
        assert False, 'This model is only for testing, you cannot optimize the parameters!!!'

    def save_networks(self, epoch):
        assert False, 'This model is only for testing!!!'

    def profile(self, config=None, verbose=True):
        netG = self.netG
        if isinstance(netG, nn.DataParallel):
            netG = netG.module
        if config is not None:
            netG.configs = config
        with torch.no_grad():
            macs = profile_macs(netG, (self.real_A[:1],))
        params = 0
        for p in netG.parameters():
            params += p.numel()
        if verbose:
            print('MACs: %.3fG\tParams: %.3fM' % (macs / 1e9, params / 1e6), flush=True)
        return macs, params

    def test(self, config=None):
        with torch.no_grad():
            self.forward(config)

    def get_current_losses(self):
        assert False, 'This model is only for testing!!!'

    def update_learning_rate(self, epoch, total_iter, logger=None):
        assert False, 'This model is only for testing!!!'
