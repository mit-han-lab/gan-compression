import argparse
import itertools
import ntpath
import os

import numpy as np
import torch
from torch import nn
from tqdm import tqdm

from data import create_eval_dataloader
from metric import create_metric_models, get_fid
from models import networks
from models.base_model import BaseModel
from models.modules.loss import GANLoss
from utils import util


class MunitModel(BaseModel):

    @staticmethod
    def modify_commandline_options(parser, is_train=False):
        assert is_train
        parser = super(MunitModel, MunitModel).modify_commandline_options(parser, is_train)
        assert isinstance(parser, argparse.ArgumentParser)
        parser.add_argument('--restore_G_A_path', type=str, default=None, help='the path to restore the generator A')
        parser.add_argument('--restore_G_B_path', type=str, default=None, help='the path to restore the generator B')
        parser.add_argument('--restore_D_A_path', type=str, default=None,
                            help='the path to restore the discriminator A')
        parser.add_argument('--restore_D_B_path', type=str, default=None,
                            help='the path to restore the discriminator B')
        parser.add_argument('--style_dim', type=int, default=8,
                            help='the dimension of the style vector')
        parser.add_argument('--n_downsample', type=int, default=2,
                            help='the number of downsample layer in the generator')
        parser.add_argument('--n_res', type=int, default=4,
                            help='the number of the ResBlock in the generator')
        parser.add_argument('--activ', type=str, default='relu',
                            help='the activation type of the generator')
        parser.add_argument('--pad_type', type=str, default='reflect',
                            help='the padding type of the generator')
        parser.add_argument('--mlp_dim', type=int, default=256,
                            help='the dimension of the mlp layer in the generator')
        parser.add_argument('--no_style_encoder', action='store_true',
                            help='whether to have the style encoder in the generator')
        parser.add_argument('--lambda_rec_x', type=float, default=10,
                            help='weight of image reconstruction loss')
        parser.add_argument('--lambda_rec_s', type=float, default=1,
                            help='weight of style reconstruction loss')
        parser.add_argument('--lambda_rec_c', type=float, default=1,
                            help='weight of content reconstruction loss')
        parser.add_argument('--lambda_gan', type=float, default=1,
                            help='weight of gan loss')
        parser.add_argument('--weight_decay', type=float, default=1e-4,
                            help='weight decay of the optimizer')
        parser.add_argument('--real_stat_A_path', type=str, required=True,
                            help='the path to load the ground-truth A images information to compute FID.')
        parser.add_argument('--real_stat_B_path', type=str, required=True,
                            help='the path to load the ground-truth B images information to compute FID.')
        parser.set_defaults(dataset_mode='unaligned', gan_mode='lsgan', load_size=256,
                            netG='munit', netD='ms_image', ndf=64, n_layers_D=4, init_type='kaiming',
                            lr_policy='step', lr=1e-4, scheduler_counter='iter',
                            nepochs=21, nepochs_decay=0, niters=1000000,
                            save_latest_freq=100000000, save_epoch_freq=1)
        return parser

    def __init__(self, opt):
        assert opt.isTrain
        assert opt.direction == 'AtoB'
        assert opt.dataset_mode == 'unaligned'
        valid_netGs = ['munit', 'mobile_munit']
        assert opt.netG in valid_netGs
        super(MunitModel, self).__init__(opt)
        self.loss_names = ['D_A', 'G_rec_xA', 'G_rec_sA', 'G_rec_cA', 'G_gan_A',
                           'D_B', 'G_rec_xB', 'G_rec_sB', 'G_rec_cB', 'G_gan_B']
        self.visual_names = ['real_A', 'fake_A', 'real_A', 'fake_B']
        self.model_names = ['G_A', 'G_B', 'D_A', 'D_B']
        self.netG_A = networks.define_G(opt.netG, init_type=opt.init_type,
                                        init_gain=opt.init_gain, gpu_ids=self.gpu_ids, opt=opt)
        self.netG_B = networks.define_G(opt.netG, init_type=opt.init_type,
                                        init_gain=opt.init_gain, gpu_ids=self.gpu_ids, opt=opt)
        self.netD_A = networks.define_D(opt.netD, input_nc=opt.input_nc, init_type='normal',
                                        init_gain=opt.init_gain, gpu_ids=self.gpu_ids, opt=opt)
        self.netD_B = networks.define_D(opt.netD, input_nc=opt.output_nc, init_type='normal',
                                        init_gain=opt.init_gain, gpu_ids=self.gpu_ids, opt=opt)

        self.criterionGAN = GANLoss(opt.gan_mode).to(self.device)
        self.criterionRec = nn.L1Loss()

        self.optimizer_G = torch.optim.Adam(itertools.chain(self.netG_A.parameters(), self.netG_B.parameters()),
                                            lr=opt.lr, betas=(opt.beta1, 0.999), weight_decay=opt.weight_decay)
        self.optimizer_D = torch.optim.Adam(itertools.chain(self.netD_A.parameters(), self.netD_B.parameters()),
                                            lr=opt.lr, betas=(opt.beta1, 0.999), weight_decay=opt.weight_decay)
        self.optimizers = [self.optimizer_G, self.optimizer_D]

        self.eval_dataloader_AtoB = create_eval_dataloader(self.opt, direction='AtoB')
        self.eval_dataloader_BtoA = create_eval_dataloader(self.opt, direction='BtoA')
        self.inception_model, _, _ = create_metric_models(opt, self.device)
        self.best_fid_A, self.best_fid_B = 1e9, 1e9
        self.fids_A, self.fids_B = [], []
        self.is_best = False
        self.npz_A = np.load(opt.real_stat_A_path)
        self.npz_B = np.load(opt.real_stat_B_path)

    def set_input(self, input):
        self.real_A = input['A'].to(self.device)
        self.real_B = input['B'].to(self.device)

    def set_single_input(self, input):
        self.real_A = input['A'].to(self.device)
        self.image_paths = input['A_paths']

    def test_single_side(self, direction):
        G_A = getattr(self, 'netG_%s' % direction[0])
        G_B = getattr(self, 'netG_%s' % direction[-1])
        opt = self.opt
        batch_size = self.real_A.size(0)
        style_dim = opt.style_dim
        with torch.no_grad():
            s = torch.randn(batch_size, style_dim, 1, 1, device=self.device)
            c, _ = G_A.encode(self.real_A, need_style=False)
            self.fake_B = G_B.decode(c, s)

    def forward(self, config=None):
        raise NotImplementedError

    def backward_G(self):
        opt = self.opt
        batch_size = self.real_A.size(0)
        style_dim = opt.style_dim

        s_a = torch.randn(batch_size, style_dim, 1, 1, device=self.device)
        s_b = torch.randn(batch_size, style_dim, 1, 1, device=self.device)
        # encode
        c_a, s_a_prime = self.netG_A.encode(self.real_A)
        c_b, s_b_prime = self.netG_B.encode(self.real_B)
        # decode (within domain)
        rec_A = self.netG_A.decode(c_a, s_a_prime)
        rec_B = self.netG_B.decode(c_b, s_b_prime)
        # decode (cross domain)
        fake_A = self.netG_A.decode(c_b, s_a)
        fake_B = self.netG_B.decode(c_a, s_b)
        # encode again
        c_b_recon, s_a_recon = self.netG_A.encode(fake_A)
        c_a_recon, s_b_recon = self.netG_B.encode(fake_B)
        # reconstruction loss
        self.loss_G_rec_xA = opt.lambda_rec_x * self.criterionRec(rec_A, self.real_A)
        self.loss_G_rec_xB = opt.lambda_rec_x * self.criterionRec(rec_B, self.real_B)
        self.loss_G_rec_sA = opt.lambda_rec_s * self.criterionRec(s_a_recon, s_a)
        self.loss_G_rec_sB = opt.lambda_rec_s * self.criterionRec(s_b_recon, s_b)
        self.loss_G_rec_cA = opt.lambda_rec_c * self.criterionRec(c_a_recon, c_a)
        self.loss_G_rec_cB = opt.lambda_rec_c * self.criterionRec(c_b_recon, c_b)
        # gan loss
        self.loss_G_gan_A = opt.lambda_gan * self.criterionGAN(self.netD_A(fake_A), True, for_discriminator=False)
        self.loss_G_gan_B = opt.lambda_gan * self.criterionGAN(self.netD_B(fake_B), True, for_discriminator=False)
        self.loss_G = self.loss_G_rec_xA + self.loss_G_rec_xB + \
                      self.loss_G_rec_sA + self.loss_G_rec_sB + \
                      self.loss_G_rec_cA + self.loss_G_rec_cB + \
                      self.loss_G_gan_A + self.loss_G_gan_B
        self.loss_G.backward()

    def backward_D(self):
        opt = self.opt
        batch_size = self.real_A.size(0)
        style_dim = opt.style_dim

        s_a = torch.randn(batch_size, style_dim, 1, 1, device=self.device)
        s_b = torch.randn(batch_size, style_dim, 1, 1, device=self.device)
        # encode
        c_a, _ = self.netG_A.encode(self.real_A, need_style=False)
        c_b, _ = self.netG_B.encode(self.real_B, need_style=False)
        # decode (cross domain)
        fake_A = self.netG_A.decode(c_b, s_a)
        fake_B = self.netG_B.decode(c_a, s_b)
        # gan loss
        self.loss_D_A = opt.lambda_gan * (self.criterionGAN(self.netD_A(fake_A.detach()), False) +
                                          self.criterionGAN(self.netD_A(self.real_A), True))
        self.loss_D_B = opt.lambda_gan * (self.criterionGAN(self.netD_B(fake_B.detach()), False) +
                                          self.criterionGAN(self.netD_B(self.real_B), True))
        self.loss_D = self.loss_D_A + self.loss_D_B
        self.loss_D.backward()

    def optimize_parameters(self, steps):
        self.set_requires_grad([self.netD_A, self.netD_B], False)  # Ds require no gradients when optimizing Gs
        self.optimizer_G.zero_grad()
        self.backward_G()
        self.optimizer_G.step()
        self.set_requires_grad([self.netD_A, self.netD_B], True)  # Ds require no gradients when optimizing Gs
        self.optimizer_D.zero_grad()  # set D_A and D_B's gradients to zero
        self.backward_D()  # calculate gradients for D_A and D_B
        self.optimizer_D.step()  # update D_A and D_B's weights

    def profile(self, config=None, verbose=True):
        raise NotImplementedError

    def test(self, config=None):
        with torch.no_grad():
            self.forward(config)

    def evaluate_model(self, step):
        ret = {}
        self.is_best = False
        save_dir = os.path.join(self.opt.log_dir, 'eval', str(step))
        os.makedirs(save_dir, exist_ok=True)
        self.netG_A.eval()
        self.netG_B.eval()
        for direction in ['AtoB', 'BtoA']:
            eval_dataloader = getattr(self, 'eval_dataloader_' + direction)
            fakes, names = [], []
            cnt = 0
            for i, data_i in enumerate(tqdm(eval_dataloader, desc='Eval %s  ' % direction, position=2, leave=False)):
                self.set_single_input(data_i)
                self.test_single_side(direction)
                fakes.append(self.fake_B.cpu())
                for j in range(len(self.image_paths)):
                    short_path = ntpath.basename(self.image_paths[j])
                    name = os.path.splitext(short_path)[0]
                    names.append(name)
                    if cnt < 10:
                        input_im = util.tensor2im(self.real_A[j])
                        fake_im = util.tensor2im(self.fake_B[j])
                        util.save_image(input_im, os.path.join(save_dir, direction, 'input', '%s.png' % name),
                                        create_dir=True)
                        util.save_image(fake_im, os.path.join(save_dir, direction, 'fake', '%s.png' % name),
                                        create_dir=True)
                    cnt += 1

            suffix = direction[-1]
            fid = get_fid(fakes, self.inception_model, getattr(self, 'npz_%s' % direction[-1]),
                          device=self.device, batch_size=self.opt.eval_batch_size, tqdm_position=2)
            if fid < getattr(self, 'best_fid_%s' % suffix):
                self.is_best = True
                setattr(self, 'best_fid_%s' % suffix, fid)
            fids = getattr(self, 'fids_%s' % suffix)
            fids.append(fid)
            if len(fids) > 3:
                fids.pop(0)
            ret['metric/fid_%s' % suffix] = fid
            ret['metric/fid_%s-mean' % suffix] = sum(getattr(self, 'fids_%s' % suffix)) / len(
                getattr(self, 'fids_%s' % suffix))
            ret['metric/fid_%s-best' % suffix] = getattr(self, 'best_fid_%s' % suffix)

        self.netG_A.train()
        self.netG_B.train()
        return ret
