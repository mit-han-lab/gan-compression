import copy
import itertools
import os
from argparse import ArgumentParser

import numpy as np
import torch
from torch import nn
from torch.nn import DataParallel

from data import create_eval_dataloader
from metric import create_metric_models
from models import networks
from models.base_model import BaseModel
from models.modules.loss import GANLoss
from models.modules.super_modules import SuperConv2d
from utils import util


class BaseMunitDistiller(BaseModel):
    @staticmethod
    def modify_commandline_options(parser, is_train):
        assert is_train
        parser = super(BaseMunitDistiller, BaseMunitDistiller).modify_commandline_options(parser, is_train)
        assert isinstance(parser, ArgumentParser)
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
        parser.add_argument('--teacher_mlp_dim', type=int, default=256,
                            help='the dimension of the mlp layer in the teacher generator')
        parser.add_argument('--student_mlp_dim', type=int, default=256,
                            help='the dimension of the mlp layer in the teacher generator')
        parser.add_argument('--teacher_no_style_encoder', action='store_true',
                            help='whether to have the style encoder in the generator')
        parser.add_argument('--student_no_style_encoder', action='store_true',
                            help='whether to have the style encoder in the generator')
        parser.add_argument('--lambda_rec_x', type=float, default=10,
                            help='weight of image reconstruction loss')
        parser.add_argument('--lambda_rec_c', type=float, default=1,
                            help='weight of content reconstruction loss')
        parser.add_argument('--lambda_rec_s', type=float, default=1,
                            help='weight of style reconstruction loss')
        parser.add_argument('--lambda_gan', type=float, default=1,
                            help='weight of gan loss')
        parser.add_argument('--weight_decay', type=float, default=1e-4,
                            help='weight decay of the optimizer')
        parser.add_argument('--style_encoder_step', type=int, default=2)
        parser.set_defaults(teacher_netG='munit', teacher_ngf=64,
                            student_netG='munit', student_ngf=24,
                            dataset_mode='unaligned', gan_mode='lsgan', load_size=256,
                            netD='ms_image', ndf=64, n_layers_D=4, init_type='kaiming',
                            lr_policy='step', lr=1e-4, scheduler_counter='iter',
                            nepochs=21, nepochs_decay=0, niters=1000000,
                            save_latest_freq=100000000, save_epoch_freq=1)
        return parser

    def create_option(self, role):
        assert role in ['teacher', 'student']
        opt = copy.deepcopy(self.opt)
        opt.ngf = getattr(self.opt, '%s_ngf' % role)
        opt.mlp_dim = getattr(self.opt, '%s_mlp_dim' % role)
        opt.no_style_encoder = getattr(self.opt, '%s_no_style_encoder' % role)
        return opt

    def __init__(self, opt):
        assert opt.isTrain
        valid_netGs = ['munit', 'super_munit', 'super_mobile_munit', 'super_mobile_munit2', 'super_mobile_munit3']
        assert opt.teacher_netG in valid_netGs and opt.student_netG in valid_netGs
        super(BaseMunitDistiller, self).__init__(opt)
        self.loss_names = ['G_gan', 'G_rec_x', 'G_rec_c', 'G_rec_s', 'D_fake', 'D_real']
        if not opt.student_no_style_encoder:
            self.loss_names.append('G_rec_s')
        self.optimizers = []
        self.image_paths = []
        self.visual_names = ['real_A', 'Sfake_B', 'Tfake_B', 'real_B']
        self.model_names = ['netG_student', 'netG_teacher', 'netD']
        opt_teacher = self.create_option('teacher')
        self.netG_teacher = networks.define_G(opt.teacher_netG, init_type=opt.init_type,
                                              init_gain=opt.init_gain, gpu_ids=self.gpu_ids, opt=opt_teacher)
        opt_student = self.create_option('student')
        self.netG_student = networks.define_G(opt.student_netG, init_type=opt.init_type,
                                              init_gain=opt.init_gain, gpu_ids=self.gpu_ids, opt=opt_student)
        self.netD = networks.define_D(opt.netD, input_nc=opt.output_nc, init_type='normal',
                                      init_gain=opt.init_gain, gpu_ids=self.gpu_ids, opt=opt)
        if hasattr(opt, 'distiller'):
            self.netA = nn.Conv2d(in_channels=4 * opt.student_ngf,
                                  out_channels=4 * opt.teacher_ngf,
                                  kernel_size=1).to(self.device)
        else:
            self.netA = SuperConv2d(in_channels=4 * opt.student_ngf,
                                    out_channels=4 * opt.teacher_ngf,
                                    kernel_size=1).to(self.device)
        networks.init_net(self.netA)
        self.netG_teacher.eval()

        self.criterionGAN = GANLoss(opt.gan_mode).to(self.device)
        self.criterionRec = torch.nn.L1Loss()

        G_params = []
        G_params.append(self.netG_student.parameters())
        G_params.append(self.netA.parameters())
        self.optimizer_G = torch.optim.Adam(itertools.chain(*G_params), lr=opt.lr,
                                            betas=(opt.beta1, 0.999), weight_decay=opt.weight_decay)
        self.optimizer_D = torch.optim.Adam(self.netD.parameters(), lr=opt.lr,
                                            betas=(opt.beta1, 0.999), weight_decay=opt.weight_decay)
        self.optimizers.append(self.optimizer_G)
        self.optimizers.append(self.optimizer_D)

        self.eval_dataloader = create_eval_dataloader(self.opt, direction=opt.direction)
        self.inception_model, _, _ = create_metric_models(opt, device=self.device)
        self.npz = np.load(opt.real_stat_path)
        self.is_best = False

    def setup(self, opt, verbose=True):
        super(BaseMunitDistiller, self).setup(opt, verbose)

    def set_input(self, input):
        AtoB = self.opt.direction == 'AtoB'
        self.real_A = input['A' if AtoB else 'B'].to(self.device)
        self.real_B = input['B' if AtoB else 'A'].to(self.device)
        self.image_paths = input['A_paths' if AtoB else 'B_paths']

    def set_single_input(self, input):
        self.real_A = input['A'].to(self.device)
        self.image_paths = input['A_paths']

    def forward(self, need_style_encoder=False):
        raise NotImplementedError

    def forward_A(self, c):
        raise NotImplementedError

    def backward_D(self):
        opt = self.opt
        fake = self.Sfake_B.detach()
        real = self.real_B.detach()
        self.loss_D_fake = opt.lambda_gan * self.criterionGAN(self.netD(fake), False, for_discriminator=True)
        self.loss_D_real = opt.lambda_gan * self.criterionGAN(self.netD(real), True, for_discriminator=True)
        self.loss_D = self.loss_D_fake + self.loss_D_real
        self.loss_D.backward()

    def backward_G(self, need_style_encoder=False):
        opt = self.opt
        self.loss_G_rec_x = opt.lambda_rec_x * self.criterionRec(self.Sfake_B, self.Tfake_B)
        self.loss_G_rec_c = opt.lambda_rec_c * self.criterionRec(self.forward_A(self.c_student), self.c_teacher)
        if need_style_encoder:
            self.loss_G_rec_s = opt.lambda_rec_s * self.criterionRec(self.s_student, self.s_teacher)
        self.loss_G_gan = opt.lambda_gan * self.criterionGAN(self.netD(self.Sfake_B), True, for_discriminator=False)
        self.loss_G = self.loss_G_rec_x + self.loss_G_rec_c + \
                      (self.loss_G_rec_s if need_style_encoder else 0) + self.loss_G_gan
        self.loss_G.backward()

    def optimize_parameters(self, steps):
        need_style_encoder = False if self.opt.student_no_style_encoder \
            else steps % self.opt.style_encoder_step != 0
        self.forward(need_style_encoder=need_style_encoder)
        self.set_requires_grad(self.netD, False)  # Ds require no gradients when optimizing Gs
        self.optimizer_G.zero_grad()
        self.backward_G(need_style_encoder=need_style_encoder)
        self.optimizer_G.step()
        self.set_requires_grad(self.netD, True)  # Ds require no gradients when optimizing Gs
        self.optimizer_D.zero_grad()
        self.backward_D()
        self.optimizer_D.step()

    def print_networks(self):
        print('---------- Networks initialized -------------')
        for name in self.model_names:
            if hasattr(self, name):
                net = getattr(self, name)
                num_params = 0
                for param in net.parameters():
                    num_params += param.numel()
                print(net)
                print('[Network %s] Total number of parameters : %.3f M' % (name, num_params / 1e6))
                with open(os.path.join(self.opt.log_dir, name + '.txt'), 'w') as f:
                    f.write(str(net) + '\n')
                    f.write('[Network %s] Total number of parameters : %.3f M\n' % (name, num_params / 1e6))
        print('-----------------------------------------------')

    def load_networks(self, verbose=True):
        util.load_network(self.netG_teacher, self.opt.restore_teacher_G_path, verbose)
        if self.opt.restore_student_G_path is not None:
            util.load_network(self.netG_student, self.opt.restore_student_G_path, verbose)
        if self.opt.restore_D_path is not None:
            util.load_network(self.netD, self.opt.restore_D_path, verbose)
        if self.opt.restore_A_path is not None:
            for i, netA in enumerate(self.netAs):
                path = '%s-%d.pth' % (self.opt.restore_A_path, i)
                util.load_network(netA, path, verbose)
        if self.opt.restore_O_path is not None:
            for i, optimizer in enumerate(self.optimizers):
                path = '%s-%d.pth' % (self.opt.restore_O_path, i)
                util.load_optimizer(optimizer, path, verbose)

    def save_networks(self, epoch):

        def save_net(net, save_path):
            if len(self.gpu_ids) > 0 and torch.cuda.is_available():
                if isinstance(net, DataParallel):
                    torch.save(net.module.cpu().state_dict(), save_path)
                else:
                    torch.save(net.cpu().state_dict(), save_path)
                net.cuda(self.gpu_ids[0])
            else:
                torch.save(net.cpu().state_dict(), save_path)

        save_filename = '%s_net_%s.pth' % (epoch, 'G')
        save_path = os.path.join(self.save_dir, save_filename)
        net = getattr(self, 'net%s_student' % 'G')
        save_net(net, save_path)

        save_filename = '%s_net_%s.pth' % (epoch, 'D')
        save_path = os.path.join(self.save_dir, save_filename)
        net = getattr(self, 'net%s' % 'D')
        save_net(net, save_path)

        for i, optimizer in enumerate(self.optimizers):
            save_filename = '%s_optim-%d.pth' % (epoch, i)
            save_path = os.path.join(self.save_dir, save_filename)
            torch.save(optimizer.state_dict(), save_path)

    def evaluate_model(self, step):
        raise NotImplementedError

    def test(self):
        with torch.no_grad():
            self.forward()
