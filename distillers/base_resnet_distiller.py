import itertools
import os

import numpy as np
import torch
from torch import nn
from torch.nn import DataParallel

import models.modules.loss
from data import create_eval_dataloader
from metric.inception import InceptionV3
from metric.mIoU_score import DRNSeg
from models import networks
from models.base_model import BaseModel
from models.modules.super_modules import SuperConv2d
from utils import util


class BaseResnetDistiller(BaseModel):
    @staticmethod
    def modify_commandline_options(parser, is_train):
        assert is_train
        parser = super(BaseResnetDistiller, BaseResnetDistiller).modify_commandline_options(parser, is_train)
        parser.add_argument('--teacher_netG', type=str, default='mobile_resnet_9blocks',
                            help='specify teacher generator architecture',
                            choices=['resnet_9blocks', 'mobile_resnet_9blocks',
                                     'super_mobile_resnet_9blocks', 'sub_mobile_resnet_9blocks'])
        parser.add_argument('--student_netG', type=str, default='mobile_resnet_9blocks',
                            help='specify student generator architecture',
                            choices=['resnet_9blocks', 'mobile_resnet_9blocks',
                                     'super_mobile_resnet_9blocks', 'sub_mobile_resnet_9blocks'])
        parser.add_argument('--teacher_ngf', type=int, default=64,
                            help='the base number of filters of the teacher generator')
        parser.add_argument('--student_ngf', type=int, default=48,
                            help='the base number of filters of the student generator')
        parser.add_argument('--restore_teacher_G_path', type=str, required=True,
                            help='the path to restore the teacher generator')
        parser.add_argument('--restore_student_G_path', type=str, default=None,
                            help='the path to restore the student generator')
        parser.add_argument('--restore_A_path', type=str, default=None,
                            help='the path to restore the adaptors for distillation')
        parser.add_argument('--restore_D_path', type=str, default=None,
                            help='the path to restore the discriminator')
        parser.add_argument('--restore_O_path', type=str, default=None,
                            help='the path to restore the optimizer')
        parser.add_argument('--recon_loss_type', type=str, default='l1',
                            choices=['l1', 'l2', 'smooth_l1', 'vgg'],
                            help='the type of the reconstruction loss')
        parser.add_argument('--lambda_distill', type=float, default=1,
                            help='weights for the intermediate activation distillation loss')
        parser.add_argument('--lambda_recon', type=float, default=100,
                            help='weights for the reconstruction loss.')
        parser.add_argument('--lambda_gan', type=float, default=1,
                            help='weight for gan loss')
        parser.add_argument('--teacher_dropout_rate', type=float, default=0)
        parser.add_argument('--student_dropout_rate', type=float, default=0)
        return parser

    def __init__(self, opt):
        assert opt.isTrain
        super(BaseResnetDistiller, self).__init__(opt)
        self.loss_names = ['G_gan', 'G_distill', 'G_recon', 'D_fake', 'D_real']
        self.optimizers = []
        self.image_paths = []
        self.visual_names = ['real_A', 'Sfake_B', 'Tfake_B', 'real_B']
        self.model_names = ['netG_student', 'netG_teacher', 'netD']
        self.netG_teacher = networks.define_G(opt.input_nc, opt.output_nc, opt.teacher_ngf,
                                              opt.teacher_netG, opt.norm, opt.teacher_dropout_rate,
                                              opt.init_type, opt.init_gain, self.gpu_ids, opt=opt)
        self.netG_student = networks.define_G(opt.input_nc, opt.output_nc, opt.student_ngf,
                                              opt.student_netG, opt.norm, opt.student_dropout_rate,
                                              opt.init_type, opt.init_gain, self.gpu_ids, opt=opt)

        if hasattr(opt, 'distiller'):
            self.netG_pretrained = networks.define_G(opt.input_nc, opt.output_nc, opt.pretrained_ngf,
                                                     opt.pretrained_netG, opt.norm, 0,
                                                     opt.init_type, opt.init_gain, self.gpu_ids, opt=opt)

        if opt.dataset_mode == 'aligned':
            self.netD = networks.define_D(opt.input_nc + opt.output_nc, opt.ndf, opt.netD,
                                          opt.n_layers_D, opt.norm, opt.init_type, opt.init_gain, self.gpu_ids)
        elif opt.dataset_mode == 'unaligned':
            self.netD = networks.define_D(opt.output_nc, opt.ndf, opt.netD,
                                          opt.n_layers_D, opt.norm, opt.init_type, opt.init_gain, self.gpu_ids)
        else:
            raise NotImplementedError('Unknown dataset mode [%s]!!!' % opt.dataset_mode)

        self.netG_teacher.eval()
        self.criterionGAN = models.modules.loss.GANLoss(opt.gan_mode).to(self.device)
        if opt.recon_loss_type == 'l1':
            self.criterionRecon = torch.nn.L1Loss()
        elif opt.recon_loss_type == 'l2':
            self.criterionRecon = torch.nn.MSELoss()
        elif opt.recon_loss_type == 'smooth_l1':
            self.criterionRecon = torch.nn.SmoothL1Loss()
        elif opt.recon_loss_type == 'vgg':
            self.criterionRecon = models.modules.loss.VGGLoss(self.device)
        else:
            raise NotImplementedError('Unknown reconstruction loss type [%s]!' % opt.loss_type)

        self.mapping_layers = ['module.model.%d' % i for i in range(9, 21, 3)]

        self.netAs = []
        self.Tacts, self.Sacts = {}, {}

        G_params = [self.netG_student.parameters()]
        for i, n in enumerate(self.mapping_layers):
            ft, fs = self.opt.teacher_ngf, self.opt.student_ngf
            if hasattr(opt, 'distiller'):
                netA = nn.Conv2d(in_channels=fs * 4, out_channels=ft * 4, kernel_size=1). \
                    to(self.device)
            else:
                netA = SuperConv2d(in_channels=fs * 4, out_channels=ft * 4, kernel_size=1). \
                    to(self.device)
            networks.init_net(netA)
            G_params.append(netA.parameters())
            self.netAs.append(netA)
            self.loss_names.append('G_distill%d' % i)

        self.optimizer_G = torch.optim.Adam(itertools.chain(*G_params), lr=opt.lr, betas=(opt.beta1, 0.999))
        self.optimizer_D = torch.optim.Adam(self.netD.parameters(), lr=opt.lr, betas=(opt.beta1, 0.999))
        self.optimizers.append(self.optimizer_G)
        self.optimizers.append(self.optimizer_D)

        self.eval_dataloader = create_eval_dataloader(self.opt, direction=opt.direction)

        block_idx = InceptionV3.BLOCK_INDEX_BY_DIM[2048]
        self.inception_model = InceptionV3([block_idx])
        self.inception_model.to(self.device)
        self.inception_model.eval()

        if 'cityscapes' in opt.dataroot:
            self.drn_model = DRNSeg('drn_d_105', 19, pretrained=False)
            util.load_network(self.drn_model, opt.drn_path, verbose=False)
            if len(opt.gpu_ids) > 0:
                self.drn_model.to(self.device)
                self.drn_model = nn.DataParallel(self.drn_model, opt.gpu_ids)
            self.drn_model.eval()

        self.npz = np.load(opt.real_stat_path)
        self.is_best = False

    def setup(self, opt, verbose=True):
        self.schedulers = [networks.get_scheduler(optimizer, opt) for optimizer in self.optimizers]
        self.load_networks(verbose)
        if verbose:
            self.print_networks()
        if self.opt.lambda_distill > 0:
            def get_activation(mem, name):
                def get_output_hook(module, input, output):
                    mem[name] = output

                return get_output_hook

            def add_hook(net, mem, mapping_layers):
                for n, m in net.named_modules():
                    if n in mapping_layers:
                        m.register_forward_hook(get_activation(mem, n))

            add_hook(self.netG_teacher, self.Tacts, self.mapping_layers)
            add_hook(self.netG_student, self.Sacts, self.mapping_layers)

    def set_input(self, input):
        AtoB = self.opt.direction == 'AtoB'
        self.real_A = input['A' if AtoB else 'B'].to(self.device)
        self.real_B = input['B' if AtoB else 'A'].to(self.device)
        self.image_paths = input['A_paths' if AtoB else 'B_paths']

    def set_single_input(self, input):
        self.real_A = input['A'].to(self.device)
        self.image_paths = input['A_paths']

    def forward(self):
        raise NotImplementedError

    def backward_D(self):
        if self.opt.dataset_mode == 'aligned':
            fake = torch.cat((self.real_A, self.Sfake_B), 1).detach()
            real = torch.cat((self.real_A, self.real_B), 1).detach()
        else:
            fake = self.Sfake_B.detach()
            real = self.real_B.detach()

        pred_fake = self.netD(fake)
        self.loss_D_fake = self.criterionGAN(pred_fake, False, for_discriminator=True)

        pred_real = self.netD(real)
        self.loss_D_real = self.criterionGAN(pred_real, True, for_discriminator=True)

        self.loss_D = (self.loss_D_fake + self.loss_D_real) * 0.5
        self.loss_D.backward()

    def calc_distill_loss(self):
        raise NotImplementedError

    def backward_G(self):
        raise NotImplementedError

    def optimize_parameters(self, steps):
        raise NotImplementedError

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
                for param_group in optimizer.param_groups:
                    param_group['lr'] = self.opt.lr

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

        for i, net in enumerate(self.netAs):
            save_filename = '%s_net_%s-%d.pth' % (epoch, 'A', i)
            save_path = os.path.join(self.save_dir, save_filename)
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
