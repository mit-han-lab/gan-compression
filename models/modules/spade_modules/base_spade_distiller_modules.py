import copy
import os

import torch
from torch import nn

from models import networks
from models.modules.loss import GANLoss, VGGLoss
from models.modules.spade_modules.spade_model_modules import SPADEModelModules
from models.modules.super_modules import SuperConv2d
from utils import util


class BaseSPADEDistillerModules(SPADEModelModules):
    def __init__(self, opt):
        assert opt.isTrain
        opt = copy.deepcopy(opt)
        if len(opt.gpu_ids) > 0:
            opt.gpu_ids = opt.gpu_ids[:1]
        self.gpu_ids = opt.gpu_ids
        super(SPADEModelModules, self).__init__()
        self.opt = opt
        self.model_names = ['G_student', 'G_teacher', 'D']
        teacher_opt = copy.deepcopy(opt)
        teacher_opt.norm_G = opt.teacher_norm_G
        teacher_opt.ngf = opt.teacher_ngf
        self.netG_teacher = networks.define_G(opt.input_nc, opt.output_nc, opt.teacher_ngf,
                                              opt.teacher_netG, opt.norm, 0,
                                              opt.init_type, opt.init_gain, self.gpu_ids, opt=teacher_opt)
        student_opt = copy.deepcopy(opt)
        student_opt.norm_G = opt.student_norm_G
        student_opt.ngf = opt.student_ngf
        self.netG_student = networks.define_G(opt.input_nc, opt.output_nc, opt.student_ngf,
                                              opt.student_netG, opt.norm, 0,
                                              opt.init_type, opt.init_gain, self.gpu_ids, opt=student_opt)
        if hasattr(opt, 'distiller'):
            pretrained_opt = copy.deepcopy(opt)
            pretrained_opt.norm_G = opt.pretrained_norm_G
            pretrained_opt.ngf = opt.pretrained_ngf
            self.netG_pretrained = networks.define_G(opt.input_nc, opt.output_nc, opt.pretrained_ngf,
                                                     opt.pretrained_netG, opt.norm, 0,
                                                     opt.init_type, opt.init_gain, self.gpu_ids, opt=pretrained_opt)
        self.netD = networks.define_D(opt.input_nc + opt.output_nc, opt.ndf,
                                      opt.netD, opt.n_layers_D, opt.norm,
                                      opt.init_type, opt.init_gain, self.gpu_ids, opt=opt)
        self.mapping_layers = ['head_0', 'G_middle_1', 'up_1']
        self.netAs = nn.ModuleList()
        for i, mapping_layer in enumerate(self.mapping_layers):
            if mapping_layer != 'up_1':
                fs, ft = opt.student_ngf * 16, opt.teacher_ngf * 16
            else:
                fs, ft = opt.student_ngf * 4, opt.teacher_ngf * 4
            if hasattr(opt, 'distiller'):
                netA = nn.Conv2d(in_channels=fs, out_channels=ft, kernel_size=1)
            else:
                netA = SuperConv2d(in_channels=fs, out_channels=ft, kernel_size=1)
            networks.init_net(netA, opt.init_type, opt.init_gain, self.gpu_ids)
            self.netAs.append(netA)
        self.criterionGAN = GANLoss(opt.gan_mode)
        self.criterionFeat = nn.L1Loss()
        self.criterionVGG = VGGLoss()
        self.optimizers = []
        self.netG_teacher.eval()
        self.config = None

    def create_optimizers(self):
        if self.opt.no_TTUR:
            beta1, beta2 = self.opt.beta1, self.opt.beta2
            G_lr, D_lr = self.opt.lr, self.opt.lr
        else:
            beta1, beta2 = 0, 0.9
            G_lr, D_lr = self.opt.lr / 2, self.opt.lr * 2
        G_params = list(self.netG_student.parameters())
        for netA in self.netAs:
            G_params += list(netA.parameters())
        optimizer_G = torch.optim.Adam(G_params, lr=G_lr, betas=(beta1, beta2))
        optimizer_D = torch.optim.Adam(list(self.netD.parameters()), lr=D_lr, betas=(beta1, beta2))
        return optimizer_G, optimizer_D

    def forward(self, input_semantics, real_B=None, mode='generate_fake'):
        if self.config is not None:
            self.netG_student.config = self.config
        if mode == 'generate_fake':
            with torch.no_grad():
                Tfake_B = self.netG_teacher(input_semantics)
                Sfake_B = self.netG_student(input_semantics)
            return Tfake_B, Sfake_B
        elif mode == 'G_loss':
            assert real_B is not None
            return self.compute_G_loss(input_semantics, real_B)
        elif mode == 'D_loss':
            assert real_B is not None
            return self.compute_D_loss(input_semantics, real_B)
        elif mode == 'calibrate':
            with torch.no_grad():
                self.netG_student(input_semantics)
            return
        else:
            raise NotImplementedError('Unknown forward mode [%s]!!!' % mode)

    def profile(self, input_semantics, config=None):
        raise NotImplementedError('The distiller is only for training!!!')

    def calc_distill_loss(self, Tacts, Sacts):
        raise NotImplementedError

    def compute_G_loss(self, input_semantics, real_B):
        with torch.no_grad():
            Tfake_B, Tacts = self.netG_teacher(input_semantics, mapping_layers=self.mapping_layers)
        Sfake_B, Sacts = self.netG_student(input_semantics, mapping_layers=self.mapping_layers)
        loss_G_distill, losses = self.calc_distill_loss(Tacts, Sacts)
        pred_fake, pred_real = self.discriminate(input_semantics, Sfake_B, real_B)
        loss_G_gan = self.criterionGAN(pred_fake, True, for_discriminator=False) * self.opt.lambda_gan
        num_D = len(pred_fake)
        loss_G_feat = 0
        for i in range(num_D):
            num_intermediate_outputs = len(pred_fake[i]) - 1
            for j in range(num_intermediate_outputs):  # for each layer output
                unweighted_loss = self.criterionFeat(
                    pred_fake[i][j], pred_real[i][j].detach())
                loss_G_feat += unweighted_loss * self.opt.lambda_feat / num_D
        loss_G_vgg = self.criterionVGG(Sfake_B, real_B) * self.opt.lambda_vgg
        loss_G = loss_G_gan + loss_G_distill + loss_G_feat + loss_G_vgg
        losses.update({'loss_G': loss_G, 'G_gan': loss_G_gan,
                       'G_distill': loss_G_distill,
                       'G_feat': loss_G_feat, 'G_vgg': loss_G_vgg})
        return losses

    def compute_D_loss(self, input_semantics, real_B):
        with torch.no_grad():
            fake_B = self.netG_student(input_semantics)
        pred_fake, pred_real = self.discriminate(input_semantics, fake_B, real_B)
        loss_D_fake = self.criterionGAN(pred_fake, False, for_discriminator=True)
        loss_D_real = self.criterionGAN(pred_real, True, for_discriminator=True)
        loss_D = loss_D_fake + loss_D_real
        losses = {'loss_D': loss_D, 'D_fake': loss_D_fake, 'D_real': loss_D_real}
        return losses

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

    def save_networks(self, epoch, save_dir):
        def save_net(net, save_path):
            if len(self.gpu_ids) > 0 and torch.cuda.is_available():
                torch.save(net.cpu().state_dict(), save_path)
                net.cuda(self.gpu_ids[0])
            else:
                torch.save(net.cpu().state_dict(), save_path)

        save_filename = '%s_net_%s.pth' % (epoch, 'G')
        save_path = os.path.join(save_dir, save_filename)
        net = getattr(self, 'net%s_student' % 'G')
        save_net(net, save_path)

        save_filename = '%s_net_%s.pth' % (epoch, 'D')
        save_path = os.path.join(save_dir, save_filename)
        net = getattr(self, 'net%s' % 'D')
        save_net(net, save_path)

        for i, net in enumerate(self.netAs):
            save_filename = '%s_net_%s-%d.pth' % (epoch, 'A', i)
            save_path = os.path.join(save_dir, save_filename)
            save_net(net, save_path)
