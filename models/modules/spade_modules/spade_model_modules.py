import copy
import os

import torch
from torch import nn
from torchprofile import profile_macs

from models import networks
from models.modules.loss import GANLoss, VGGLoss
from utils import util


class SPADEModelModules(nn.Module):
    def __init__(self, opt):
        opt = copy.deepcopy(opt)
        if len(opt.gpu_ids) > 0:
            opt.gpu_ids = opt.gpu_ids[:1]
        self.gpu_ids = opt.gpu_ids
        super(SPADEModelModules, self).__init__()
        self.opt = opt
        self.model_names = ['G']
        self.visual_names = ['labels', 'fake_B', 'real_B']
        self.netG = networks.define_G(opt.input_nc, opt.output_nc, opt.ngf, opt.netG,
                                      opt.norm, opt.dropout_rate, opt.init_type,
                                      opt.init_gain, self.gpu_ids, opt=opt)
        if opt.isTrain:
            self.model_names.append('D')
            self.netD = networks.define_D(opt.input_nc + opt.output_nc, opt.ndf,
                                          opt.netD, opt.n_layers_D, opt.norm,
                                          opt.init_type, opt.init_gain, self.gpu_ids, opt=opt)
            self.criterionGAN = GANLoss(opt.gan_mode)
            self.criterionFeat = nn.L1Loss()
            self.criterionVGG = VGGLoss()
            self.optimizers = []
            self.loss_names = ['G_gan', 'G_feat', 'G_vgg', 'D_real', 'D_fake']
        else:
            self.netG.eval()
        self.config = None

    def create_optimizers(self):
        if self.opt.no_TTUR:
            beta1, beta2 = self.opt.beta1, self.opt.beta2
            G_lr, D_lr = self.opt.lr, self.opt.lr
        else:
            beta1, beta2 = 0, 0.9
            G_lr, D_lr = self.opt.lr / 2, self.opt.lr * 2
        optimizer_G = torch.optim.Adam(list(self.netG.parameters()), lr=G_lr, betas=(beta1, beta2))
        optimizer_D = torch.optim.Adam(list(self.netD.parameters()), lr=D_lr, betas=(beta1, beta2))
        return optimizer_G, optimizer_D

    def forward(self, input_semantics, real_B=None, mode='generate_fake'):
        if self.config is not None:
            self.netG.config = self.config
        if mode == 'generate_fake':
            fake_B = self.netG(input_semantics)
            return fake_B
        elif mode == 'G_loss':
            assert real_B is not None
            return self.compute_G_loss(input_semantics, real_B)
        elif mode == 'D_loss':
            assert real_B is not None
            return self.compute_D_loss(input_semantics, real_B)
        elif mode == 'calibrate':
            with torch.no_grad():
                self.netG(input_semantics)
        else:
            raise NotImplementedError('Unknown forward mode [%s]!!!' % mode)

    def profile(self, input_semantics):
        netG = self.netG
        if isinstance(netG, nn.DataParallel):
            netG = netG.module
        if self.config is not None:
            netG.config = self.config
        with torch.no_grad():
            macs = profile_macs(netG, (input_semantics,))
        params = 0
        for p in netG.parameters():
            params += p.numel()
        return macs, params

    def compute_G_loss(self, input_semantics, real_B):
        fake_B = self.netG(input_semantics)
        pred_fake, pred_real = self.discriminate(input_semantics, fake_B, real_B)
        loss_G_gan = self.criterionGAN(pred_fake, True, for_discriminator=False) * self.opt.lambda_gan
        num_D = len(pred_fake)
        loss_G_feat = 0
        for i in range(num_D):
            num_intermediate_outputs = len(pred_fake[i]) - 1
            for j in range(num_intermediate_outputs):  # for each layer output
                unweighted_loss = self.criterionFeat(
                    pred_fake[i][j], pred_real[i][j].detach())
                loss_G_feat += unweighted_loss * self.opt.lambda_feat / num_D
        loss_G_vgg = self.criterionVGG(fake_B, real_B) * self.opt.lambda_vgg
        loss_G = loss_G_gan + loss_G_feat + loss_G_vgg
        losses = {'loss_G': loss_G, 'G_gan': loss_G_gan,
                  'G_feat': loss_G_feat, 'G_vgg': loss_G_vgg}
        return losses

    def compute_D_loss(self, input_semantics, real_B):
        with torch.no_grad():
            fake_B = self.netG(input_semantics)
        pred_fake, pred_real = self.discriminate(input_semantics, fake_B, real_B)
        loss_D_fake = self.criterionGAN(pred_fake, False, for_discriminator=True)
        loss_D_real = self.criterionGAN(pred_real, True, for_discriminator=True)
        loss_D = loss_D_fake + loss_D_real
        losses = {'loss_D': loss_D, 'D_fake': loss_D_fake, 'D_real': loss_D_real}
        return losses

    def discriminate(self, input_semantics, fake_B, real_B):
        fake_concat = torch.cat([input_semantics, fake_B], dim=1)
        real_concat = torch.cat([input_semantics, real_B], dim=1)
        fake_and_real = torch.cat([fake_concat, real_concat], dim=0)
        discriminator_out = self.netD(fake_and_real)
        pred_fake, pred_real = self.divide_pred(discriminator_out)
        return pred_fake, pred_real

    # Take the prediction of fake and real images from the combined batch
    def divide_pred(self, pred):
        # the prediction contains the intermediate outputs of multiscale GAN,
        # so it's usually a list
        if type(pred) == list:
            fake = []
            real = []
            for p in pred:
                fake.append([tensor[:tensor.size(0) // 2] for tensor in p])
                real.append([tensor[tensor.size(0) // 2:] for tensor in p])
        else:
            fake = pred[:pred.size(0) // 2]
            real = pred[pred.size(0) // 2:]

        return fake, real

    def load_networks(self, verbose=True):
        for name in self.model_names:
            net = getattr(self, 'net' + name, None)
            path = getattr(self.opt, 'restore_%s_path' % name, None)
            if path is not None:
                util.load_network(net, path, verbose)

    def save_networks(self, epoch, save_dir):
        for name in self.model_names:
            if isinstance(name, str):
                save_filename = '%s_net_%s.pth' % (epoch, name)
                save_path = os.path.join(save_dir, save_filename)
                net = getattr(self, 'net' + name)
                if len(self.gpu_ids) > 0 and torch.cuda.is_available():
                    torch.save(net.cpu().state_dict(), save_path)
                    net.cuda(self.gpu_ids[0])
                else:
                    torch.save(net.cpu().state_dict(), save_path)
