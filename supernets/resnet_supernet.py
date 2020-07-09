import ntpath
import os

import torch
import torch.nn.functional as F
from torch import nn
from tqdm import tqdm

from configs import decode_config
from configs.resnet_configs import get_configs
from configs.single_configs import SingleConfigs
from distillers.base_resnet_distiller import BaseResnetDistiller
from metric import get_fid, get_mIoU
from models.modules.super_modules import SuperConv2d
from utils import util


class ResnetSupernet(BaseResnetDistiller):
    @staticmethod
    def modify_commandline_options(parser, is_train):
        assert is_train
        parser = super(ResnetSupernet, ResnetSupernet).modify_commandline_options(parser, is_train)
        parser.set_defaults(norm='instance', student_netG='super_mobile_resnet_9blocks',
                            dataset_mode='aligned', log_dir='logs/supernet')
        return parser

    def __init__(self, opt):
        assert opt.isTrain
        assert 'super' in opt.student_netG
        super(ResnetSupernet, self).__init__(opt)
        self.best_fid_largest = 1e9
        self.best_fid_smallest = 1e9
        self.best_mIoU_largest = -1e9
        self.best_mIoU_smallest = -1e9
        self.fids_largest, self.fids_smallest = [], []
        self.mIoUs_largest, self.mIoUs_smallest = [], []
        if opt.config_set is not None:
            assert opt.config_str is None
            self.configs = get_configs(opt.config_set)
            self.opt.eval_mode = 'both'
        else:
            assert opt.config_str is not None
            self.configs = SingleConfigs(decode_config(opt.config_str))
            self.opt.eval_mode = 'largest'

    def forward(self, config):
        with torch.no_grad():
            self.Tfake_B = self.netG_teacher(self.real_A)
        if isinstance(self.netG_student, nn.DataParallel):
            self.netG_student.module.configs = config
        else:
            self.netG_student.configs = config
        self.Sfake_B = self.netG_student(self.real_A)

    def calc_distill_loss(self):
        losses = []
        for i, netA in enumerate(self.netAs):
            assert isinstance(netA, SuperConv2d)
            n = self.mapping_layers[i]
            Tact = self.Tacts[n]
            Sact = self.Sacts[n]
            Sact = netA(Sact, {'channel': netA.out_channels})
            loss = F.mse_loss(Sact, Tact)
            setattr(self, 'loss_G_distill%d' % i, loss)
            losses.append(loss)
        return sum(losses)

    def backward_G(self):
        if self.opt.dataset_mode == 'aligned':
            self.loss_G_recon = self.criterionRecon(self.Sfake_B, self.real_B) * self.opt.lambda_recon
            fake = torch.cat((self.real_A, self.Sfake_B), 1)
        else:
            self.loss_G_recon = self.criterionRecon(self.Sfake_B, self.Tfake_B) * self.opt.lambda_recon
            fake = self.Sfake_B
        pred_fake = self.netD(fake)
        self.loss_G_gan = self.criterionGAN(pred_fake, True, for_discriminator=False) * self.opt.lambda_gan
        if self.opt.lambda_distill > 0:
            self.loss_G_distill = self.calc_distill_loss() * self.opt.lambda_distill
        else:
            self.loss_G_distill = 0
        self.loss_G = self.loss_G_gan + self.loss_G_recon + self.loss_G_distill
        self.loss_G.backward()

    def optimize_parameters(self, steps):
        self.optimizer_D.zero_grad()
        self.optimizer_G.zero_grad()
        config = self.configs.sample()
        self.forward(config=config)
        util.set_requires_grad(self.netD, True)
        self.backward_D()
        util.set_requires_grad(self.netD, False)
        self.backward_G()
        self.optimizer_D.step()
        self.optimizer_G.step()

    def evaluate_model(self, step):
        ret = {}
        self.is_best = False
        save_dir = os.path.join(self.opt.log_dir, 'eval', str(step))
        os.makedirs(save_dir, exist_ok=True)
        self.netG_student.eval()
        if self.opt.eval_mode == 'both':
            setting = ('largest', 'smallest')
        else:
            setting = (self.opt.eval_mode,)
        for config_name in setting:
            config = self.configs(config_name)
            fakes, names = [], []
            cnt = 0
            for i, data_i in enumerate(tqdm(self.eval_dataloader)):
                if self.opt.dataset_mode == 'aligned':
                    self.set_input(data_i)
                else:
                    self.set_single_input(data_i)
                self.test(config)
                fakes.append(self.Sfake_B.cpu())
                for j in range(len(self.image_paths)):
                    short_path = ntpath.basename(self.image_paths[j])
                    name = os.path.splitext(short_path)[0]
                    names.append(name)
                    if i < 10:
                        Sfake_im = util.tensor2im(self.Sfake_B[j])
                        real_im = util.tensor2im(self.real_B[j])
                        Tfake_im = util.tensor2im(self.Tfake_B[j])
                        util.save_image(real_im, os.path.join(save_dir, 'real', '%s.png' % name), create_dir=True)
                        util.save_image(Sfake_im, os.path.join(save_dir, 'Sfake_%s' % config_name, '%s.png' % name),
                                        create_dir=True)
                        util.save_image(Tfake_im, os.path.join(save_dir, 'Tfake', '%s.png' % name), create_dir=True)
                        if self.opt.dataset_mode == 'aligned':
                            input_im = util.tensor2im(self.real_A[j])
                            util.save_image(input_im, os.path.join(save_dir, 'input', '%s.png') % name, create_dir=True)
                    cnt += 1

            fid = get_fid(fakes, self.inception_model, self.npz,
                          device=self.device, batch_size=self.opt.eval_batch_size)
            if fid < getattr(self, 'best_fid_%s' % config_name):
                self.is_best = True
                setattr(self, 'best_fid_%s' % config_name, fid)
            fids = getattr(self, 'fids_%s' % config_name)
            fids.append(fid)
            if len(fids) > 3:
                fids.pop(0)

            ret['metric/fid_%s' % config_name] = fid
            ret['metric/fid_%s-mean' % config_name] = sum(getattr(self, 'fids_%s' % config_name)) / len(
                getattr(self, 'fids_%s' % config_name))
            ret['metric/fid_%s-best' % config_name] = getattr(self, 'best_fid_%s' % config_name)

            if 'cityscapes' in self.opt.dataroot:
                mIoU = get_mIoU(fakes, names, self.drn_model, self.device,
                                table_path=self.opt.table_path,
                                data_dir=self.opt.cityscapes_path,
                                batch_size=self.opt.eval_batch_size,
                                num_workers=self.opt.num_threads)
                if mIoU > getattr(self, 'best_mIoU_%s' % config_name):
                    self.is_best = True
                    setattr(self, 'best_mIoU_%s' % config_name, mIoU)
                mIoUs = getattr(self, 'mIoUs_%s' % config_name)
                mIoUs.append(mIoU)
                if len(mIoUs) > 3:
                    mIoUs.pop(0)
                ret['metric/mIoU_%s' % config_name] = mIoU
                ret['metric/mIoU_%s-mean' % config_name] = sum(getattr(self, 'mIoUs_%s' % config_name)) / len(
                    getattr(self, 'mIoUs_%s' % config_name))
                ret['metric/mIoU_%s-best' % config_name] = getattr(self, 'best_mIoU_%s' % config_name)

        self.netG_student.train()
        return ret

    def test(self, config):
        with torch.no_grad():
            self.forward(config)
