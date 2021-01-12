import ntpath
import os

import torch
from torch import nn
from tqdm import tqdm

from configs import decode_config
from configs.munit_configs import get_configs
from configs.single_configs import SingleConfigs
from distillers.base_munit_distiller import BaseMunitDistiller
from metric import get_fid
from utils import util
from argparse import ArgumentParser


class MunitSupernet(BaseMunitDistiller):
    @staticmethod
    def modify_commandline_options(parser, is_train):
        assert is_train
        parser = super(MunitSupernet, MunitSupernet).modify_commandline_options(parser, is_train)
        assert isinstance(parser, ArgumentParser)
        parser.set_defaults(student_netG='super_munit', student_ngf=64)
        return parser

    def __init__(self, opt):
        assert 'super' in opt.student_netG
        super(MunitSupernet, self).__init__(opt)
        self.best_fid_largest = 1e9
        self.best_fid_smallest = 1e9
        self.fids_largest, self.fids_smallest = [], []
        if opt.config_set is not None:
            assert opt.config_str is None
            self.configs = get_configs(opt.config_set)
            self.opt.eval_mode = 'both'
        else:
            assert opt.config_str is not None
            self.configs = SingleConfigs(decode_config(opt.config_str))
            self.opt.eval_mode = 'largest'

    def forward(self, config, need_style_encoder=False):
        opt = self.opt
        batch_size = self.real_A.size(0)
        style_dim = opt.style_dim

        if isinstance(self.netG_student, nn.DataParallel):
            self.netG_student.module.configs = config
        else:
            self.netG_student.configs = config

        self.netG_student(self.real_B)

        if need_style_encoder:
            with torch.no_grad():
                _, self.s_teacher = self.netG_teacher.encode(self.real_B, need_content=False)
            _, self.s_student = self.netG_student.encode(self.real_B, need_content=False)
        else:
            self.s_teacher = torch.randn(batch_size, style_dim, 1, 1, device=self.device)
            self.s_student = self.s_teacher
        with torch.no_grad():
            self.c_teacher, _ = self.netG_teacher.encode(self.real_A, need_style=False)
            self.Tfake_B = self.netG_teacher.decode(self.c_teacher, self.s_teacher)
        self.c_student, _ = self.netG_student.encode(self.real_A, need_style=False)
        self.Sfake_B = self.netG_student.decode(self.c_student, self.s_student)

    def forward_A(self, c):
        return self.netA(c, {'channel': self.netA.out_channels})

    def optimize_parameters(self, steps):
        need_style_encoder = False if self.opt.student_no_style_encoder \
            else steps % self.opt.style_encoder_step != 0
        self.optimizer_D.zero_grad()
        self.optimizer_G.zero_grad()
        config = self.configs.sample()
        self.forward(config=config,need_style_encoder=need_style_encoder)
        util.set_requires_grad(self.netD, True)
        self.backward_D()
        util.set_requires_grad(self.netD, False)
        self.backward_G(need_style_encoder=need_style_encoder)
        self.optimizer_D.step()
        self.optimizer_G.step()

    def evaluate_model(self, step):
        ret = {}
        self.is_best = False
        save_dir = os.path.join(self.opt.log_dir, 'eval', str(step))
        os.makedirs(save_dir, exist_ok=True)
        self.netG_student.eval()
        if self.opt.eval_mode == 'both':
            settings = ('largest', 'smallest')
        else:
            settings = (self.opt.eval_mode,)
        for config_name in settings:
            config = self.configs(config_name)
            fakes, names = [], []
            cnt = 0
            for i, data_i in enumerate(tqdm(self.eval_dataloader, desc='Eval       ', position=2, leave=False)):
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
                    if cnt < 10:
                        input_im = util.tensor2im(self.real_A[j])
                        Sfake_im = util.tensor2im(self.Sfake_B[j])
                        Tfake_im = util.tensor2im(self.Tfake_B[j])
                        util.save_image(input_im, os.path.join(save_dir, 'input', '%s.png') % name, create_dir=True)
                        util.save_image(Sfake_im, os.path.join(save_dir, 'Sfake', '%s.png' % name), create_dir=True)
                        util.save_image(Tfake_im, os.path.join(save_dir, 'Tfake', '%s.png' % name), create_dir=True)
                        if self.opt.dataset_mode == 'aligned':
                            real_im = util.tensor2im(self.real_B[j])
                            util.save_image(real_im, os.path.join(save_dir, 'real', '%s.png' % name), create_dir=True)
                    cnt += 1
            fid = get_fid(fakes, self.inception_model, self.npz, device=self.device,
                          batch_size=self.opt.eval_batch_size, tqdm_position=2)
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

        self.netG_student.train()
        return ret

    def test(self, config):
        with torch.no_grad():
            self.forward(config)

    def load_networks(self, verbose=True):
        super(MunitSupernet, self).load_networks()
