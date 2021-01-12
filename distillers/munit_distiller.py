import ntpath
import os
from argparse import ArgumentParser

import torch
from tqdm import tqdm

from distillers.base_munit_distiller import BaseMunitDistiller
from metric import get_fid
from utils import util


class MunitDistiller(BaseMunitDistiller):
    @staticmethod
    def modify_commandline_options(parser, is_train):
        assert is_train
        parser = super(MunitDistiller, MunitDistiller).modify_commandline_options(parser, is_train)
        assert isinstance(parser, ArgumentParser)
        return parser

    def __init__(self, opt):
        assert opt.isTrain
        super(MunitDistiller, self).__init__(opt)
        self.best_fid = 1e9
        self.fids = []

    def forward(self, need_style_encoder=False):
        opt = self.opt
        batch_size = self.real_A.size(0)
        style_dim = opt.style_dim
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
        return self.netA(c)

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

    def evaluate_model(self, step):
        self.is_best = False
        save_dir = os.path.join(self.opt.log_dir, 'eval', str(step))
        os.makedirs(save_dir, exist_ok=True)
        self.netG_student.eval()
        fakes, names = [], []
        cnt = 0
        for i, data_i in enumerate(tqdm(self.eval_dataloader, desc='Eval       ', position=2, leave=False)):
            if self.opt.dataset_mode == 'aligned':
                self.set_input(data_i)
            else:
                self.set_single_input(data_i)
            self.test()
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
        if fid < self.best_fid:
            self.is_best = True
            self.best_fid = fid
        self.fids.append(fid)
        if len(self.fids) > 3:
            self.fids.pop(0)
        ret = {'metric/fid': fid, 'metric/fid-mean': sum(self.fids) / len(self.fids), 'metric/fid-best': self.best_fid}
        self.netG_student.train()
        return ret

    def test(self):
        with torch.no_grad():
            self.forward()
