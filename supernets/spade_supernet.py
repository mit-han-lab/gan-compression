import argparse
import copy
import ntpath
import os

import torch
from tqdm import tqdm

from configs import decode_config
from configs.single_configs import SingleConfigs
from configs.spade_configs import get_configs
from data import create_dataloader
from distillers.base_spade_distiller import BaseSPADEDistiller
from metric import get_fid, get_cityscapes_mIoU
from models import networks
from utils import util


class SPADESupernet(BaseSPADEDistiller):
    @staticmethod
    def modify_commandline_options(parser, is_train):
        parser = super(SPADESupernet, SPADESupernet).modify_commandline_options(parser, is_train)
        assert isinstance(parser, argparse.ArgumentParser)
        parser.set_defaults(netD='multi_scale', dataset_mode='cityscapes', batch_size=16,
                            print_freq=50, save_latest_freq=10000000000, save_epoch_freq=10,
                            nepochs=100, nepochs_decay=100, init_type='xavier',
                            teacher_ngf=64, student_ngf=48,
                            student_netG='super_mobile_spade', ndf=64)
        parser = networks.modify_commandline_options(parser, is_train)
        return parser

    def __init__(self, opt):
        super(SPADESupernet, self).__init__(opt)
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
            self.opt.no_calibration = False
        else:
            assert opt.config_str is not None
            self.configs = SingleConfigs(decode_config(opt.config_str))
            self.opt.eval_mode = 'largest'
            self.opt.no_calibration = True
        opt = copy.deepcopy(opt)
        opt.load_in_memory = False
        opt.max_dataset_size = 256
        self.train_dataloader = create_dataloader(opt, verbose=False)

    def optimize_parameters(self, steps):
        config = self.configs.sample()
        self.modules_on_one_gpu.config = config
        super(SPADESupernet, self).optimize_parameters(steps)

    def evaluate_model(self, step):
        ret = {}
        self.is_best = False
        save_dir = os.path.join(self.opt.log_dir, 'eval', str(step))
        os.makedirs(save_dir, exist_ok=True)
        if self.opt.eval_mode == 'both':
            settings = ('largest', 'smallest')
        else:
            settings = (self.opt.eval_mode,)
        for config_name in settings:
            config = self.configs(config_name)
            fakes, names = [], []
            self.modules_on_one_gpu.netG_student.train()
            self.calibrate(config, 2)
            tqdm_position = 2 + int(not self.opt.no_calibration)
            self.modules_on_one_gpu.netG_student.eval()
            torch.cuda.empty_cache()

            cnt = 0
            for i, data_i in enumerate(tqdm(self.eval_dataloader, desc='Eval       ', position=tqdm_position, leave=False)):
                self.set_input(data_i)
                self.test(config=config)
                fakes.append(self.Sfake_B.cpu())
                for j in range(len(self.image_paths)):
                    short_path = ntpath.basename(self.image_paths[j])
                    name = os.path.splitext(short_path)[0]
                    names.append(name)
                    if cnt < 10:
                        input_im = util.tensor2label(self.input_semantics[j], self.opt.input_nc + 2)
                        real_im = util.tensor2im(self.real_B[j])
                        Tfake_im = util.tensor2im(self.Tfake_B[j])
                        Sfake_im = util.tensor2im(self.Sfake_B[j])
                        util.save_image(input_im, os.path.join(save_dir, 'input', '%s.png' % name), create_dir=True)
                        util.save_image(real_im, os.path.join(save_dir, 'real', '%s.png' % name), create_dir=True)
                        util.save_image(Tfake_im, os.path.join(save_dir, 'Tfake', '%s.png' % name), create_dir=True)
                        util.save_image(Sfake_im, os.path.join(save_dir, 'Sfake', '%s.png' % name), create_dir=True)
                    cnt += 1

            if not self.opt.no_fid:
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
            if 'cityscapes' in self.opt.dataroot and not self.opt.no_mIoU:
                mIoU = get_cityscapes_mIoU(fakes, names, self.drn_model, self.device,
                                           table_path=self.opt.table_path,
                                           data_dir=self.opt.cityscapes_path,
                                           batch_size=self.opt.eval_batch_size,
                                           num_workers=self.opt.num_threads, tqdm_position=2)
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
            self.modules_on_one_gpu.netG_student.train()

        torch.cuda.empty_cache()
        return ret

    def calibrate(self, config, tqdm_position=None):
        if self.opt.no_calibration:
            return
        if tqdm_position is None or tqdm_position >= 0:
            calibrate_tqdm = tqdm(self.train_dataloader, desc='Calibrate  ', position=tqdm_position, leave=False)
        else:
            calibrate_tqdm = self.train_dataloader
        config = copy.deepcopy(config)
        for i, data in enumerate(calibrate_tqdm):
            self.set_input(data)
            if i == 0:
                config['calibrate_bn'] = True
            self.modules_on_one_gpu.config = config
            self.modules(self.input_semantics, mode='calibrate')
