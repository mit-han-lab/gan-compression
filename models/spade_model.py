import argparse
import copy
import ntpath
import os
from collections import OrderedDict

import numpy as np
import torch
from tqdm import tqdm

from data import create_eval_dataloader
from data import create_train_dataloader
from metric import get_fid, get_mIoU
from metric.fid_score import InceptionV3
from metric.mIoU_score import DRNSeg
from models import networks
from models.base_model import BaseModel
from models.modules.spade_modules.spade_model_modules import SPADEModelModules
from models.modules.sync_batchnorm import DataParallelWithCallback
from utils import util


class SPADEModel(BaseModel):
    @staticmethod
    def modify_commandline_options(parser, is_train):
        assert isinstance(parser, argparse.ArgumentParser)
        parser.set_defaults(netG='sub_mobile_spade')
        parser.add_argument('--separable_conv_norm', type=str, default='instance',
                            choices=('none', 'instance', 'batch'),
                            help='whether to use instance norm for the separable convolutions')
        parser.add_argument('--norm_G', type=str, default='spadesyncbatch3x3',
                            help='instance normalization or batch normalization')
        parser.add_argument('--num_upsampling_layers',
                            choices=('normal', 'more', 'most'), default='more',
                            help="If 'more', adds upsampling layer between the two middle resnet blocks. "
                                 "If 'most', also add one more upsampling + resnet layer at the end of the generator")
        if is_train:
            parser.add_argument('--restore_G_path', type=str, default=None,
                                help='the path to restore the generator')
            parser.add_argument('--restore_D_path', type=str, default=None,
                                help='the path to restore the discriminator')
            parser.add_argument('--real_stat_path', type=str, required=True,
                                help='the path to load the groud-truth images information to compute FID.')
            parser.add_argument('--lambda_gan', type=float, default=1, help='weight for gan loss')
            parser.add_argument('--lambda_feat', type=float, default=10, help='weight for gan feature loss')
            parser.add_argument('--lambda_vgg', type=float, default=10, help='weight for vgg loss')
            parser.add_argument('--beta2', type=float, default=0.999, help='momentum term of adam')
            parser.add_argument('--no_TTUR', action='store_true', help='Use TTUR training scheme')
            parser.add_argument('--no_fid', action='store_true', help='No FID evaluation during training')
            parser.add_argument('--no_mIoU', action='store_true', help='No mIoU evaluation during training '
                                                                       '(sometimes because there are CUDA memory)')
            parser.set_defaults(netD='multi_scale', ndf=64, dataset_mode='cityscapes', batch_size=16,
                                print_freq=50, save_latest_freq=10000000000, save_epoch_freq=10,
                                nepochs=100, nepochs_decay=100, init_type='xavier')
        parser = networks.modify_commandline_options(parser, is_train)
        return parser

    def __init__(self, opt):
        super(SPADEModel, self).__init__(opt)
        self.model_names = ['G']
        self.visual_names = ['labels', 'fake_B', 'real_B']
        self.modules = SPADEModelModules(opt).to(self.device)
        if len(opt.gpu_ids) > 0:
            self.modules = DataParallelWithCallback(self.modules, device_ids=opt.gpu_ids)
            self.modules_on_one_gpu = self.modules.module
        else:
            self.modules_on_one_gpu = self.modules
        if opt.isTrain:
            self.model_names.append('D')
            self.loss_names = ['G_gan', 'G_feat', 'G_vgg', 'D_real', 'D_fake']
            self.optimizer_G, self.optimizer_D = self.modules_on_one_gpu.create_optimizers()
            self.optimizers = [self.optimizer_G, self.optimizer_D]
            if not opt.no_fid:
                block_idx = InceptionV3.BLOCK_INDEX_BY_DIM[2048]
                self.inception_model = InceptionV3([block_idx])
                self.inception_model.to(self.device)
                self.inception_model.eval()
            if 'cityscapes' in opt.dataroot and not opt.no_mIoU:
                self.drn_model = DRNSeg('drn_d_105', 19, pretrained=False)
                util.load_network(self.drn_model, opt.drn_path, verbose=False)
                self.drn_model.to(self.device)
                self.drn_model.eval()
            self.eval_dataloader = create_eval_dataloader(self.opt)
            self.best_fid = 1e9
            self.best_mIoU = -1e9
            self.fids, self.mIoUs = [], []
            self.is_best = False
            self.npz = np.load(opt.real_stat_path)
        else:
            self.modules.eval()
        self.train_dataloader = create_train_dataloader(opt)

    def set_input(self, input):
        self.data = input
        self.image_paths = input['path']
        self.labels = input['label'].to(self.device)
        self.input_semantics, self.real_B = self.preprocess_input(input)

    def test(self, config=None):
        with torch.no_grad():
            self.forward(on_one_gpu=True, config=config)

    def preprocess_input(self, data):
        # move to GPU and change data types
        data['label'] = data['label'].long()
        data['label'] = data['label'].to(self.device)
        data['instance'] = data['instance'].to(self.device)
        data['image'] = data['image'].to(self.device)

        # create one-hot label map
        label_map = data['label']
        bs, _, h, w = label_map.size()
        nc = self.opt.input_nc + 1 if self.opt.contain_dontcare_label \
            else self.opt.input_nc
        input_label = torch.zeros([bs, nc, h, w], device=self.device)
        input_semantics = input_label.scatter_(1, label_map, 1.0)

        # concatenate instance map if it exists
        if not self.opt.no_instance:
            inst_map = data['instance']
            instance_edge_map = self.get_edges(inst_map)
            input_semantics = torch.cat((input_semantics, instance_edge_map), dim=1)

        return input_semantics, data['image']

    def forward(self, on_one_gpu=False, config=None):
        if config is not None:
            self.modules_on_one_gpu.config = config
        if on_one_gpu:
            self.fake_B = self.modules_on_one_gpu(self.input_semantics)
        else:
            self.fake_B = self.modules(self.input_semantics)

    def get_edges(self, t):
        edge = torch.zeros(t.size(), dtype=torch.uint8, device=self.device)
        edge[:, :, :, 1:] = edge[:, :, :, 1:] | ((t[:, :, :, 1:] != t[:, :, :, :-1]).byte())
        edge[:, :, :, :-1] = edge[:, :, :, :-1] | ((t[:, :, :, 1:] != t[:, :, :, :-1]).byte())
        edge[:, :, 1:, :] = edge[:, :, 1:, :] | ((t[:, :, 1:, :] != t[:, :, :-1, :]).byte())
        edge[:, :, :-1, :] = edge[:, :, :-1, :] | ((t[:, :, 1:, :] != t[:, :, :-1, :]).byte())
        return edge.float()

    def profile(self, config=None, verbose=True):
        if config is not None:
            self.modules_on_one_gpu.config = config
        macs, params = self.modules_on_one_gpu.profile(self.input_semantics[:1])
        if verbose:
            print('MACs: %.3fG\tParams: %.3fM' % (macs / 1e9, params / 1e6), flush=True)
        return macs, params

    def backward_G(self):
        losses = self.modules(self.input_semantics, self.real_B, mode='G_loss')
        loss_G = losses['loss_G'].mean()
        for loss_name in self.loss_names:
            if loss_name.startswith('G'):
                setattr(self, 'loss_%s' % loss_name, losses[loss_name].detach().mean())
        loss_G.backward()

    def backward_D(self):
        losses = self.modules(self.input_semantics, self.real_B, mode='D_loss')
        loss_D = losses['loss_D'].mean()
        for loss_name in self.loss_names:
            if loss_name.startswith('D'):
                setattr(self, 'loss_%s' % loss_name, losses[loss_name].detach().mean())
        loss_D.backward()

    def optimize_parameters(self, steps):
        # self.forward()
        self.set_requires_grad(self.modules_on_one_gpu.netD, False)
        self.optimizer_G.zero_grad()
        self.backward_G()
        self.optimizer_G.step()
        self.set_requires_grad(self.modules_on_one_gpu.netD, True)
        self.optimizer_D.zero_grad()
        self.backward_D()
        self.optimizer_D.step()

    def evaluate_model(self, step):
        self.is_best = False
        save_dir = os.path.join(self.opt.log_dir, 'eval', str(step))
        os.makedirs(save_dir, exist_ok=True)
        self.modules_on_one_gpu.netG.eval()
        torch.cuda.empty_cache()
        fakes, names = [], []
        ret = {}
        cnt = 0
        for i, data_i in enumerate(tqdm(self.eval_dataloader)):
            self.set_input(data_i)
            self.test()
            fakes.append(self.fake_B.cpu())
            for j in range(len(self.image_paths)):
                short_path = ntpath.basename(self.image_paths[j])
                name = os.path.splitext(short_path)[0]
                names.append(name)
                if cnt < 10:
                    input_im = util.tensor2label(self.input_semantics[j], self.opt.input_nc + 2)
                    real_im = util.tensor2im(self.real_B[j])
                    fake_im = util.tensor2im(self.fake_B[j])
                    util.save_image(input_im, os.path.join(save_dir, 'input', '%s.png' % name), create_dir=True)
                    util.save_image(real_im, os.path.join(save_dir, 'real', '%s.png' % name), create_dir=True)
                    util.save_image(fake_im, os.path.join(save_dir, 'fake', '%s.png' % name), create_dir=True)
                cnt += 1
        if not self.opt.no_fid:
            fid = get_fid(fakes, self.inception_model, self.npz,
                          device=self.device, batch_size=self.opt.eval_batch_size)
            if fid < self.best_fid:
                self.is_best = True
                self.best_fid = fid
            self.fids.append(fid)
            if len(self.fids) > 3:
                self.fids.pop(0)
            ret['metric/fid'] = fid
            ret['metric/fid-mean'] = sum(self.fids) / len(self.fids)
            ret['metric/fid-best'] = self.best_fid
        if 'cityscapes' in self.opt.dataroot and not self.opt.no_mIoU:
            mIoU = get_mIoU(fakes, names, self.drn_model, self.device,
                            table_path=self.opt.table_path,
                            data_dir=self.opt.cityscapes_path,
                            batch_size=self.opt.eval_batch_size,
                            num_workers=self.opt.num_threads)
            if mIoU > self.best_mIoU:
                self.is_best = True
                self.best_mIoU = mIoU
            self.mIoUs.append(mIoU)
            if len(self.mIoUs) > 3:
                self.mIoUs = self.mIoUs[1:]
            ret['metric/mIoU'] = mIoU
            ret['metric/mIoU-mean'] = sum(self.mIoUs) / len(self.mIoUs)
            ret['metric/mIoU-best'] = self.best_mIoU

        self.modules_on_one_gpu.netG.train()
        torch.cuda.empty_cache()
        return ret

    def print_networks(self):
        print('---------- Networks initialized -------------')
        for name in self.model_names:
            if isinstance(name, str):
                net = getattr(self.modules_on_one_gpu, 'net' + name)
                num_params = 0
                for param in net.parameters():
                    num_params += param.numel()
                print(net)
                print('[Network %s] Total number of parameters : %.3f M' % (name, num_params / 1e6))
                if hasattr(self.opt, 'log_dir'):
                    with open(os.path.join(self.opt.log_dir, 'net' + name + '.txt'), 'w') as f:
                        f.write(str(net) + '\n')
                        f.write('[Network %s] Total number of parameters : %.3f M\n' % (name, num_params / 1e6))
        print('-----------------------------------------------')

    def load_networks(self, verbose=True):
        self.modules_on_one_gpu.load_networks(verbose)

    def get_current_visuals(self):
        """Return visualization images. train.py will display these images with visdom, and save the images to a HTML"""
        visual_ret = OrderedDict()
        for name in self.visual_names:
            if isinstance(name, str) and hasattr(self, name):
                visual_ret[name] = getattr(self, name)
        return visual_ret

    def save_networks(self, epoch):
        self.modules_on_one_gpu.save_networks(epoch, self.save_dir)

    def calibrate(self, config):
        self.modules_on_one_gpu.netG.train()
        config = copy.deepcopy(config)
        for i, data in enumerate(self.train_dataloader):
            self.set_input(data)
            if i == 0:
                config['calibrate_bn'] = True
            self.modules_on_one_gpu.config = config
            self.modules(self.input_semantics, mode='calibrate')
        self.modules_on_one_gpu.netG.eval()
