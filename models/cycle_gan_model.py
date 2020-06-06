import itertools
import ntpath
import os

import numpy as np
import torch
from torch import nn
from tqdm import tqdm

import models.modules.loss
from data import create_eval_dataloader
from metric import get_fid, get_mIoU
from metric.inception import InceptionV3
from metric.mIoU_score import DRNSeg
from models import networks
from models.base_model import BaseModel
from utils import util
from utils.image_pool import ImagePool


class CycleGANModel(BaseModel):
    """
    This class implements the CycleGAN model, for learning image-to-image translation without paired data.

    The model training requires '--dataset_mode unaligned' dataset.
    By default, it uses a '--netG resnet_9blocks' ResNet generator,
    a '--netD basic' discriminator (PatchGAN introduced by pix2pix),
    and a least-square GANs objective ('--gan_mode lsgan').

    CycleGAN paper: https://arxiv.org/pdf/1703.10593.pdf
    """

    @staticmethod
    def modify_commandline_options(parser, is_train=True):
        """Add new dataset-specific options, and rewrite default values for existing options.

        Parameters:
            parser          -- original option parser
            is_train (bool) -- whether training phase or test phase. You can use this flag to add training-specific or test-specific options.

        Returns:
            the modified parser.

        For CycleGAN, in addition to GAN losses, we introduce lambda_A, lambda_B, and lambda_identity for the following losses.
        A (source domain), B (target domain).
        Generators: G_A: A -> B; G_B: B -> A.
        Discriminators: D_A: G_A(A) vs. B; D_B: G_B(B) vs. A.
        Forward cycle loss:  lambda_A * ||G_B(G_A(A)) - A|| (Eqn. (2) in the paper)
        Backward cycle loss: lambda_B * ||G_A(G_B(B)) - B|| (Eqn. (2) in the paper)
        Identity loss (optional): lambda_identity * (||G_A(B) - B|| * lambda_B + ||G_B(A) - A|| * lambda_A) (Sec 5.2 "Photo generation from paintings" in the paper)
        Dropout is not used in the original CycleGAN paper.
        """
        assert is_train
        parser = super(CycleGANModel, CycleGANModel).modify_commandline_options(parser, is_train)
        parser.add_argument('--restore_G_A_path', type=str, default=None,
                            help='the path to restore the generator G_A')
        parser.add_argument('--restore_D_A_path', type=str, default=None,
                            help='the path to restore the discriminator D_A')
        parser.add_argument('--restore_G_B_path', type=str, default=None,
                            help='the path to restore the generator G_B')
        parser.add_argument('--restore_D_B_path', type=str, default=None,
                            help='the path to restore the discriminator D_B')
        parser.add_argument('--lambda_A', type=float, default=10.0,
                            help='weight for cycle loss (A -> B -> A)')
        parser.add_argument('--lambda_B', type=float, default=10.0,
                            help='weight for cycle loss (B -> A -> B)')
        parser.add_argument('--lambda_identity', type=float, default=0.5,
                            help='use identity mapping. '
                                 'Setting lambda_identity other than 0 has an effect of scaling the weight of the identity mapping loss. '
                                 'For example, if the weight of the identity loss should be 10 times smaller than the weight of the reconstruction loss, please set lambda_identity = 0.1')
        parser.add_argument('--real_stat_A_path', type=str, required=True,
                            help='the path to load the ground-truth A images information to compute FID.')
        parser.add_argument('--real_stat_B_path', type=str, required=True,
                            help='the path to load the ground-truth B images information to compute FID.')
        parser.set_defaults(norm='instance', dataset_mode='unaligned',
                            batch_size=1, ndf=64, gan_mode='lsgan',
                            nepochs=100, nepochs_decay=100, save_epoch_freq=20)
        return parser

    def __init__(self, opt):
        """Initialize the CycleGAN class.

        Parameters:
            opt (Option class)-- stores all the experiment flags; needs to be a subclass of BaseOptions
        """
        assert opt.isTrain
        assert opt.direction == 'AtoB'
        assert opt.dataset_mode == 'unaligned'
        BaseModel.__init__(self, opt)
        # specify the training losses you want to print out. The training/test scripts will call <BaseModel.get_current_losses>
        self.loss_names = ['D_A', 'G_A', 'G_cycle_A', 'G_idt_A',
                           'D_B', 'G_B', 'G_cycle_B', 'G_idt_B']
        # specify the images you want to save/display. The training/test scripts will call <BaseModel.get_current_visuals>
        visual_names_A = ['real_A', 'fake_B', 'rec_A']
        visual_names_B = ['real_B', 'fake_A', 'rec_B']
        if self.opt.lambda_identity > 0.0:  # if identity loss is used, we also visualize idt_B=G_A(B) ad idt_A=G_A(B)
            visual_names_A.append('idt_B')
            visual_names_B.append('idt_A')

        self.visual_names = visual_names_A + visual_names_B  # combine visualizations for A and B
        # specify the models you want to save to the disk. The training/test scripts will call <BaseModel.save_networks> and <BaseModel.load_networks>.
        self.model_names = ['G_A', 'G_B', 'D_A', 'D_B']

        # define networks (both Generators and discriminators)
        # The naming is different from those used in the paper.
        # Code (vs. paper): G_A (G), G_B (F), D_A (D_Y), D_B (D_X)
        self.netG_A = networks.define_G(opt.input_nc, opt.output_nc, opt.ngf, opt.netG, opt.norm,
                                        opt.dropout_rate, opt.init_type, opt.init_gain, self.gpu_ids)
        self.netG_B = networks.define_G(opt.output_nc, opt.input_nc, opt.ngf, opt.netG, opt.norm,
                                        opt.dropout_rate, opt.init_type, opt.init_gain, self.gpu_ids)

        self.netD_A = networks.define_D(opt.output_nc, opt.ndf, opt.netD,
                                        opt.n_layers_D, opt.norm, opt.init_type, opt.init_gain, self.gpu_ids)
        self.netD_B = networks.define_D(opt.input_nc, opt.ndf, opt.netD,
                                        opt.n_layers_D, opt.norm, opt.init_type, opt.init_gain, self.gpu_ids)

        if opt.lambda_identity > 0.0:  # only works when input and output images have the same number of channels
            assert (opt.input_nc == opt.output_nc)
        self.fake_A_pool = ImagePool(opt.pool_size)  # create image buffer to store previously generated images
        self.fake_B_pool = ImagePool(opt.pool_size)  # create image buffer to store previously generated images

        # define loss functions
        self.criterionGAN = models.modules.loss.GANLoss(opt.gan_mode).to(self.device)  # define GAN loss.
        self.criterionCycle = torch.nn.L1Loss()
        self.criterionIdt = torch.nn.L1Loss()

        # initialize optimizers; schedulers will be automatically created by function <BaseModel.setup>.
        self.optimizer_G = torch.optim.Adam(itertools.chain(self.netG_A.parameters(), self.netG_B.parameters()),
                                            lr=opt.lr, betas=(opt.beta1, 0.999))
        self.optimizer_D = torch.optim.Adam(itertools.chain(self.netD_A.parameters(), self.netD_B.parameters()),
                                            lr=opt.lr, betas=(opt.beta1, 0.999))

        self.optimizers = []
        self.optimizers.append(self.optimizer_G)
        self.optimizers.append(self.optimizer_D)

        self.eval_dataloader_AtoB = create_eval_dataloader(self.opt, direction='AtoB')
        self.eval_dataloader_BtoA = create_eval_dataloader(self.opt, direction='BtoA')

        block_idx = InceptionV3.BLOCK_INDEX_BY_DIM[2048]
        self.inception_model = InceptionV3([block_idx])
        self.inception_model.to(self.device)
        self.inception_model.eval()

        if 'cityscapes' in opt.dataroot:
            self.drn_model = DRNSeg('drn_d_105', 19, pretrained=False)
            util.load_network(self.drn_model, opt.drn_path, verbose=False)
            if len(opt.gpu_ids) > 0:
                self.drn_model = nn.DataParallel(self.drn_model, opt.gpu_ids)
            self.drn_model.eval()

        self.best_fid_A, self.best_fid_B = 1e9, 1e9
        self.best_mIoU = -1e9
        self.fids_A, self.fids_B = [], []
        self.mIoUs = []
        self.is_best = False
        self.npz_A = np.load(opt.real_stat_A_path)
        self.npz_B = np.load(opt.real_stat_B_path)

    def set_input(self, input):
        """Unpack input data from the dataloader and perform necessary pre-processing steps.

        Parameters:
            input (dict): include the data itself and its metadata information.

        The option 'direction' can be used to swap domain A and domain B.
        """
        # Since it is a cycle.
        self.real_A = input['A'].to(self.device)
        self.real_B = input['B'].to(self.device)

    def set_single_input(self, input):
        self.real_A = input['A'].to(self.device)
        self.image_paths = input['A_paths']

    def forward(self):
        """Run forward pass; called by both functions <optimize_parameters> and <test>."""
        self.fake_B = self.netG_A(self.real_A)  # G_A(A)
        self.rec_A = self.netG_B(self.fake_B)  # G_B(G_A(A))
        self.fake_A = self.netG_B(self.real_B)  # G_B(B)
        self.rec_B = self.netG_A(self.fake_A)  # G_A(G_B(B))

    def backward_D_basic(self, netD, real, fake):
        """Calculate GAN loss for the discriminator

        Parameters:
            netD (network)      -- the discriminator D
            real (tensor array) -- real images
            fake (tensor array) -- images generated by a generator

        Return the discriminator loss.
        We also call loss_D.backward() to calculate the gradients.
        """
        # Real
        pred_real = netD(real)
        loss_D_real = self.criterionGAN(pred_real, True)
        # Fake
        pred_fake = netD(fake.detach())
        loss_D_fake = self.criterionGAN(pred_fake, False)
        # Combined loss and calculate gradients
        loss_D = (loss_D_real + loss_D_fake) * 0.5
        loss_D.backward()
        return loss_D

    def backward_D_A(self):
        """Calculate GAN loss for discriminator D_A"""
        fake_B = self.fake_B_pool.query(self.fake_B)
        self.loss_D_A = self.backward_D_basic(self.netD_A, self.real_B, fake_B)

    def backward_D_B(self):
        """Calculate GAN loss for discriminator D_B"""
        fake_A = self.fake_A_pool.query(self.fake_A)
        self.loss_D_B = self.backward_D_basic(self.netD_B, self.real_A, fake_A)

    def backward_G(self):
        """Calculate the loss for generators G_A and G_B"""
        lambda_idt = self.opt.lambda_identity
        lambda_A = self.opt.lambda_A
        lambda_B = self.opt.lambda_B
        # Identity loss
        if lambda_idt > 0:
            # G_A should be identity if real_B is fed: ||G_A(B) - B||
            self.idt_A = self.netG_A(self.real_B)
            self.loss_G_idt_A = self.criterionIdt(self.idt_A, self.real_B) * lambda_B * lambda_idt
            # G_B should be identity if real_A is fed: ||G_B(A) - A||
            self.idt_B = self.netG_B(self.real_A)
            self.loss_G_idt_B = self.criterionIdt(self.idt_B, self.real_A) * lambda_A * lambda_idt
        else:
            self.loss_G_idt_A = 0
            self.loss_G_idt_B = 0

        # GAN loss D_A(G_A(A))
        self.loss_G_A = self.criterionGAN(self.netD_A(self.fake_B), True)
        # GAN loss D_B(G_B(B))
        self.loss_G_B = self.criterionGAN(self.netD_B(self.fake_A), True)
        # Forward cycle loss || G_B(G_A(A)) - A||
        self.loss_G_cycle_A = self.criterionCycle(self.rec_A, self.real_A) * lambda_A
        # Backward cycle loss || G_A(G_B(B)) - B||
        self.loss_G_cycle_B = self.criterionCycle(self.rec_B, self.real_B) * lambda_B
        # combined loss and calculate gradients
        self.loss_G = self.loss_G_A + self.loss_G_B + self.loss_G_cycle_A + self.loss_G_cycle_B + self.loss_G_idt_A + self.loss_G_idt_B
        self.loss_G.backward()

    def optimize_parameters(self):
        """Calculate losses, gradients, and update network weights; called in every training iteration"""
        # forward
        self.forward()  # compute fake images and reconstruction images.
        # G_A and G_B
        self.set_requires_grad([self.netD_A, self.netD_B], False)  # Ds require no gradients when optimizing Gs
        self.optimizer_G.zero_grad()  # set G_A and G_B's gradients to zero
        self.backward_G()  # calculate gradients for G_A and G_B
        self.optimizer_G.step()  # update G_A and G_B's weights
        # D_A and D_B
        self.set_requires_grad([self.netD_A, self.netD_B], True)
        self.optimizer_D.zero_grad()  # set D_A and D_B's gradients to zero
        self.backward_D_A()  # calculate gradients for D_A
        self.backward_D_B()  # calculate graidents for D_B
        self.optimizer_D.step()  # update D_A and D_B's weights

    def test_single_side(self, direction):
        generator = getattr(self, 'netG_%s' % direction[0])
        with torch.no_grad():
            self.fake_B = generator(self.real_A)

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
            # print(len(eval_dataset))
            for i, data_i in enumerate(tqdm(eval_dataloader)):
                self.set_single_input(data_i)
                self.test_single_side(direction)
                # print(self.image_paths)
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
                          device=self.device, batch_size=self.opt.eval_batch_size)
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

            if 'cityscapes' in self.opt.dataroot and direction == 'BtoA':
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

        self.netG_A.train()
        self.netG_B.train()
        return ret
