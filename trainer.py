import os
import random
import sys
import time
import warnings

import numpy as np
import torch
from torch.backends import cudnn
from tqdm import tqdm, trange

from data import create_dataloader
from utils.logger import Logger


def set_seed(seed):
    cudnn.benchmark = False  # if benchmark=True, deterministic will be False
    cudnn.deterministic = True
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


class Trainer:
    def __init__(self, task):
        if task == 'train':
            from options.train_options import TrainOptions as Options
            from models import create_model as create_model
        elif task == 'distill':
            from options.distill_options import DistillOptions as Options
            from distillers import create_distiller as create_model
        elif task == 'supernet':
            from options.supernet_options import SupernetOptions as Options
            from supernets import create_supernet as create_model
        else:
            raise NotImplementedError('Unknown task [%s]!!!' % task)
        opt = Options().parse()
        opt.tensorboard_dir = opt.log_dir if opt.tensorboard_dir is None else opt.tensorboard_dir
        print(' '.join(sys.argv))
        if opt.phase != 'train':
            warnings.warn('You are not using training set for %s!!!' % task)
        with open(os.path.join(opt.log_dir, 'opt.txt'), 'a') as f:
            f.write(' '.join(sys.argv) + '\n')
        set_seed(opt.seed)

        dataloader = create_dataloader(opt)  # create a dataset given opt.dataset_mode and other options
        dataset_size = len(dataloader.dataset)  # get the number of images in the dataset.
        print('The number of training images = %d' % dataset_size)

        model = create_model(opt)  # create a model given opt.model and other options
        model.setup(opt)  # regular setup: load and print networks; create schedulers
        logger = Logger(opt)

        self.opt = opt
        self.dataloader = dataloader
        self.model = model
        self.logger = logger

    def evaluate(self, epoch, iter, message):
        start_time = time.time()
        metrics = self.model.evaluate_model(iter)
        self.logger.print_current_metrics(epoch, iter, metrics, time.time() - start_time)
        self.logger.plot(metrics, iter)
        self.logger.print_info(message)
        self.model.save_networks('latest')

    def start(self):
        opt = self.opt
        dataloader = self.dataloader
        model = self.model
        logger = self.logger

        start_epoch = opt.epoch_base
        end_epoch = opt.nepochs + opt.nepochs_decay
        total_iter = opt.iter_base
        epoch_tqdm = trange(start_epoch, end_epoch + 1, desc='Epoch      ', position=0, leave=False)
        self.logger.set_progress_bar(epoch_tqdm)
        for epoch in epoch_tqdm:
            epoch_start_time = time.time()  # timer for entire epoch
            for i, data_i in enumerate(tqdm(dataloader, desc='Batch      ', position=1, leave=False)):
                iter_start_time = time.time()
                total_iter += 1
                model.set_input(data_i)
                model.optimize_parameters(total_iter)

                if total_iter % opt.print_freq == 0:
                    losses = model.get_current_losses()
                    logger.print_current_errors(epoch, total_iter, losses, time.time() - iter_start_time)
                    logger.plot(losses, total_iter)

                if total_iter % opt.save_latest_freq == 0:
                    self.evaluate(epoch, total_iter,
                                  'Saving the latest model (epoch %d, total_iters %d)' % (epoch, total_iter))
                    if model.is_best:
                        model.save_networks('iter%d' % total_iter)

                if opt.scheduler_counter == 'iter':
                    model.update_learning_rate(epoch, total_iter, logger=logger)

                if total_iter >= opt.niters:
                    break
            logger.print_info(
                'End of epoch %d / %d \t Time Taken: %.2f sec' % (epoch, end_epoch, time.time() - epoch_start_time))
            if epoch % opt.save_epoch_freq == 0 or epoch == end_epoch or total_iter >= opt.niters:
                self.evaluate(epoch, total_iter,
                              'Saving the model at the end of epoch %d, iters %d' % (epoch, total_iter))
                model.save_networks(epoch)
            if opt.scheduler_counter == 'epoch':
                model.update_learning_rate(epoch, total_iter, logger=logger)
