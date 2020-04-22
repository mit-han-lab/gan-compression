import sys
import time
import warnings

import torch
import tqdm
from torch.backends import cudnn

from configs import decode_config
from data import create_dataloader
from models import create_model
from options.test_options import TestOptions


def check(opt):
    assert opt.serial_batches
    assert opt.no_flip
    assert opt.load_size == opt.crop_size
    assert opt.preprocess == 'resize_and_crop'
    assert opt.batch_size == 1

    if not opt.no_fid:
        assert opt.real_stat_path is not None
    if opt.phase == 'train':
        warnings.warn('You are using training set for inference.')


if __name__ == '__main__':
    cudnn.enabled = True
    opt = TestOptions().parse()
    print(' '.join(sys.argv))
    if opt.config_str is not None:
        assert 'super' in opt.netG or 'sub' in opt.netG
        config = decode_config(opt.config_str)
    else:
        assert 'super' not in opt.model
        config = None

    dataloader = create_dataloader(opt)
    model = create_model(opt)
    model.setup(opt)

    for data in dataloader:
        model.set_input(data)
        break

    # Warm-up times
    for i in tqdm.trange(opt.times):
        model.test(config)
        if len(opt.gpu_ids) > 0:
            torch.cuda.synchronize()

    start_time = time.time()
    for i in tqdm.trange(opt.times):
        model.test(config)
        if len(opt.gpu_ids) > 0:
            torch.cuda.synchronize()
    cost_time = time.time() - start_time
    print('Cost Time: %.2fs\tLatency: %.4fs' % (cost_time, cost_time / opt.times))
