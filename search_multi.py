import copy
import ntpath
import os
import pickle
import random
import sys
import warnings

import numpy as np
import torch
import tqdm
from torch import multiprocessing as mp
from torch import nn
from torch.backends import cudnn

from configs import encode_config
from data import create_dataloader
from metric import get_fid, get_mIoU
from metric.inception import InceptionV3
from metric.cityscapes_mIoU import DRNSeg
from models import create_model
from options.search_options import SearchOptions
from utils import util


def set_seed(seed):
    cudnn.benchmark = False  # if benchmark=True, deterministic will be False
    cudnn.deterministic = True
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def check(opt):
    assert opt.serial_batches
    assert opt.no_flip
    assert opt.load_size == opt.crop_size
    assert opt.preprocess == 'resize_and_crop'
    assert opt.config_set is not None
    if len(opt.gpu_ids) == 0:
        raise ValueError("Multi-gpu searching doesn't support cpu. Please specify at least one gpu.")
    if opt.phase == 'train':
        warnings.warn('You are using training set for evaluation.')


def main(configs, opt, gpu_id, queue, verbose):
    opt.gpu_ids = [gpu_id]
    dataloader = create_dataloader(opt, verbose)
    model = create_model(opt, verbose)
    model.setup(opt, verbose)
    device = model.device
    if not opt.no_fid:
        block_idx = InceptionV3.BLOCK_INDEX_BY_DIM[2048]
        inception_model = InceptionV3([block_idx])
        inception_model.to(device)
        inception_model.eval()
    if 'cityscapes' in opt.dataroot and opt.direction == 'BtoA':
        drn_model = DRNSeg('drn_d_105', 19, pretrained=False)
        util.load_network(drn_model, opt.drn_path, verbose=False)
        if len(opt.gpu_ids) > 0:
            drn_model = nn.DataParallel(drn_model, opt.gpu_ids)
        drn_model.eval()

    npz = np.load(opt.real_stat_path)
    results = []

    for data_i in dataloader:
        model.set_input(data_i)
        break

    for config in tqdm.tqdm(configs):
        qualified = True
        macs, _ = model.profile(config)
        if macs > opt.budget:
            qualified = False
        else:
            qualified = True

        fakes, names = [], []

        if qualified:
            for i, data_i in enumerate(dataloader):
                model.set_input(data_i)

                model.test(config)
                fakes.append(model.fake_B.cpu())
                for path in model.get_image_paths():
                    short_path = ntpath.basename(path)
                    name = os.path.splitext(short_path)[0]
                    names.append(name)

        result = {'config_str': encode_config(config), 'macs': macs}
        if not opt.no_fid:
            if qualified:
                fid = get_fid(fakes, inception_model, npz, device, opt.batch_size, use_tqdm=False)
                result['fid'] = fid
            else:
                result['fid'] = 1e9
        if 'cityscapes' in opt.dataroot and opt.direction == 'BtoA':
            if qualified:
                mIoU = get_cityscapes_mIoU(fakes, names, drn_model, device,
                                           data_dir=opt.cityscapes_path,
                                           batch_size=opt.batch_size,
                                           num_workers=opt.num_threads, use_tqdm=False)
                result['mIoU'] = mIoU
            else:
                result['mIoU'] = mIoU
        print(result, flush=True)
        results.append(result)
    queue.put(results)


if __name__ == '__main__':
    warnings.warn(
        'This script is deprecated. Please set up multi-GPU searching manually '
        '(for more details, please refer to ./docs/training_tutorial.md).')
    mp.set_start_method('spawn')
    opt = SearchOptions().parse()
    print(' '.join(sys.argv), flush=True)
    check(opt)
    set_seed(opt.seed)

    if 'resnet' in opt.netG:
        from configs.resnet_configs import get_configs
    elif 'spade' in opt.netG:
        # TODO
        pass
    else:
        raise NotImplementedError
    configs = get_configs(config_name=opt.config_set)
    configs = list(configs.all_configs())
    random.shuffle(configs)

    chunk_size = (len(configs) + len(opt.gpu_ids) - 1) // len(opt.gpu_ids)

    processes = []
    queue = mp.Queue()

    for i, gpu_id in enumerate(opt.gpu_ids):
        start = min(i * chunk_size, len(configs))
        end = min((i + 1) * chunk_size, len(configs))
        p = mp.Process(target=main, args=(configs[start:end], copy.deepcopy(opt), gpu_id, queue, i == 0))
        processes.append(p)
        p.start()
    for p in processes:
        p.join()

    results = []
    for p in processes:
        results += queue.get()

    os.makedirs(os.path.dirname(opt.output_path), exist_ok=True)
    with open(opt.output_path, 'wb') as f:
        pickle.dump(results, f)
    print('Successfully finish searching!!!', flush=True)
