import ntpath
import os
import pickle
import random
import sys
import warnings

import numpy as np
import torch
import tqdm
from torch import nn
from torch.backends import cudnn

from configs import encode_config
from data import create_dataloader
from metric import get_fid, get_mAP
from metric.inception import InceptionV3
from metric.mAP_score import DRNSeg
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
    if len(opt.gpu_ids) > 1:
        warnings.warn('The code only supports single GPU. Only gpu [%d] will be used.' % opt.gpu_ids[0])
    if opt.phase == 'train':
        warnings.warn('You are using training set for evaluation.')


if __name__ == '__main__':
    opt = SearchOptions().parse()
    print(' '.join(sys.argv), flush=True)
    check(opt)
    set_seed(opt.seed)

    if 'resnet' in opt.netG:
        from configs.resnet_configs import get_configs
    elif 'spade' in opt.netG:
        # TODO
        raise NotImplementedError
    else:
        raise NotImplementedError
    configs = get_configs(config_name=opt.config_set)
    configs = list(configs.all_configs())

    dataloader = create_dataloader(opt)
    model = create_model(opt)
    model.setup(opt)
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
    for config in tqdm.tqdm(configs):
        fakes, names = [], []
        for i, data_i in enumerate(dataloader):
            model.set_input(data_i)
            if i == 0:
                macs = model.profile(config)
            model.test(config)
            fakes.append(model.fake_B.cpu())
            for path in model.get_image_paths():
                short_path = ntpath.basename(path)
                name = os.path.splitext(short_path)[0]
                names.append(name)

        result = {'config_str': encode_config(config), 'macs': macs}
        if not opt.no_fid:
            fid = get_fid(fakes, inception_model, npz, device, opt.batch_size, use_tqdm=False)
            result['fid'] = fid
        if 'cityscapes' in opt.dataroot and opt.direction == 'BtoA':
            mAP = get_mAP(fakes, names, drn_model, device,
                          data_dir=opt.cityscapes_path,
                          batch_size=opt.batch_size,
                          num_workers=opt.num_threads,
                          use_tqdm=False)
            result['mAP'] = mAP
        print(result, flush=True)
        results.append(result)

    os.makedirs(os.path.dirname(opt.output_path), exist_ok=True)
    with open(opt.output_path, 'wb') as f:
        pickle.dump(results, f)
    print('Successfully finish searching!!!', flush=True)