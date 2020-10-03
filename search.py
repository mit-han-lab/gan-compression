import ntpath
import os
import pickle
import random
import sys
import time
import warnings

import numpy as np
import torch
import tqdm
from torch.backends import cudnn

from configs import encode_config
from data import create_dataloader
from metric import create_metric_models
from metric import get_fid, get_cityscapes_mIoU, get_coco_scores
from models import create_model
from models.spade_model import SPADEModel
from options.search_options import SearchOptions


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
    assert opt.config_set is not None
    if len(opt.gpu_ids) > 1:
        warnings.warn('The code only supports single GPU. Only gpu [%d] will be used.' % opt.gpu_ids[0])
    if opt.phase == 'train':
        warnings.warn('You are using training set for evaluation.')
    warnings.filterwarnings("ignore")


def restore_results(opt):
    if opt.restore_pkl_path is not None:
        with open(opt.restore_pkl_path, 'rb') as f:
            results = pickle.load(f)
    else:
        results = []
    eval_configs = set()
    for result in results:
        assert isinstance(result, dict)
        if result['macs'] > opt.budget:
            eval_configs.add(result['config_str'])
        elif result.get('fid', 0) != 0 or result.get('mIoU', 1e9) != 1e9:
            eval_configs.add(result['config_str'])
    return results, eval_configs


def save(opt, results):
    os.makedirs(os.path.dirname(opt.output_path), exist_ok=True)
    with open(opt.output_path, 'wb') as f:
        pickle.dump(results, f)


def get_config_split(opt):
    if 'resnet' in opt.netG:
        from configs.resnet_configs import get_configs
    elif 'spade' in opt.netG:
        from configs.spade_configs import get_configs
    else:
        raise NotImplementedError
    configs = list(get_configs(config_name=opt.config_set).all_configs())
    random.shuffle(configs)
    configs = np.array_split(np.array(configs), opt.num_splits)[opt.split]
    return configs


if __name__ == '__main__':
    opt = SearchOptions().parse()
    print(' '.join(sys.argv), flush=True)
    check(opt)
    set_seed(opt.seed)

    configs = get_config_split(opt)

    dataloader = create_dataloader(opt)
    model = create_model(opt)
    model.setup(opt)
    device = model.device

    inception_model, drn_model, deeplabv2_model = create_metric_models(opt, device)
    npz = np.load(opt.real_stat_path)

    results, eval_configs = restore_results(opt)

    last_save_time = time.time()

    for data_i in dataloader:
        model.set_input(data_i)
        break

    configs_tqdm = tqdm.tqdm(configs, desc='Configs    ', position=0)
    for config in configs_tqdm:
        config_str = encode_config(config)
        if config_str in eval_configs:
            continue
        macs, _ = model.profile(config, verbose=False)
        result = {'config_str': config_str, 'macs': macs}
        qualified = (macs <= opt.budget)

        fakes, names = [], []
        if qualified:
            if isinstance(model, SPADEModel):
                model.calibrate(config)
            for i, data_i in enumerate(dataloader):
                model.set_input(data_i)
                model.test(config)
                fakes.append(model.fake_B.cpu())
                for path in model.get_image_paths():
                    short_path = ntpath.basename(path)
                    name = os.path.splitext(short_path)[0]
                    names.append(name)
        if inception_model is not None:
            if qualified:
                result['fid'] = get_fid(fakes, inception_model, npz, device, opt.batch_size,
                                        tqdm_position=1)
            else:
                result['fid'] = 1e9
        if drn_model is not None:
            if qualified:
                result['mIoU'] = get_cityscapes_mIoU(fakes, names, drn_model, device, data_dir=opt.cityscapes_path,
                                                     batch_size=opt.batch_size, num_workers=opt.num_threads,
                                                     tqdm_position=1)
            else:
                result['mIoU'] = 0
        if deeplabv2_model is not None:
            if qualified:
                torch.cuda.empty_cache()
                result['accu'], result['mIoU'] = get_coco_scores(fakes, names, deeplabv2_model, device, opt.dataroot, 1,
                                                                 num_workers=0, tqdm_position=1)
            else:
                result['accu'], result['mIoU'] = 0, 0
        results.append(result)
        eval_configs.add(config_str)
        configs_tqdm.write(str(result))
        current_time = time.time()
        if current_time - last_save_time > opt.save_freq * 60:
            last_save_time = current_time
            save(opt, results)
            configs_tqdm.write('Save the latest results at [%s].' % (opt.output_path))
    save(opt, results)
    print('Successfully finish searching!!!', flush=True)
