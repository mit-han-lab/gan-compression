import copy
import ntpath
import os
import pickle
import random
import sys
import time
import warnings

import numpy as np
import torch
from torch.backends import cudnn
from tqdm import tqdm, trange

from configs import encode_config
from data import create_dataloader
from metric import create_metric_models
from metric import get_fid, get_coco_scores, get_cityscapes_mIoU
from models import create_model
from models.spade_model import SPADEModel
from options.evolution_options import EvolutionOptions


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


def tuple2item(info):
    result, config, macs = info
    ret = copy.deepcopy(result)
    ret['config_str'] = encode_config(config)
    ret['macs'] = macs
    return ret


def dict2str(d: dict):
    ret = ''
    for i, (k, v) in enumerate(d.items()):
        if i == 0:
            ret += ' {%s: ' % k
        else:
            ret += ' %s: ' % k
        if isinstance(v, float):
            ret += '%.2f' % v
        else:
            ret += str(v)
    ret += '}'
    return ret


class EvolutionSearcher:

    def __init__(self, opt):
        self.opt = opt
        if 'resnet' in opt.netG:
            from configs.resnet_configs import get_configs
        elif 'spade' in opt.netG:
            from configs.spade_configs import get_configs
        else:
            raise NotImplementedError
        self.configs = get_configs(config_name=opt.config_set)

        self.dataloader = create_dataloader(opt)
        model = create_model(opt)
        model.setup(opt)
        for data_i in self.dataloader:
            model.set_input(data_i)
            break
        self.model = model
        self.device = model.device
        self.inception_model, self.drn_model, self.deeplabv2_model = create_metric_models(opt, self.device)
        self.npz = np.load(opt.real_stat_path)
        self.macs_cache = {}
        self.result_cache = {}

    def random_sample(self):
        while True:
            sample = self.configs.sample(weighted_sample=self.opt.weighted_sample)
            macs, _ = self.model.profile(sample, verbose=False)
            macs = self.macs_cache.get(encode_config(sample))
            if macs is None:
                macs, _ = self.model.profile(sample, verbose=False)
                if len(self.macs_cache) < self.opt.max_cache_size:
                    self.macs_cache[encode_config(sample)] = macs
            if macs <= self.opt.budget:
                return sample, macs

    def mutate_sample(self, sample):
        while True:
            new_sample = copy.deepcopy(sample)
            for i in range(len(new_sample['channels'])):
                if random.random() < self.opt.mutate_prob:
                    new_sample['channels'][i] = self.configs.sample_layer(i)
            macs = self.macs_cache.get(encode_config(new_sample))
            if macs is None:
                macs, _ = self.model.profile(new_sample, verbose=False)
                if len(self.macs_cache) < self.opt.max_cache_size:
                    self.macs_cache[encode_config(new_sample)] = macs
            if macs <= self.opt.budget:
                return new_sample, macs

    def crossover_sample(self, sample1, sample2):
        while True:
            new_sample = copy.deepcopy(sample1)
            for i in range(len(new_sample['channels'])):
                new_sample['channels'][i] = random.choice([sample1['channels'][i], sample2['channels'][i]])
            macs = self.macs_cache.get(encode_config(new_sample))
            if macs is None:
                macs, _ = self.model.profile(new_sample, verbose=False)
                if len(self.macs_cache) < self.opt.max_cache_size:
                    self.macs_cache[encode_config(new_sample)] = macs
            if macs <= self.opt.budget:
                return new_sample, macs

    def evaluate(self, child_pool):
        results = []
        for child in tqdm(child_pool, position=1, desc='Evaluate   ', leave=False):
            result = self.result_cache.get(encode_config(child))
            if result is None:
                result = {}
                fakes, names = [], []
                if isinstance(self.model, SPADEModel):
                    self.model.calibrate(child)
                for i, data_i in enumerate(self.dataloader):
                    self.model.set_input(data_i)
                    self.model.test(child)
                    fakes.append(self.model.fake_B.cpu())
                    for path in self.model.get_image_paths():
                        short_path = ntpath.basename(path)
                        name = os.path.splitext(short_path)[0]
                        names.append(name)
                if self.inception_model is not None:
                    result['fid'] = get_fid(fakes, self.inception_model, self.npz,
                                            self.device, opt.batch_size, tqdm_position=2)
                if self.drn_model is not None:
                    result['mIoU'] = get_cityscapes_mIoU(fakes, names, self.drn_model, self.device,
                                                         data_dir=opt.cityscapes_path, batch_size=opt.batch_size,
                                                         num_workers=opt.num_threads, tqdm_position=2)
                if self.deeplabv2_model is not None:
                    torch.cuda.empty_cache()
                    result['accu'], result['mIoU'] = get_coco_scores(fakes, names, self.deeplabv2_model, self.device,
                                                                     opt.dataroot, 1, num_workers=0, tqdm_position=2)
                if len(self.result_cache) < self.opt.max_cache_size:
                    self.result_cache[encode_config(child)] = result
            results.append(result)
        return results

    def better(self, a, b):
        if self.opt.criterion == 'fid':
            return a < b
        else:
            return a > b

    def restore_pkl(self, path):
        with open(path, 'rb') as f:
            pkl = pickle.load(f)
        self.macs_cache = pkl['macs_cache']
        self.result_cache = pkl['result_cache']
        return pkl['population'], pkl['best_valids'], pkl['best_infos']

    def save_ckpt(self, opt, population, best_valids, best_infos, generation):
        pkl = {}
        pkl['macs_cache'] = self.macs_cache
        pkl['result_cache'] = self.result_cache
        pkl['population'] = population
        pkl['best_valids'] = best_valids
        pkl['best_infos'] = best_infos
        pkl['generation'] = generation
        with open(os.path.join(opt.output_dir, 'latest_ckpt.pkl'), 'wb') as f:
            pickle.dump(pkl, f)
        with open(os.path.join(opt.output_dir, '%d_ckpt.pkl' % generation), 'wb') as f:
            pickle.dump(pkl, f)

    def run_evolution_search(self):
        opt = self.opt
        population_size = opt.population_size
        mutation_numbers = int(round(opt.mutation_ratio * population_size))
        parents_size = int(round(opt.parent_ratio * population_size))
        print('Start Evolution...')
        last_save_time = time.time()
        parents = []
        if opt.restore_pkl_path is not None:
            population, best_valids, best_infos = self.restore_pkl(opt.restore_pkl_path)
        else:
            population, child_pool, macs_pool = [], [], []
            best_valids, best_infos = [], []
            for _ in trange(population_size, desc='Sample     '):
                sample, macs = self.random_sample()
                child_pool.append(sample)
                macs_pool.append(macs)
            results = self.evaluate(child_pool)
            for i in range(mutation_numbers):
                population.append((results[i], child_pool[i], macs_pool[i]))

        evolution_tqdm = trange(opt.generation_base, opt.generation_base + opt.evolution_iters,
                                desc='Evolution  ', position=0)
        if time.time() - last_save_time > opt.save_freq * 60:
            last_save_time = time.time()
            self.save_ckpt(opt, population, best_valids, best_infos, 1)
            evolution_tqdm.write('Save the latest results at [%s].' % (os.path.join(opt.output_dir, 'latest_ckpt.pkl')))
        for iter in evolution_tqdm:
            need_reverse = opt.criterion != 'fid'
            parents = sorted(population, key=lambda x: x[0][opt.criterion], reverse=need_reverse)[:parents_size]
            performance = parents[0][0]
            if len(best_valids) == 0 or self.better(performance[opt.criterion], best_valids[-1]):
                best_valids.append(performance[opt.criterion])
                best_infos.append(parents[0])
            evolution_tqdm.write('Iter %d: %s' % (iter, dict2str(tuple2item(best_infos[-1]))))
            population = parents
            child_pool, macs_pool = [], []
            for __ in trange(mutation_numbers, desc='Mutation   ', position=1, leave=False):
                par_sample = population[np.random.randint(parents_size)][1]
                new_sample, macs = self.mutate_sample(par_sample)
                child_pool.append(new_sample)
                macs_pool.append(macs)
            for __ in trange(population_size - mutation_numbers, desc='Cross Over ', position=1, leave=False):
                par_sample1 = population[np.random.randint(parents_size)][1]
                par_sample2 = population[np.random.randint(parents_size)][1]
                new_sample, macs = self.crossover_sample(par_sample1, par_sample2)
                child_pool.append(new_sample)
                macs_pool.append(macs)
            results = self.evaluate(child_pool)
            for i in range(mutation_numbers):
                population.append((results[i], child_pool[i], macs_pool[i]))
            if time.time() - last_save_time > opt.save_freq * 60:
                last_save_time = time.time()
                self.save_ckpt(opt, population, best_valids, best_infos, iter + 1)
                evolution_tqdm.write(
                    'Save the latest results at [%s].' % (os.path.join(opt.output_dir, 'latest_ckpt.pkl')))
        print('Finish...')
        return best_valids, best_infos, parents


def save_results(infos, path):
    results = []
    for info in infos:
        results.append(tuple2item(info))
    with open(path, 'wb') as f:
        pickle.dump(results, f)


if __name__ == '__main__':
    opt = EvolutionOptions().parse()
    print(' '.join(sys.argv), flush=True)
    check(opt)
    set_seed(opt.seed)
    os.makedirs(opt.output_dir, exist_ok=True)
    searcher = EvolutionSearcher(opt)
    best_valids, best_infos, final_parents = searcher.run_evolution_search()
    save_results(best_infos, os.path.join(opt.output_dir, 'best_infos.pkl'))
    save_results(final_parents, os.path.join(opt.output_dir, 'final_parents.pkl'))
