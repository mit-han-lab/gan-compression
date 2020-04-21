import ntpath
import os
import sys
import warnings

import numpy as np
import tqdm
from torch import nn

from configs import decode_config
from data import create_dataloader
from metric import get_mAP, get_fid
from metric.inception import InceptionV3
from metric.mAP_score import DRNSeg
from models import create_model
from options.test_options import TestOptions
from utils import html, util


def save_images(webpage, visuals, image_path, opt):
    def convert_visuals_to_numpy(visuals):
        for key, t in visuals.items():
            tile = opt.batch_size > 8
            if key == 'labels':
                t = util.tensor2label(t, opt.input_nc + 2, tile=tile)
            else:
                t = util.tensor2im(t, tile=tile)
            visuals[key] = t
        return visuals

    visuals = convert_visuals_to_numpy(visuals)

    image_dir = webpage.get_image_dir()
    short_path = ntpath.basename(image_path[0])
    name = os.path.splitext(short_path)[0]

    webpage.add_header(name)
    ims = []
    txts = []
    links = []

    for label, image_numpy in visuals.items():
        image_name = os.path.join(label, '%s.png' % (name))
        save_path = os.path.join(image_dir, image_name)
        util.save_image(image_numpy, save_path, create_dir=True)

        ims.append(image_name)
        txts.append(label)
        links.append(image_name)
    webpage.add_images(ims, txts, links, width=opt.display_winsize)


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

    web_dir = opt.results_dir  # define the website directory
    webpage = html.HTML(web_dir, 'restore_G_path: %s' % (opt.restore_G_path))
    fakes, names = [], []
    for i, data in enumerate(tqdm.tqdm(dataloader)):
        model.set_input(data)  # unpack data from data loader
        if i == 0 and opt.need_profile:
            model.profile(config)
        model.test(config)  # run inference
        visuals = model.get_current_visuals()  # get image results
        generated = visuals['fake_B'].cpu()
        fakes.append(generated)
        for path in model.get_image_paths():
            short_path = ntpath.basename(path)
            name = os.path.splitext(short_path)[0]
            names.append(name)
        if i < opt.num_test:
            save_images(webpage, visuals, model.get_image_paths(), opt)
    webpage.save()  # save the HTML
    device = model.device

    if 'cityscapes' in opt.dataroot and not opt.no_mAP and opt.direction == 'BtoA':
        drn_model = DRNSeg('drn_d_105', 19, pretrained=False)
        util.load_network(drn_model, opt.drn_path, verbose=False)
        if len(opt.gpu_ids) > 0:
            drn_model = nn.DataParallel(drn_model, opt.gpu_ids)
        drn_model.eval()
        mAP = get_mAP(fakes, names, drn_model, device,
                      data_dir=opt.cityscapes_path,
                      batch_size=opt.batch_size,
                      num_workers=opt.num_threads)
        print('mAP: %.2f' % mAP)

    if not opt.no_fid:
        print('Calculating FID...', flush=True)
        block_idx = InceptionV3.BLOCK_INDEX_BY_DIM[2048]
        inception_model = InceptionV3([block_idx])
        inception_model.to(device)
        inception_model.eval()
        npz = np.load(opt.real_stat_path)
        fid = get_fid(fakes, inception_model, npz, device, opt.batch_size)
        print('fid score: %.2f' % fid, flush=True)
