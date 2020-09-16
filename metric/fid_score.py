#!/usr/bin/env python3
"""Calculates the Frechet Inception Distance (FID) to evalulate GANs

The FID metric calculates the distance between two distributions of images.
Typically, we have summary statistics (mean & covariance matrix) of one
of these distributions, while the 2nd distribution is given by a GAN.

When run as a stand-alone program, it compares the distribution of
images that are stored as PNG/JPEG at a specified location with a
distribution given by summary statistics (in pickle format).

The FID is calculated by assuming that X_1 and X_2 are the activations of
the pool_3 layer of the inception net for generated samples and real world
samples respectively.

See --help to see further details.

Code apapted from https://github.com/bioinf-jku/TTUR to use PyTorch instead
of Tensorflow

Copyright 2018 Institute of Bioinformatics, JKU Linz

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

   http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
"""
import os
import pathlib
from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter

import cv2
import numpy as np
import torch
from cv2 import imread
from scipy import linalg
from torch.nn.functional import adaptive_avg_pool2d

from .inception import InceptionV3

# try:
#     from tqdm import tqdm
# except ImportError:
#     # If not tqdm is not available, provide a mock version of it
#     def tqdm(x):
#         return x

parser = ArgumentParser(formatter_class=ArgumentDefaultsHelpFormatter)
parser.add_argument('path', type=str, nargs=2,
                    help=('Path to the generated images or '
                          'to .npz statistic files'))
parser.add_argument('--batch-size', type=int, default=50,
                    help='Batch size to use')
parser.add_argument('--dims', type=int, default=2048,
                    choices=list(InceptionV3.BLOCK_INDEX_BY_DIM),
                    help=('Dimensionality of Inception features to use. '
                          'By default, uses pool3 features'))
parser.add_argument('-c', '--gpu', default='', type=str,
                    help='GPU to use (leave blank for CPU only)')


def get_activations(files, model, batch_size=50, dims=2048,
                    cuda=False, verbose=False, use_tqdm=False):
    """Calculates the activations of the pool_3 layer for all images.

    Params:
    -- files       : List of image files paths
    -- model       : Instance of inception model
    -- batch_size  : Batch size of images for the model to process at once.
                     Make sure that the number of samples is a multiple of
                     the batch size, otherwise some samples are ignored. This
                     behavior is retained to match the original FID score
                     implementation.
    -- dims        : Dimensionality of features returned by Inception
    -- cuda        : If set to True, use GPU
    -- verbose     : If set to True and parameter out_step is given, the number
                     of calculated batches is reported.
    Returns:
    -- A numpy array of dimension (num images, dims) that contains the
       activations of the given tensor when feeding inception with the
       query tensor.
    """
    model.eval()

    if len(files) % batch_size != 0:
        print(('Warning: number of images is not a multiple of the '
               'batch size. Some samples are going to be ignored.'))
    if batch_size > len(files):
        print(('Warning: batch size is bigger than the datasets size. '
               'Setting batch size to datasets size'))
        batch_size = len(files)
    # print(len(files), batch_size)

    n_batches = len(files) // batch_size
    n_used_imgs = n_batches * batch_size

    pred_arr = np.empty((n_used_imgs, dims))

    if use_tqdm:
        from tqdm import tqdm
    else:
        def tqdm(x):
            return x
    for i in tqdm(range(n_batches)):
        if verbose:
            print('\rPropagating batch %d/%d' % (i + 1, n_batches),
                  end='', flush=True)
        start = i * batch_size
        end = start + batch_size
        images = np.array([imread(str(f)).astype(np.float32)
                           for f in files[start:end]])
        if len(images.shape) != 4:
            images = imread(str(files[start]))
            images = cv2.cvtColor(images, cv2.COLOR_BGR2GRAY)
            images = np.array([images.astype(np.float32)])
        # Reshape to (n_images, 3, height, width)
        images = images.transpose((0, 3, 1, 2))
        images /= 255

        batch = torch.from_numpy(images).type(torch.FloatTensor)
        if cuda:
            batch = batch.cuda()
        with torch.no_grad():
            pred = model(batch)[0]

        # If model output is not scalar, apply global spatial average pooling.
        # This happens if you choose a dimensionality not equal 2048.

        if pred.shape[2] != 1 or pred.shape[3] != 1:
            pred = adaptive_avg_pool2d(pred, output_size=(1, 1))

        pred_arr[start:end] = pred.cpu().data.numpy().reshape(end - start, -1)

    if verbose:
        print(' done')

    return pred_arr


def get_activations_from_ims(ims, model, batch_size=50, dims=2048,
                             device=None, verbose=False, tqdm_position=None):
    """Calculates the activations of the pool_3 layer for all images.

    Params:
    -- files       : List of image files paths
    -- model       : Instance of inception model
    -- batch_size  : Batch size of images for the model to process at once.
                     Make sure that the number of samples is a multiple of
                     the batch size, otherwise some samples are ignored. This
                     behavior is retained to match the original FID score
                     implementation.
    -- dims        : Dimensionality of features returned by Inception
    -- cuda        : If set to True, use GPU
    -- verbose     : If set to True and parameter out_step is given, the number
                     of calculated batches is reported.
    Returns:
    -- A numpy array of dimension (num images, dims) that contains the
       activations of the given tensor when feeding inception with the
       query tensor.
    """
    model.eval()

    # if len(ims) % batch_size != 0:
    #     print(('Warning: number of images is not a multiple of the '
    #            'batch size. Some samples are going to be ignored.'))
    # if batch_size > len(ims):
    #     print(('Warning: batch size is bigger than the datasets size. '
    #            'Setting batch size to datasets size'))
    #     batch_size = len(ims)
    # print(len(files), batch_size)

    n_batches = (len(ims) + batch_size - 1) // batch_size
    n_used_imgs = len(ims)

    pred_arr = np.empty((n_used_imgs, dims))

    if tqdm_position is None or tqdm_position >= 0:
        import tqdm
        dataloader_tqdm = tqdm.trange(n_batches, desc='FID', position=tqdm_position, leave=False)
    else:
        dataloader_tqdm = range(n_batches)
    for i in dataloader_tqdm:
        start = i * batch_size
        end = start + batch_size
        if end > len(ims):
            end = len(ims)
        images = ims[start:end]
        if images.shape[1] != 3:
            images = images.transpose((0, 3, 1, 2))
        images /= 255

        batch = torch.from_numpy(images).type(torch.FloatTensor).to(device)
        # if cuda:
        #     batch = batch.cuda()
        with torch.no_grad():
            pred = model(batch)[0]

        # If model output is not scalar, apply global spatial average pooling.
        # This happens if you choose a dimensionality not equal 2048.
        # print(pred.shape)
        if pred.shape[2] != 1 or pred.shape[3] != 1:
            pred = adaptive_avg_pool2d(pred, output_size=(1, 1))

        pred_arr[start:end] = pred.cpu().data.numpy().reshape(end - start, -1)

    if verbose:
        print(' done')

    return pred_arr


def calculate_frechet_distance(mu1, sigma1, mu2, sigma2, eps=1e-6):
    """Numpy implementation of the Frechet Distance.
    The Frechet distance between two multivariate Gaussians X_1 ~ N(mu_1, C_1)
    and X_2 ~ N(mu_2, C_2) is
            d^2 = ||mu_1 - mu_2||^2 + Tr(C_1 + C_2 - 2*sqrt(C_1*C_2)).

    Stable version by Dougal J. Sutherland.

    Params:
    -- mu1   : Numpy array containing the activations of a layer of the
               inception net (like returned by the function 'get_predictions')
               for generated samples.
    -- mu2   : The sample mean over activations, precalculated on an
               representative datasets set.
    -- sigma1: The covariance matrix over activations for generated samples.
    -- sigma2: The covariance matrix over activations, precalculated on an
               representative datasets set.

    Returns:
    --   : The Frechet Distance.
    """

    mu1 = np.atleast_1d(mu1)
    mu2 = np.atleast_1d(mu2)

    sigma1 = np.atleast_2d(sigma1)
    sigma2 = np.atleast_2d(sigma2)

    assert mu1.shape == mu2.shape, \
        'Training and test mean vectors have different lengths'
    assert sigma1.shape == sigma2.shape, \
        'Training and test covariances have different dimensions'

    diff = mu1 - mu2

    # print(t.max())
    # print(t.min())
    # print(abs(t).mean())
    # print(t.mean())

    # Product might be almost singular
    t = sigma1.dot(sigma2)
    for i in range(30):
        # print(i)
        flag = True
        covmean, _ = linalg.sqrtm(t, disp=False)
        if not np.isfinite(covmean).all():
            msg = ('fid calculation produces singular product; '
                   'adding %s to diagonal of cov estimates') % eps
            print(msg)
            offset = np.eye(sigma1.shape[0]) * eps
            covmean = linalg.sqrtm((sigma1 + offset).dot(sigma2 + offset))

        # Numerical error might give slight imaginary component
        if np.iscomplexobj(covmean):
            if not np.allclose(np.diagonal(covmean).imag, 0, atol=1e-3):
                # raise ValueError('Imaginary component {}'.format(m))
                flag = False
            covmean = covmean.real
        if flag:
            break
    if not flag:
        print('Warning: the fid may be incorrect!')
    tr_covmean = np.trace(covmean)

    return (diff.dot(diff) + np.trace(sigma1) + np.trace(sigma2) - 2 * tr_covmean)


def calculate_activation_statistics(files, model, batch_size=50,
                                    dims=2048, cuda=False, verbose=False, use_tqdm=False):
    """Calculation of the statistics used by the FID.
    Params:
    -- files       : List of image files paths
    -- model       : Instance of inception model
    -- batch_size  : The images numpy array is split into batches with
                     batch size batch_size. A reasonable batch size
                     depends on the hardware.
    -- dims        : Dimensionality of features returned by Inception
    -- cuda        : If set to True, use GPU
    -- verbose     : If set to True and parameter out_step is given, the
                     number of calculated batches is reported.
    Returns:
    -- mu    : The mean over samples of the activations of the pool_3 layer of
               the inception model.
    -- sigma : The covariance matrix of the activations of the pool_3 layer of
               the inception model.
    """
    act = get_activations(files, model, batch_size, dims, cuda, verbose, use_tqdm=use_tqdm)
    mu = np.mean(act, axis=0)
    sigma = np.cov(act, rowvar=False)
    return mu, sigma


def _compute_statistics_of_path(path, model, batch_size, dims, cuda, use_tqdm=False):
    if path.endswith('.npz'):
        f = np.load(path)
        m, s = f['mu'][:], f['sigma'][:]
        f.close()
    else:
        path = pathlib.Path(path)
        files = list(path.glob('*.jpg')) + list(path.glob('*.png'))
        m, s = calculate_activation_statistics(files, model, batch_size,
                                               dims, cuda, use_tqdm=use_tqdm)
    return m, s


def _compute_statistics_of_ims(ims, model, batch_size, dims, device, tqdm_position=None):
    act = get_activations_from_ims(ims, model, batch_size, dims, device, verbose=False, tqdm_position=tqdm_position)
    mu = np.mean(act, axis=0)
    sigma = np.cov(act, rowvar=False)
    return mu, sigma


def calculate_fid_given_ims(ims_fake, ims_real, batch_size, cuda, dims, model=None, tqdm_position=None):
    if model is None:
        block_idx = InceptionV3.BLOCK_INDEX_BY_DIM[dims]

        model = InceptionV3([block_idx])
        if cuda:
            model.cuda()

    m1, s1 = _compute_statistics_of_ims(ims_fake, model, batch_size,
                                        dims, cuda, tqdm_position=tqdm_position)
    m2, s2 = _compute_statistics_of_ims(ims_real, model, batch_size,
                                        dims, cuda, tqdm_position=tqdm_position)

    fid_value = calculate_frechet_distance(m1, s1, m2, s2)

    return fid_value


def calculate_fid_given_paths(paths, batch_size, cuda, dims, model=None, use_tqdm=False):
    """Calculates the FID of two paths"""
    for p in paths:
        if not os.path.exists(p):
            raise RuntimeError('Invalid path: %s' % p)
    if model is None:
        block_idx = InceptionV3.BLOCK_INDEX_BY_DIM[dims]

        model = InceptionV3([block_idx])
        if cuda:
            model.cuda()
    else:
        pass

    m1, s1 = _compute_statistics_of_path(paths[0], model, batch_size,
                                         dims, cuda, use_tqdm=use_tqdm)
    m2, s2 = _compute_statistics_of_path(paths[1], model, batch_size,
                                         dims, cuda, use_tqdm=use_tqdm)
    fid_value = calculate_frechet_distance(m1, s1, m2, s2)

    return fid_value

# def main():
#     fid_value = calculate_fid_given_paths(('datasets/coco_stuff/val_img', 'datasets/coco_stuff/val_img'),
#                                           1, True, 2048)
#     print('FID: ', fid_value)
