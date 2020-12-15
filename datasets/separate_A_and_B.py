import argparse
import os

import cv2
from tqdm import tqdm
import multiprocessing


def work(input_path, pathA, pathB):
    im_AB = cv2.imread(input_path, 1)  # [H, W, C]
    H, W, C = im_AB.shape
    im_A = im_AB[:, :W // 2]
    im_B = im_AB[:, W // 2:]
    cv2.imwrite(pathA, im_A)
    cv2.imwrite(pathB, im_B)


def main(opt):
    input_dir = os.path.join(opt.input_dir, opt.phase)
    output_dirA = os.path.join(opt.output_dir, '%sA' % opt.phase)
    output_dirB = os.path.join(opt.output_dir, '%sB' % opt.phase)
    os.makedirs(output_dirA, exist_ok=True)
    os.makedirs(output_dirB, exist_ok=True)
    files = os.listdir(input_dir)
    pool = multiprocessing.Pool(processes=opt.num_workers)
    results = []
    for file in tqdm(files):
        assert isinstance(file, str)
        if file.endswith('.png') or file.endswith('.jpg'):
            input_path = os.path.join(input_dir, file)
            pathA = os.path.join(output_dirA, file.replace('_AB', ''))
            pathB = os.path.join(output_dirB, file.replace('_AB', ''))
            # work(input_path, pathA, pathB)
            results.append(pool.apply_async(work, args=(input_path, pathA, pathB)))
    for result in tqdm(results):
        assert isinstance(result, multiprocessing.pool.AsyncResult)
        result.wait()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_dir', type=str, required=True,
                        help='the input directory of the aligned data, e.g., database/edges2shoes/')
    parser.add_argument('--output_dir', type=str, required=True,
                        help='the output directory, e.g., database/edges2shoes-unaligned')
    parser.add_argument('--phase', type=str, default='train',
                        choices=['train', 'val', 'test'], help='the data split')
    parser.add_argument('--num_workers', type=int, default=16)
    opt = parser.parse_args()
    main(opt)
