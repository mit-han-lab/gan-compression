"""A modified image folder class

We modify the official PyTorch image folder (https://github.com/pytorch/vision/blob/master/torchvision/datasets/folder.py)
so that this class can load images from both current directory and its subdirectories.
"""

import os
import os.path

from PIL import Image

IMG_EXTENSIONS = [
    '.jpg', '.JPG', '.jpeg', '.JPEG',
    '.png', '.PNG', '.ppm', '.PPM', '.bmp', '.BMP',
]


def is_image_file(filename):
    return any(filename.endswith(extension) for extension in IMG_EXTENSIONS)


def make_dataset_rec(dir, images):
    assert os.path.isdir(dir), '%s is not a valid directory' % dir

    for root, dnames, fnames in sorted(os.walk(dir, followlinks=True)):
        for fname in fnames:
            if is_image_file(fname):
                path = os.path.join(root, fname)
                images.append(path)


def make_dataset(dir, max_dataset_size=float("inf"), recursive=False, meta_path=None, write_cache=False):
    images = []

    if meta_path is not None:
        with open(meta_path, 'r') as f:
            lines = f.readlines()
        from tqdm import tqdm
        for line in tqdm(lines):
            line = line.strip()
            if len(line) > 0 and is_image_file(line):
                images.append(os.path.join(dir, line))
    else:
        if recursive:
            make_dataset_rec(dir, images)
        else:
            assert os.path.isdir(dir) or os.path.islink(dir), '%s is not a valid directory' % dir

            for root, dnames, fnames in sorted(os.walk(dir)):
                for fname in fnames:
                    if is_image_file(fname):
                        path = os.path.join(root, fname)
                        images.append(path)

    if max_dataset_size == -1:
        return images
    else:
        return images[:min(max_dataset_size, len(images))]


def default_loader(path):
    return Image.open(path).convert('RGB')
