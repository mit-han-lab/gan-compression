"""
Copyright (C) 2019 NVIDIA Corporation.  All rights reserved.
Licensed under the CC BY-NC-SA 4.0 license (https://creativecommons.org/licenses/by-nc-sa/4.0/legalcode).
"""

import numpy as np
import torch
import tvm
from PyQt5.QtGui import QImage, QPixmap
from torch import nn


def load_network(net, load_path):
    print('Load network at %s' % load_path)
    weights = torch.load(load_path)
    if isinstance(net, nn.DataParallel):
        net = net.module
    net.load_state_dict(weights)
    return net


def check(tensor):
    tensor = tensor[0]
    c, h, w = tensor.size()
    for i in range(h):
        for j in range(w):
            if not (tensor[:, i, j].sum() == 255 * 3 or tensor[:, i, j].sum() == 0):
                return False
    return True


def pixmap2tensor(pixmap):
    nchannels = 3
    # pixmap = reshape_qim(pixmap, MODEL_DIMENSIONS)
    image = pixmap.toImage()
    b = image.constBits()
    w, h = pixmap.width(), pixmap.height()
    b.setsize(h * w * 4)
    arr = np.frombuffer(b, np.uint8).reshape((h, w, 4)).astype(np.float32)  # [h, w, c]
    arr = arr[:, :, :nchannels]
    tensor = torch.tensor(arr).permute([2, 0, 1])  # [h, w, c]
    tensor = tensor.unsqueeze(0)  # [1, c, h, w]
    tensor = (tensor / 255 - 0.5) * 2
    return tensor


def pixmap2tvm(pixmap):
    nchannels = 3
    # pixmap = reshape_qim(pixmap, MODEL_DIMENSIONS)
    image = pixmap.toImage()
    b = image.constBits()
    w, h = pixmap.width(), pixmap.height()
    b.setsize(h * w * 4)
    arr = np.frombuffer(b, np.uint8).reshape((h, w, 4)).astype(np.float32)  # [h, w, c]
    arr = arr[:, :, :nchannels]
    arr = np.transpose(arr, [2, 0, 1])
    arr = np.expand_dims(arr, axis=0)
    arr = (arr / 255 - 0.5) * 2
    return tvm.nd.array(arr, device=tvm.cuda())


def tensor2pixmap(tensor):  # [1, c, h, w]
    tensor = (tensor / 2 + 0.5) * 255
    tensor = tensor[0].permute([1, 2, 0])  # [h, w, c]
    arr = tensor.numpy().astype(np.uint32)
    h, w, c = arr.shape
    b = (255 << 24 | arr[:, :, 0] << 16 | arr[:, :, 1] << 8 | arr[:, :, 2]).flatten()
    im = QImage(b, w, h, QImage.Format_RGB32)
    # im = reshape_qim(im, CANVAS_DIMENSIONS)
    return QPixmap.fromImage(im)


def tvm2pixmap(arr):
    arr = arr.asnumpy()
    arr = (arr / 2 + 0.5) * 255
    arr = np.transpose(arr[0], [1, 2, 0]).astype(np.uint32)
    h, w, c = arr.shape
    b = (255 << 24 | arr[:, :, 0] << 16 | arr[:, :, 1] << 8 | arr[:, :, 2]).flatten()
    im = QImage(b, w, h, QImage.Format_RGB32)
    # im = reshape_qim(im, CANVAS_DIMENSIONS)
    return QPixmap.fromImage(im)
