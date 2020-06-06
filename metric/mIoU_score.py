import math
import os

import numpy as np
import torch
import torch.nn as nn
from PIL import Image
from torch.utils.data import Dataset, DataLoader

from . import drn


# from tqdm import tqdm


class Resize(object):
    def __init__(self, size):
        self.size = size

    def __call__(self, image_numpy, label):
        image = Image.fromarray(image_numpy)
        interpolation = Image.CUBIC
        w, h = image.size
        tw, th = self.size
        if w != tw or h != th:
            image = image.resize((tw, th), interpolation)

        return image, label


class Normalize(object):
    """Given mean: (R, G, B) and std: (R, G, B),
    will normalize each channel of the torch.*Tensor, i.e.
    channel = (channel - mean) / std
    """

    def __init__(self, mean, std):
        self.mean = torch.FloatTensor(mean)
        self.std = torch.FloatTensor(std)

    def __call__(self, image, label):
        for t, m, s in zip(image, self.mean, self.std):
            t.sub_(m).div_(s)
        return image, label


class ToTensor(object):
    """Converts a PIL.Image or numpy.ndarray (H x W x C) in the range
    [0, 255] to a torch.FloatTensor of shape (C x H x W) in the range [0.0, 1.0].
    """

    def __call__(self, pic, label):
        img = torch.ByteTensor(torch.ByteStorage.from_buffer(pic.tobytes()))
        # PIL image mode: 1, L, P, I, F, RGB, YCbCr, RGBA, CMYK
        if pic.mode == 'YCbCr':
            nchannel = 3
        else:
            nchannel = len(pic.mode)
        img = img.view(pic.size[1], pic.size[0], nchannel)
        img = img.transpose(0, 1).transpose(0, 2).contiguous()
        img = img.float() / 255
        return img, torch.from_numpy(np.array(label, dtype=np.int))


class Compose(object):
    """Composes several transforms together.
    """

    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, *args):
        for t in self.transforms:
            args = t(*args)
        return args


class SegList(Dataset):

    def __init__(self, images, names, table_path, data_dir):
        self.images = images
        self.names = names
        self.table_path = table_path
        self.data_dir = data_dir
        self.transforms = Compose([
            Resize([2048, 1024]),
            ToTensor(),
            Normalize(mean=[0.29010095242892997, 0.32808144844279574, 0.28696394422942517],
                      std=[0.1829540508368939, 0.18656561047509476, 0.18447508988480435])
        ])
        self.read_lists()

    def __getitem__(self, index):
        data = [self.images[index], Image.open(os.path.join(self.data_dir, self.label_list[index]))]
        data = list(self.transforms(*data))
        return tuple(data)

    def __len__(self):
        return len(self.names)

    def read_lists(self):
        self.label_list = []
        table = []
        with open(self.table_path, 'r') as f:
            lines = f.readlines()
            for line in lines:
                table.append(line.strip().split(' '))
        for name in self.names:
            for item in table:
                if item[0] == name or item[2][:-len('.png')].endswith(name):
                    self.label_list.append(item[1])
                    break
        assert len(self.label_list) == len(self.names)


def per_class_iu(hist):
    return np.diag(hist) / (hist.sum(1) + hist.sum(0) - np.diag(hist))


def fill_up_weights(up):
    w = up.weight.data
    f = math.ceil(w.size(2) / 2)
    c = (2 * f - 1 - f % 2) / (2. * f)
    for i in range(w.size(2)):
        for j in range(w.size(3)):
            w[0, 0, i, j] = \
                (1 - math.fabs(i / f - c)) * (1 - math.fabs(j / f - c))
    for c in range(1, w.size(0)):
        w[c, 0, :, :] = w[0, 0, :, :]


class DRNSeg(nn.Module):
    def __init__(self, model_name, classes, pretrained_model=None,
                 pretrained=True, use_torch_up=False):
        super(DRNSeg, self).__init__()
        model = drn.__dict__.get(model_name)(
            pretrained=pretrained, num_classes=1000)

        pmodel = nn.DataParallel(model)
        if pretrained_model is not None:
            pmodel.load_state_dict(pretrained_model)
        self.base = nn.Sequential(*list(model.children())[:-2])

        self.seg = nn.Conv2d(model.out_dim, classes,
                             kernel_size=1, bias=True)
        self.softmax = nn.LogSoftmax()
        m = self.seg
        n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
        m.weight.data.normal_(0, math.sqrt(2. / n))
        m.bias.data.zero_()
        if use_torch_up:
            self.up = nn.UpsamplingBilinear2d(scale_factor=8)
        else:
            up = nn.ConvTranspose2d(classes, classes, 16, stride=8, padding=4,
                                    output_padding=0, groups=classes,
                                    bias=False)
            fill_up_weights(up)
            up.weight.requires_grad = False
            self.up = up

    def forward(self, x):
        x = self.base(x)
        x = self.seg(x)
        y = self.up(x)
        return self.softmax(y), x

    def optim_parameters(self, memo=None):
        raise NotImplementedError('This code is just for evaluation!!!')


def fast_hist(pred, label, n):
    k = (label >= 0) & (label < n)
    return np.bincount(
        n * label[k].astype(int) + pred[k], minlength=n ** 2).reshape(n, n)


def test(fakes, names, model, device, table_path='datasets/table.txt', data_dir='database/cityscapes',
         batch_size=1, num_workers=8, num_classes=19, use_tqdm=True):
    dataset = SegList(fakes, names, table_path, data_dir)
    eval_dataloader = DataLoader(dataset, batch_size=batch_size,
                                 shuffle=False, num_workers=num_workers)
    model.eval()
    hist = np.zeros((num_classes, num_classes))
    if use_tqdm:
        from tqdm import tqdm
    else:
        def tqdm(x):
            return x
    with torch.no_grad():
        for iter, (image, label) in enumerate(tqdm(eval_dataloader)):
            image = image.to(device)
            final = model(image)[0]
            _, pred = torch.max(final, 1)
            pred = pred.cpu().numpy()
            label = label.numpy()
            hist += fast_hist(pred.flatten(), label.flatten(), num_classes)

    ious = per_class_iu(hist) * 100
    return round(np.nanmean(ious), 2)
