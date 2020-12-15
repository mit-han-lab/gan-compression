import os

import cv2
import numpy as np
import torch
from PIL import Image
from torch.nn import functional as F
from torch.utils.data import Dataset, DataLoader


class CocoStuff164k(Dataset):

    def __init__(self, root, images, names):
        self.root = root
        self.ignore_label = 255
        self.mean_bgr = np.array((104.008, 116.669, 122.675))
        self.label_paths = []
        self.images = images
        self.names = names
        self._set_files()
        cv2.setNumThreads(0)

    def _set_files(self):
        label_paths = []
        for name in self.names:
            path = os.path.join(self.root, 'val_label', '%s.png' % name)
            assert os.path.exists(path)
            label_paths.append(path)
        self.label_paths = label_paths

    def _load_data(self, index):
        # Set paths
        image_id = self.names[index]
        label_path = self.label_paths[index]
        # Load an image and label
        image = self.images[index]
        label = cv2.imread(label_path, cv2.IMREAD_GRAYSCALE)
        return image_id, image, label

    def __getitem__(self, index):
        image_id, image, label = self._load_data(index)
        h, w = label.shape
        image_pil = Image.fromarray(image)
        if image_pil.size[0] != w or image_pil.size[1] != h:
            image_pil = image_pil.resize((w, h), Image.BICUBIC)
        image = np.asarray(image_pil)
        image = np.flip(image, axis=2)
        # Mean subtraction
        image = image - self.mean_bgr
        # HWC -> CHW
        image = image.transpose(2, 0, 1)
        return image_id, image.astype(np.float32), label.astype(np.int64)

    def __len__(self):
        return len(self.names)


def _fast_hist(label_true, label_pred, n_class):
    mask = (label_true >= 0) & (label_true < n_class)
    hist = np.bincount(
        n_class * label_true[mask].astype(int) + label_pred[mask],
        minlength=n_class ** 2,
    ).reshape(n_class, n_class)
    return hist


def compute_scores(label_trues, label_preds, n_class):
    hist = np.zeros((n_class, n_class))
    for lt, lp in zip(label_trues, label_preds):
        hist += _fast_hist(lt.flatten(), lp.flatten(), n_class)
    accu = np.diag(hist).sum() / hist.sum()
    iu = np.diag(hist) / (hist.sum(axis=1) + hist.sum(axis=0) - np.diag(hist))
    valid = hist.sum(axis=1) > 0  # added
    mIoU = np.nanmean(iu[valid])
    return accu * 100, mIoU * 100


def test(fakes, names, model, device, data_dir, batch_size=1, num_workers=0, tqdm_position=None):
    dataset = CocoStuff164k(data_dir, fakes, names)
    dataloader = DataLoader(dataset=dataset, batch_size=batch_size,
                            num_workers=num_workers, shuffle=False)
    preds, gts = [], []
    if tqdm_position is None or tqdm_position >= 0:
        from tqdm import tqdm
        dataloader_tqdm = tqdm(dataloader, desc='Coco Scores', position=tqdm_position, leave=False)
    else:
        dataloader_tqdm = dataloader
    with torch.no_grad():
        for image_ids, images, gt_labels in dataloader_tqdm:
            images = images.to(device)
            logits = model(images)
            _, H, W = gt_labels.shape
            if logits.shape[-2] != H or logits.shape[-1] != W:
                logits = F.interpolate(
                    logits, size=(H, W), mode="bilinear", align_corners=False
                )
            probs = F.softmax(logits, dim=1)
            labels = torch.argmax(probs, dim=1)
            preds += list(labels.cpu().numpy())
            gts += list(gt_labels.numpy())

    return compute_scores(gts, preds, n_class=182)
