import torch

from metric.fid_score import _compute_statistics_of_ims, calculate_frechet_distance
from utils import util
from .mAP_score import test


def get_fid(fakes, model, npz, device, batch_size=1, use_tqdm=True):
    m1, s1 = npz['mu'], npz['sigma']
    fakes = torch.cat(fakes, dim=0)
    fakes = util.tensor2im(fakes).astype(float)
    m2, s2 = _compute_statistics_of_ims(fakes, model, batch_size, 2048,
                                        device, use_tqdm=use_tqdm)
    return float(calculate_frechet_distance(m1, s1, m2, s2))


def get_mAP(fakes, names, model, device,
            table_path='datasets/table.txt',
            data_dir='database/cityscapes',
            batch_size=1, num_workers=8, num_classes=19,
            use_tqdm=True):
    fakes = torch.cat(fakes, dim=0)
    fakes = util.tensor2im(fakes)
    mAP = test(fakes, names, model, device, table_path=table_path, data_dir=data_dir,
               batch_size=batch_size, num_workers=num_workers, num_classes=num_classes, use_tqdm=use_tqdm)
    return float(mAP)
