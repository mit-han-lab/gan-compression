import argparse
import os
import shutil

import numpy as np
from tqdm import tqdm
from sklearn.model_selection import train_test_split

parser = argparse.ArgumentParser('split datasets into train and test')
parser.add_argument('--fold_src', dest='fold_src', help='input directory', type=str, default='./database/fomm/all')
parser.add_argument('--fold_dst', dest='fold_dst', help='output directory', type=str, default='./database/fomm/')
parser.add_argument('--use_copy', dest='copy', help='Use copy intead move mode', action='store_true')
parser.add_argument('--test_size', dest='test_size', help='test size', type=float, default=0.1)
parser.add_argument('--val_size', dest='val_size', help='val size', type=float, default=0.1)
parser.add_argument('--random_state', dest='random_state', help='random state', type=float, default=None)
parser.add_argument('--no_shuffle', dest='shuffle', help='shuffle data before splitting', action='store_false')
parser.set_defaults(shuffle=True, copy=False)
args = parser.parse_args()

for arg in vars(args):
    print('[%s] = ' % arg, getattr(args, arg))


img_list = os.listdir(args.fold_src)
print(f'DATABASE SIZE: {len(img_list)}')


train_img_list, test_img_list = train_test_split(img_list,
                                                 test_size=args.test_size,
                                                 random_state=args.random_state,
                                                 shuffle=args.shuffle)

train_img_list, val_img_list = train_test_split(train_img_list,
                                                test_size=args.val_size / (1 - args.test_size),
                                                random_state=args.random_state,
                                                shuffle=args.shuffle)


splits = [
    ('train', train_img_list),
    ('test', test_img_list),
    ('val', val_img_list),
]

for sp, sp_img_list in splits:
    
    img_fold_src = args.fold_src 
    img_fold_dst = os.path.join(args.fold_dst, sp)
    
    if not os.path.exists(img_fold_dst):
        os.mkdir(img_fold_dst)
    
    for img_name in tqdm(sp_img_list):
        path_src = os.path.join(img_fold_src, img_name)
        path_dst = os.path.join(img_fold_dst, img_name)
        
        if os.path.isfile(path_src):
            if args.copy:
                shutil.copy2(path_src, path_dst)
            else:
                shutil.move(path_src, path_dst)