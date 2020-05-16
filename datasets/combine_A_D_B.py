import argparse
import os

import cv2
import numpy as np

parser = argparse.ArgumentParser('create image triplet')
parser.add_argument('--fold_A', dest='fold_A', help='input directory for image A', type=str,
                    default='./database/fomm/source')
parser.add_argument('--fold_D', dest='fold_D', help='input directory for image D', type=str,
                    default='./database/fomm/drive')
parser.add_argument('--fold_B', dest='fold_B', help='input directory for image B', type=str,
                    default='./database/fomm/predict')
parser.add_argument('--fold_ADB', dest='fold_ADB', help='output directory', type=str, default='./database/fomm/')
parser.add_argument('--num_imgs', dest='num_imgs', help='number of images', type=int, default=1000000)
parser.add_argument('--use_ADB', dest='use_ADB', help='if true: (0001_A, 0001_D, 0001_B) to (0001_ADB)', action='store_true')
parser.add_argument('--splits', dest='use_splits', help='if true: should have subfolders train and test', action='store_true')
args = parser.parse_args()

for arg in vars(args):
    print('[%s] = ' % arg, getattr(args, arg))

splits = os.listdir(args.fold_A) if args.use_splits else ['all'] # ex. [train, test, val]

for sp in splits:
    img_fold_A = os.path.join(args.fold_A, sp) if args.use_splits else args.fold_A
    img_fold_D = os.path.join(args.fold_D, sp) if args.use_splits else args.fold_D
    img_fold_B = os.path.join(args.fold_B, sp) if args.use_splits else args.fold_B
    img_list = os.listdir(img_fold_A)
    if args.use_ADB:
        img_list = [img_path for img_path in img_list if '_A.' in img_path]

    num_imgs = min(args.num_imgs, len(img_list))
    print('split = %s, use %d/%d images' % (sp, num_imgs, len(img_list)))
    img_fold_ADB = os.path.join(args.fold_ADB, sp)
    if not os.path.isdir(img_fold_ADB):
        os.makedirs(img_fold_ADB)
    print('split = %s, number of images = %d' % (sp, num_imgs))
    for n in tqdm(range(num_imgs), decs=sp):
        name_A = img_list[n]
        path_A = os.path.join(img_fold_A, name_A)
        if args.use_ADB:
            name_D = name_D.replace('_A.', '_D.')
            name_B = name_A.replace('_A.', '_B.')
        else:
            name_D = name_A
            name_B = name_A
        path_D = os.path.join(img_fold_D, name_D)    
        path_B = os.path.join(img_fold_B, name_B)
        if os.path.isfile(path_A) and os.path.isfile(path_B) and os.path.isfile(path_D):
            name_ADB = name_A
            if args.use_ADB:
                name_ADB = name_AB.replace('_A.', '.')  # remove _A
            path_ADB = os.path.join(img_fold_ADB, name_ADB)
            im_A = cv2.imread(path_A, 1)  # python2: cv2.CV_LOAD_IMAGE_COLOR; python3: cv2.IMREAD_COLOR
            im_D = cv2.imread(path_D, 1)  # python2: cv2.CV_LOAD_IMAGE_COLOR; python3: cv2.IMREAD_COLOR
            im_B = cv2.imread(path_B, 1)  # python2: cv2.CV_LOAD_IMAGE_COLOR; python3: cv2.IMREAD_COLOR
            im_ADB = np.concatenate([im_A, im_D, im_B], 1)
            cv2.imwrite(path_ADB, im_ADB)
