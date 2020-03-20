import glob
import os

from PIL import Image
from tqdm import tqdm

help_msg = """
The dataset can be downloaded from https://cityscapes-dataset.com.
Please download the datasets [gtFine_trainvaltest.zip] and [leftImg8bit_trainvaltest.zip] and unzip them.
gtFine contains the semantics segmentations. Use --gtFine_dir to specify the path to the unzipped gtFine_trainvaltest directory.
leftImg8bit contains the dashcam photographs. Use --leftImg8bit_dir to specify the path to the unzipped leftImg8bit_trainvaltest directory.
The processed images will be placed at --output_dir.

Example usage:

python prepare_cityscapes_dataset.py --gitFine_dir ./gtFine/ --leftImg8bit_dir ./leftImg8bit --output_dir ./datasets/cityscapes/
"""


def load_resized_img(path):
    return Image.open(path).convert('RGB').resize((256, 256))


def check_matching_pair(segmap_path, photo_path):
    segmap_identifier = os.path.basename(segmap_path).replace('_gtFine_color', '')
    photo_identifier = os.path.basename(photo_path).replace('_leftImg8bit', '')

    assert segmap_identifier == photo_identifier, \
        "[%s] and [%s] don't seem to be matching. Aborting." % (segmap_path, photo_path)


def process_cityscapes(gtFine_dir, leftImg8bit_dir, output_dir, phase, table_path=None):
    save_phase = 'test' if phase == 'val' else 'train'
    savedir = os.path.join(output_dir, save_phase)
    os.makedirs(savedir, exist_ok=True)
    os.makedirs(savedir + 'A', exist_ok=True)
    os.makedirs(savedir + 'B', exist_ok=True)
    print("Directory structure prepared at %s" % output_dir)

    segmap_expr = os.path.join(gtFine_dir, phase) + "/*/*_color.png"
    segmap_paths = glob.glob(segmap_expr)
    segmap_paths = sorted(segmap_paths)

    photo_expr = os.path.join(leftImg8bit_dir, phase) + "/*/*_leftImg8bit.png"
    photo_paths = glob.glob(photo_expr)
    photo_paths = sorted(photo_paths)

    assert len(segmap_paths) == len(photo_paths), \
        "%d images that match [%s], and %d images that match [%s]. Aborting." % (
            len(segmap_paths), segmap_expr, len(photo_paths), photo_expr)

    if table_path is not None:
        f = open(table_path, 'w')
    else:
        f = None

    for i, (segmap_path, photo_path) in enumerate(tqdm(zip(segmap_paths, photo_paths))):
        check_matching_pair(segmap_path, photo_path)
        segmap = load_resized_img(segmap_path)
        photo = load_resized_img(photo_path)

        # data for pix2pix where the two images are placed side-by-side
        sidebyside = Image.new('RGB', (512, 256))
        sidebyside.paste(segmap, (256, 0))
        sidebyside.paste(photo, (0, 0))
        savepath = os.path.join(savedir, "%d.jpg" % i)
        sidebyside.save(savepath, format='JPEG', subsampling=0, quality=100)

        # data for cycle_gan where the two images are stored at two distinct directories
        savepath = os.path.join(savedir + 'A', "%d_A.jpg" % i)
        photo.save(savepath, format='JPEG', subsampling=0, quality=100)
        savepath = os.path.join(savedir + 'B', "%d_B.jpg" % i)
        segmap.save(savepath, format='JPEG', subsampling=0, quality=100)

        if f is not None:
            rel_segmap_path = os.path.relpath(segmap_path, os.path.dirname(os.path.abspath(gtFine_dir)))
            rel_photo_path = os.path.relpath(photo_path, os.path.dirname(os.path.abspath(leftImg8bit_dir)))
            f.write('%d %s %s\n' % (i, rel_segmap_path.replace('_color', '_trainIds'), rel_photo_path))


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('--gtFine_dir', type=str, required=True,
                        help='Path to the Cityscapes gtFine directory.')
    parser.add_argument('--leftImg8bit_dir', type=str, required=True,
                        help='Path to the Cityscapes leftImg8bit_trainvaltest directory.')
    parser.add_argument('--output_dir', type=str, required=True,
                        default='database/cityscapes-origin',
                        help='Directory the output images will be written to.')
    parser.add_argument('--table_path', type=str, default='datasets/table.txt',
                        help='Generate a mapping table to map the generated images to the original images. '
                             'The table will be used for mAP computation.')
    opt = parser.parse_args()

    print(help_msg)

    print('Preparing Cityscapes Dataset for val phase')
    process_cityscapes(opt.gtFine_dir, opt.leftImg8bit_dir,
                       opt.output_dir, "val", table_path=opt.table_path)
    print('Preparing Cityscapes Dataset for train phase')
    process_cityscapes(opt.gtFine_dir, opt.leftImg8bit_dir,
                       opt.output_dir, "train")

    print('Done')
