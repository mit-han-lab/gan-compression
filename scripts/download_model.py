import argparse
import os

import wget


def check(opt):
    if opt.model == "pix2pix":
        assert opt.task in [
            "edges2shoes-r",
            "map2sat",
            "cityscapes",
            "cityscapes_fast",
            "edges2shoes-r_fast",
            "map2sat_fast",
        ]
    elif opt.model == "cycle_gan":
        assert opt.task in ["horse2zebra", "horse2zebra_fast"]
    elif opt.model == "gaugan":
        assert opt.task in ["cityscapes", "cityscapes_fast", "coco_fast"]
    elif opt.model == "munit":
        assert opt.task in ["edges2shoes-r_fast"]
    else:
        raise NotImplementedError("Unsupported model [%s]!" % opt.model)


def download(path):
    url = "https://huggingface.co/mit-han-lab/gan-compression/resolve/main/" + path
    dir = os.path.dirname(path)
    os.makedirs(dir, exist_ok=True)
    wget.download(url, path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Download a pretrained model.")
    parser.add_argument(
        "--stage",
        type=str,
        default="compressed",
        choices=["full", "mobile", "distill", "supernet", "finetune", "compressed", "legacy"],
        help="specify the stage you want to download",
    )
    parser.add_argument(
        "--model",
        type=str,
        default="pix2pix",
        choices=["pix2pix", "cycle_gan", "gaugan", "munit"],
        help="specify the model you want to download",
    )
    parser.add_argument("--task", type=str, default="horse2zebra", help="the base number of filters of the generator")
    opt = parser.parse_args()
    check(opt)
    path = os.path.join("pretrained", opt.model, opt.task, opt.stage, "latest_net_G.pth")
    download(path)
    if opt.stage != "compressed" and opt.stage != "legacy":
        path = os.path.join("pretrained", opt.model, opt.task, opt.stage, "latest_net_D.pth")
        download(path)
