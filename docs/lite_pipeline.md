# GAN Compression Lite Training Tutorial
## Prerequisites

* Linux
* Python 3
* CPU or NVIDIA GPU + CUDA CuDNN

## Getting Started

### Preparations

Please refer to our [README](../README.md) for the installation, dataset preparations, and the evaluation (FID and mIoU).

### Pipeline

Below we show a lite pipeline for compressing pix2pix and cycleGAN models. **We provide pre-trained models after each step. You could use the pretrained models to skip some steps.** For more training details, please refer to our codes.

## Pix2pix Model Compression

We will show the whole pipeline on `edges2shoes-r` dataset. You could change the dataset name to other datasets (`map2sat` and `cityscapes`).

##### Train a MobileNet Teacher Model (The same as the full pipeline)

Train a MobileNet-style teacher model from scratch.
```shell
bash scripts/pix2pix/edges2shoes-r_lite/train_mobile.sh
```
We provide a pre-trained teacher for each dataset. You could download the pre-trained model by
```shell
python scripts/download_model.py --model pix2pix --task edges2shoes-r_lite --stage mobile
```

and test the model by

```shell
bash scripts/pix2pix/edges2shoes-r_lite/test_mobile.sh
```

##### "Once-for-all" Network Training

Train a "once-for-all" network from a pre-trained student model to search for the efficient architectures.

```shell
bash scripts/pix2pix/edges2shoes-r_lite/train_supernet.sh
```

We provide a trained once-for-all network for each dataset. You could download the model by

```shell
python scripts/download_model.py --model pix2pix --task edges2shoes-r_lite --stage supernet
```

##### Select the Best Model

Evaluate all the candidate sub-networks given a specific configuration

```shell
bash scripts/pix2pix/edges2shoes-r_lite/search.sh
```

The search result will be stored in the python `pickle` form. The pickle file is a python `list` object that stores all the candidate sub-networks information, whose element is a python `dict ` object in the form of

```
{'config_str': $config_str, 'macs': $macs, 'fid'/'mIoU': $fid_or_mIoU}
```

such as

```python
{'config_str': '32_32_48_40_64_40_16_32', 'macs': 5761662976, 'fid': 30.594936138634836}
```

`'config_str'` is a channel configuration description to identify a specific subnet within the "once-for-all" network.

To accelerate the search process, you may need to want to search the sub-networks on multiple GPUs. You could manually split the search space with [search.py](../search.py). All you need to do is add additional arguments `--split` and `--remainder`. For example, if you need to search the sub-networks  with 2 GPUs, you could use the following commands:

* On the first GPU:

  ```bash
  python search.py --dataroot database/edges2shoes-r \
    --restore_G_path logs/pix2pix/edges2shoes-r_lite/supernet-stage2/checkpoints/latest_net_G.pth \
    --output_path logs/pix2pix/edges2shoes-r_lite/supernet-stage2/pkls/result0.pkl \
    --ngf 64 --batch_size 32 \
    --config_set channels-64-pix2pix \
    --real_stat_path real_stat/edges2shoes-r_B.npz --load_in_memory --budget 6.5e9 \
    --split 2 --remainder 0
  ```

* On the second GPU:

  ```bash
  python search.py --dataroot database/edges2shoes-r \
    --restore_G_path logs/pix2pix/edges2shoes-r_lite/supernet-stage2/checkpoints/latest_net_G.pth \
    --output_path logs/pix2pix/edges2shoes-r_lite/supernet-stage2/pkls/result1.pkl \
    --ngf 64 --batch_size 32 \
    --config_set channels-64-pix2pix \
    --real_stat_path real_stat/edges2shoes-r_B.npz --load_in_memory --budget 6.5e9 \
    --split 2 --remainder 1 --gpu_ids 1
  ```

Then you could merge the search results with [merge.py](../merge.py)

```bash
python merge.py --input_dir logs/pix2pix/edges2shoes-r_lite/supernet-stage2/pkls \
  --output_path logs/cycle_gan/horse2zebra/supernet
```

Once you get the search results, you could use our auxiliary script [select_arch.py](../select_arch.py) to select the architecture you want.

```shell
python select_arch.py --macs 6.5e9 --fid 32 \ 
  --pkl_path logs/pix2pix/edges2shoes-r/supernet/result.pkl
```

##### Fine-tuning the Best Model

(Optional) Fine-tune a specific subnet within the pre-trained "once-for-all" network. To further improve the performance of your chosen subnet, you may need to fine-tune the subnet. For example, if you want to fine-tune a subnet within the "once-for-all" network with `'config_str': 32_32_48_40_64_40_16_32`, use the following command:

```shell
bash scripts/pix2pix/edges2shoes-r_lite/finetune.sh 32_32_48_40_64_40_16_32
```

##### Export the Model

Extract a subnet from the "once-for-all" network. We provide a code [export.py](../export.py) to extract a specific subnet according to a configuration description. For example, if the `config_str` of your chosen subnet is `32_32_48_32_48_48_16_16`, then you can export the model by this command:

```shell
bash scripts/pix2pix/edges2shoes-r_lite/export.sh 32_32_48_40_64_40_16_32
```

## CycleGAN Model Compression

The pipeline is almost identical to pix2pix. We will show the pipeline on `horse2zebra` dataset.

##### Train a MobileNet Teacher Model

Train a MobileNet-style teacher model from scratch.

```shell
bash scripts/cycle_gan/horse2zebra_lite/train_mobile.sh
```

We provide a pre-trained teacher model for each dataset. You could download the model using

```shell
python scripts/download_model.py --model cycle_gan --task horse2zebra_lite --stage mobile
```

and test the model by

```shell
bash scripts/cycle_gan/horse2zebra_lite/test_mobile.sh
```

##### "Once-for-all" Network Training

Train a "once-for-all" network from a pre-trained student model to search for the efficient architectures.

```shell
bash scripts/cycle_gan/horse2zebra_lite/train_supernet.sh
```

We provide a pre-trained once-for-all network for each dataset. You could download the model by

```shell
python scripts/download_model.py --model cycle_gan --task horse2zebra_lite --stage supernet
```

##### Select the Best Model

Evaluate all the candidate sub-networks given a specific configuration

```shell
bash scripts/cycle_gan/horse2zebra_lite/search.sh
```
To support multi-GPU search, you could manually split the search space with additional arguments `--split` and `--remainder` and merge them with [merge.py](../merge.py), which is the same as pix2pix.

You could also use our auxiliary script [select_arch.py](../select_arch.py) to select the architecture you want. The usage is the same as pix2pix.

##### Fine-tuning the Best Model

During our experiments, we observe that fine-tuning the model on horse2zebra increases FID.  **You may skip the fine-tuning.**

##### Export the Model

Extract a subnet from the supernet. We provide a code [export.py](../export.py) to extract a specific subnet according to a configuration description. For example, if the `config_str` of your chosen subnet is `24_16_32_16_32_64_16_24`, then you can export the model by this command:

```shell
bash scripts/cycle_gan/horse2zebra_lite/export.sh 24_16_32_16_32_64_16_24
```
