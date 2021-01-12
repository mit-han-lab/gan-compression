# Fast GAN Compression Training Tutorial
## Prerequisites

* Linux
* Python 3
* CPU or NVIDIA GPU + CUDA CuDNN

## Getting Started

### Preparations

Please refer to our [README](../../README.md) for the installation, dataset preparations, and the evaluation (FID and mIoU).

### Pipeline

Below we show the pipeline of Fast GAN Compression to compress pix2pix and cycleGAN models. **We provide pre-trained models after each step. You could use the pretrained models to skip some steps.** For more training details, please refer to our codes. For more details about the differences between [GAN Compression](gan_compression.md) and Fast GAN Compression, please refer to [Section 4.1 Pipelines](https://arxiv.org/abs/2003.08936) of our paper.

## Pix2pix Model Compression

We will show the whole pipeline on `edges2shoes-r` dataset. You could change the dataset name to other datasets (such as `map2sat`).

##### Train an Original Full Teacher Model (if you already have the full model, you could skip it)

Train an original full teacher model from scratch.
```shell
bash scripts/pix2pix/edges2shoes-r/train_full.sh
```
We provide a pre-trained teacher for each dataset. You could download the pre-trained model by
```shell
python scripts/download_model.py --model pix2pix --task edges2shoes-r --stage full
```

and test the model by

```shell
bash scripts/pix2pix/edges2shoes-r/test_full.sh
```

##### "Once-for-all" Network Training

Train a "once-for-all" network from scratch to search for the efficient architectures.

```shell
bash scripts/pix2pix/edges2shoes-r_fast/train_supernet.sh
```

We provide a trained once-for-all network for each dataset. You could download the model by

```shell
python scripts/download_model.py --model pix2pix --task edges2shoes-r_fast --stage supernet
```

##### Select the Best Model

The evolution searching uses the evolution algorithm to search for the best-performed subnet. It is much much faster than the brute force searching. You could run:

```bash
bash scripts/pix2pix/edges2shoes-r_fast/evolution_search.sh
```

It will directly tells you the information of the best-performed subnet which satisfies the computation budget in the following format:

```
{config_str: $config_str, macs: $macs, fid/mIoU: $fid_or_mIoU}
```

##### Fine-tuning the Best Model

(Optional) Fine-tune a specific subnet within the pre-trained "once-for-all" network. To further improve the performance of your chosen subnet, you may need to fine-tune the subnet. For example, if you want to fine-tune a subnet within the "once-for-all" network with `'config_str': 32_32_40_40_40_64_16_16`, use the following command:

```shell
bash scripts/pix2pix/edges2shoes-r_fast/finetune.sh 32_32_48_40_64_40_16_32
```

During our experiments, we observe that fine-tuning the model on cityscapes doesn't increase mIoU. **You may skip the fine-tuning on cityscapes.**

##### Export the Model

Extract a subnet from the "once-for-all" network. We provide a code [export.py](../../export.py) to extract a specific subnet according to a configuration description. For example, if the `config_str` of your chosen subnet is `32_32_40_40_40_64_16_16`, then you can export the model by this command:

```shell
bash scripts/pix2pix/edges2shoes-r_fast/export.sh 32_32_40_40_40_64_16_16
```

## CycleGAN Model Compression

The pipeline is almost identical to pix2pix. We will show the pipeline on `horse2zebra` dataset.

##### Train an Original Full Teacher Model (if you already have the full model, you could skip it)

Train an original full teacher model from scratch.

```shell
bash scripts/cycle_gan/horse2zebra/train_full.sh
```

We provide a pre-trained teacher model for each dataset. You could download the model using

```shell
python scripts/download_model.py --model cycle_gan --task horse2zebra --stage full
```

and test the model by

```shell
bash scripts/cycle_gan/horse2zebra/test_full.sh
```

##### "Once-for-all" Network Training

Train a "once-for-all" network from scratch to search for the efficient architectures.

```shell
bash scripts/cycle_gan/horse2zebra_fast/train_supernet.sh
```

We provide a pre-trained once-for-all network for each dataset. You could download the model by

```shell
python scripts/download_model.py --model cycle_gan --task horse2zebra_fast --stage supernet
```

##### Select the Best Model

This stage is almost the same as pix2pix.

```bash
bash scripts/cycle_gan/horse2zebra_fast/evolution_search.sh
```

##### Fine-tuning the Best Model

During our experiments, we observe that fine-tuning the model on horse2zebra increases FID.  **You may skip the fine-tuning.**

##### Export the Model

Extract a subnet from the supernet. We provide a code [export.py](../export.py) to extract a specific subnet according to a configuration description. For example, if the `config_str` of your chosen subnet is `16_16_24_16_32_64_16_24`, then you can export the model by this command:

```shell
bash scripts/cycle_gan/horse2zebra_fast/export.sh 16_16_24_16_32_64_16_24
```

## GauGAN Model Compression

The pipeline is almost identical to pix2pix. We will show the pipeline on `cityscapes` dataset.

##### Train an Original Full Teacher Model (if you already have the full model, you could skip it)

Train an original full teacher model from scratch.

```shell
bash scripts/gaugan/cityscapes/train_full.sh
```

We provide a pre-trained teacher model for each dataset. You could download the model using

```shell
python scripts/download_model.py --model gaugan --task cityscapes --stage full
```

and test the model by

```shell
bash scripts/gaugan/cityscapes/test_full.sh
```

##### "Once-for-all" Network Training

**Note:** If your original full model uses spectral norm, please remove it before the "once-for-all" network training. You could remove it in this way:

```bash
python remove_spectral_norm.py --netG spade \
  --restore_G_path logs/gaugan/cityscapes/full/checkpoints/latest_net_G.pth \
  --output_path logs/gaugan/cityscapes/full/export/latest_net_G.pth
```

Train a "once-for-all" network from scratch to search for the efficient architectures.

```shell
bash scripts/gaugan/cityscapes_fast/train_supernet.sh
```

We provide a pre-trained once-for-all network for each dataset. You could download the model by

```shell
python scripts/download_model.py --model gaugan --task cityscapes_fast --stage supernet
```

##### Select the Best Model

This stage is almost the same as pix2pix.

```bash
bash scripts/gaugan/cityscapes_fast/evolution_search.sh
```

##### Fine-tuning the Best Model

(Optional) Fine-tune a specific subnet within the pre-trained "once-for-all" network. To further improve the performance of your chosen subnet, you may need to fine-tune the subnet. For example, if you want to fine-tune a subnet within the "once-for-all" network with `'config_str': 32_32_32_48_32_24_24_32`, use the following command:

```shell
bash scripts/gaugan/cityscapes_fast/finetune.sh 32_32_32_48_32_24_24_32
```

##### Export the Model

Extract a subnet from the supernet. We provide a code [export.py](../../export.py) to extract a specific subnet according to a configuration description. For example, if the `config_str` of your chosen subnet is `32_32_32_48_32_24_24_32`, then you can export the model by this command:

```shell
bash scripts/gaugan/cityscapes_fast/export.sh 32_32_32_48_32_24_24_32
```

## MUNIT Model Compression

The pipeline is almost identical to pix2pix. We will show the pipeline on `edges2shoes-r-unaligned` dataset.

##### Train an Original Full Teacher Model (if you already have the full model, you could skip it)

Train an original full teacher model from scratch.

```shell
bash scripts/munit/edges2shoes-r_fast/train_full.sh
```

We provide a pre-trained teacher model for each dataset. You could download the model using

```shell
python scripts/download_model.py --model munit --task edges2shoes-r_fast --stage full
```

and test the model by

```shell
bash scripts/munit/edges2shoes-r_fast/test_full.sh
```

##### "Once-for-all" Network Training

Train a "once-for-all" network from scratch to search for the efficient architectures.

```shell
bash scripts/munit/edges2shoes-r_fast/train_supernet.sh
```

We provide a pre-trained once-for-all network for each dataset. You could download the model by

```shell
python scripts/download_model.py --model munit --task edges2shoes-r_fast --stage supernet
```

##### Select the Best Model

This stage is almost the same as pix2pix.

```bash
bash scripts/munit/edges2shoes-r_fast/evolution_search.sh
```

##### Fine-tuning the Best Model

During our experiments, we observe that fine-tuning the model increases FID.  **You may skip the fine-tuning.**

##### Export the Model

Extract a subnet from the supernet. We provide a code [export.py](../../export.py) to extract a specific subnet according to a configuration description. For example, if the `config_str` of your chosen subnet is `16_16_16_24_56_16_40_40_32_24`, then you can export the model by this command:

```shell
bash scripts/munit/edges2shoes-r_fast/export.sh 16_16_16_24_56_16_40_40_32_24
```