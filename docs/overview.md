# Overview of Code Structure
To help users better understand and use our codebase, we briefly overview the functionality and implementation of each package and each module. Please see the documentation in each file for more details. 

## Scripts
* [assemble.py](../assemble.py) is an auxiliary script to assemble two MUNIT generators to a new one for inference. Specificly, it will export a new MUNIT generator which consists of the content encoder and the decoder in generator A and the style encoder in generator B.
* [distill.py](../distill.py) is a script for distillation. The distiller supports ResNet and SPADE models (with option `--distiller`: e.g. `spade`, `resnet`). You could specify the teacher model with options `--teacher_netG` and `--teacher_ngf`, and load the pretrained teacher weight with `--restore_teacher_G_path`. Similarly, You could specify the student model with options `--student_netG` and `--student_ngf`, and load the pretrained student weight with `--restore_student_G_path`. We also support pruning before distillation. You just need to specify the model you would like to prune with options `--pretrained_netG` and `--pretrained_ngf`, and load the weight with `restore_pretrain_G_path`. 
* [evolution_search.py](../evolution_search.py) is a script for evolution searching. Once you have get your supernet weight, you can use this script to search for the best performed subnet. It will load a saved supernet model from `--resotre_G_path` and save the searching results to `--output_dir`.
* [export.py](../export.py) is an auxiliary script to extract a specific subnet for a supernet and export it. You need specify the  supernet model with`--model` and `--ngf` and the model weight with `--input_path`. To extract the specific subnet, you need to provide the subnet configuration with `--config_str` and the exported model will be saved to `--output_path`.
* [get_real_stat.py](../get_real_stat.py) is an auxiliary script to get the statistical information of the ground-truth images to compute FID. You need to specify the dataset with options `--dataroot`, `--dataset_mode` and the direction you would like to train with  `--direction`. 
* [latency.py](../latency.py) is a general-purpose test script to measure the latency of the models. The usage is almost the same as [test.py](../test.py).
* [merge.py](../merge.py) is an auxiliary script to merge multiple searching results. It is usually used in manually-split parallel searching.
* [remove_spectral_norm.py](../remove_spectral_norm.py) is an auxiliary script to remove the spectral normalization of the GauGAN model.
* [search.py](../search.py) is a script for evaluating all candidate subnets. Once you have get your supernet weight, you can use this script to evaluate the performance of candidate subnets. It will load a saved supernet model from `--resotre_G_path` and save the evaluation results to `--output_path`. See the our training tutorials of [Fast GAN Compression](tutorials/fast_gan_compression.md) and [GAN Compression](tutorials/gan_compression.md) for more details.
* [select_arch.py](../select_arch.py) is an auxiliary script to parse the output pickle by the [search.py](../search.py) and select the architecture configurations you want.
* [test.py](../test.py) is a general-purpose test script. Once you have get your model weight, you can use this script to test your model. It will load a saved model from `--restore_G_path` and save the results to `--results_dir`.
* [train.py](../train.py) is a general-purpose original model training script. It works for various models (with option `--model`: e.g., `pix2pix`, `cycle_gan`) and different datasets (with option `--dataset_mode`: e.g., `aligned`, `unaligned`). See the our training tutorials of [Fast GAN Compression](tutorials/fast_gan_compression.md) and [GAN Compression](tutorials/gan_compression.md) for more details.
* [train_supernet.py](../train_supernet.py) is a script for the "once-for-all" network training and finetuning. The "once-for-all" network supports ResNet and SPADE models (with option `--supernet`: e.g. `spade`, `resnet`). Like distillation, you could specify the teacher model with options `--teacher_netG` and `--teacher_ngf`, and load the pre-trained teacher weight with `--restore_teacher_G_path`. Similarly, You could specify the student model with options `--student_netG` and `--student_ngf`, and load the pre-trained student weight with `--restore_student_G_path`. Moreover, you need to specify the candidate subnet set with option `--config_set` when training a supernet. When we are fine-tuning a specific subnet, you need to specify the chosen subnet configuration with option `--config_str`.
* [trainer.py](../trainer.py) is a module that implements the training logic for [train.py](../train.py), [distill.py](../distill.py) and [train_supernet.py](../train_supernet.py).

## Directories
### [configs](../configs)

[configs](../configs) directory contains modules related to the search space configuration used in "once-for-all" network training.

* [\_\_init\_\_.py](../configs/__init__.py) contains an encoding and a decoding function of the configuration description string.
* [resnet_configs.py](../configs/resnet_configs.py) is a module that implements a configuration set class that used in training ResNet-based "once-for-all" network. 
* [single_configs.py](../configs/single_configs.py) is a module that implements a configuration set class that only contains a single configuration. Usually, it is used in fine-tuning.
* [spade_configs.py](../configs/spade_configs.py) is a module that implements a configuration set class that used in training SPADE-based "once-for-all" network. 

### [data](../data)

[data](../data) directory contains all the modules related to data loading and preprocessing. To add a custom dataset class called `dummy`, you need to add a file called `dummy_dataset.py` and define a subclass `DummyDataset` inherited from `BaseDataset`. You need to implement four functions: `__init__` (initialize the class, you need to first call `BaseDataset.__init__(self, opt)`), `__len__` (return the size of dataset), `__getitem__`　(get a data point), and optionally `modify_commandline_options` (add dataset-specific options and set default options). Now you can use the dataset class by specifying flag `--dataset_mode dummy`.  Below we explain each file in details.

* [\_\_init\_\_.py](../data/__init__.py) implements the interface between this package and training and test scripts. Other scripts will call `dataset = create_dataset(opt)` to create a dataset for training given the option `opt`. They can also call `dataset = create_eval_dataset(opt)` to create a dataset for evaluation given the option `opt`.
* [aligned_dataset.py](../data/aligned_dataset.py) includes a dataset class that can load image pairs for pix2pix. It assumes a single image directory `/path/to/data/train`, which contains image pairs in the form of {A,B}. See [here](https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix/blob/master/docs/tips.md#prepare-your-own-datasets-for-pix2pix) on how to prepare aligned datasets. During test time, you need to prepare a directory `/path/to/data/val` as test data.
* [base_dataset.py](../data/base_dataset.py) implements an abstract base class ([ABC](https://docs.python.org/3/library/abc.html)) for datasets. It also includes common transformation functions (e.g., `get_transform`, `__scale_width`), which can be later used in subclasses.
* [cityscapes_dataset.py](../data/cityscapes_dataset.py) includes a dataset class that can load cityscapes datasets for GauGAN.
* [coco_dataset.py](../data/coco_dataset.py) includes a dataset class that can load Coco-Stuff datasets for GauGAN.
* [image_folder.py](../data/image_folder.py) implements an image folder class. We modify the official PyTorch image folder [code](https://github.com/pytorch/vision/blob/master/torchvision/datasets/folder.py) so that this class can load images from both the current directory and its subdirectories.
* [single_dataset.py](../data/single_dataset.py) includes a dataset class that can load a set of single images specified by the path `--dataroot /path/to/data`. It can be used for generating CycleGAN results only for one side with the model option `--model test`.
* [spade_dataset.py](../data/spade_dataset.py) implements an abstract base class for the datasets for GauGAN model.
* [unaligned_dataset.py](../data/unaligned_dataset.py) includes a dataset class that can load unaligned/unpaired datasets. It assumes that two directories to host training images from domain A `/path/to/data/trainA` and from domain B `/path/to/data/trainB` respectively. Then you can train the model with the dataset flag `--dataroot /path/to/data`. Similarly, you need to prepare two directories `/path/to/data/testA` and `/path/to/data/testB` during test time.

### [datasets](../datasets)
[datasets](../datasets) directory contains some scripts to prepare the datasets you will use.

### [distillers](../distillers)
[distillers](../distillers) directory contains modules related to distillation for different model architectures.

* [\_\_init\_\_.py](../distillers/__init__.py)  implements the interface between this package and distill scripts.  `distill.py` calls `from distillers import create_distiller` and `distiller = create_distiller(opt)` to create a distiller given the option `opt`. You also need to call `distiller.setup(opt)` to properly initialize the model.
* [base_resnet_distiller.py](../models/base_resnet_distiller.py) implements an abstract base class for the distiller for ResNet architectures. It also includes commonly used helper functions for intermediate distillation, which can be later used in subclasses.
* [base_spade_distiller.py](../models/base_spade_distiller.py) implements an abstract base class for the distiller for SPADE architectures. It also includes commonly used helper functions for intermediate distillation, which can be later used in subclasses.
* [resnet_distiller.py](../models/resnet_distiller.py) is a subclass of [base_resnet_distiller.py](../models/base_resnet_distiller.py) implements an class for the distiller of ResNet architectures.
* [spade_distiller.py](../models/spade_distiller.py) is a subclass of [base_spade_distiller.py](../models/base_spade_distiller.py) implements an class for the distiller of SPADE architectures.

### [metric](../metric)
[metric](../metric) directory contains modules related to evaluation metric. 

* [\_\_init\_\_.py](../metric/__init__.py)  implements the interface of creating the metric models and computing the metrics.
* [cityscapes_mIoU.py](../metric/cityscapes_mIoU.py) implements the logic of computing the mIoU metric of cityscapes.
* [coco_scores.py](../metric/coco_scores.py) implements the logic of computing the mIoU and pixel accuracy of Coco-Stuff.
* [deeplabv2.py](../metric/deeplabv2.py) implements the [deeplabv2](https://github.com/kazuto1011/deeplab-pytorch) model for coco scores computation.
* [drn.py](../metric/drn.py) implements the [drn](https://github.com/kazuto1011/deeplab-pytorch) model for cityscapes mIoU computation.
* [fid_score.py](../metric/fid_score.py) implements the logic of computing the FID score.
* [inception.py](../metric/inception.py) implements the [inception_v3](https://github.com/mseitzer/pytorch-fid) model for FID computation.

### [models](../models)

[models](../models) directory contains modules related to original model training, testing and network architectures. 

* [modules](../models/modules) directory contains many pytorch nn.Module networks and loss moduels.
  * [munit_architecture](../models/modules/munit_architecture) directory contains some MUNIT generator Pytorch modules:
    * [munit_generator.py](../models/modules/resnet_architecture/munit_generator.py) implements the original MUNIT Adain generator in the original [MUNIT repository](https://github.com/NVlabs/MUNIT).
  * [resnet_architecture](../models/modules/resnet_architecture) directory contains some ResNet-based generator Pytorch modules:
    * [legacy_sub_mobile_resnet_generator.py](../models/modules/resnet_architecture/legacy_sub_mobile_resnet_generator.py) implements a deprecated old version of [sub_mobile_resnet_generator.py](../models/modules/resnet_architecture/sub_mobile_resnet_generator.py), which is only used to test some of our old paper models.
    * [mobile_resnet_generator.py](../models/modules/resnet_architecture/mobile_resnet_generator.py) implements a MobileNet-style [resnet_generator.py](../models/modules/resnet_architecture/resnet_generator.py).
    * [resnet_generator.py](../models/modules/resnet_architecture/resnet_generator.py) implements the original ResNet-based generator in the original [CycleGAN and pix2pix repository](https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix).
    * [sub_mobile_resnet_generator.py](../models/modules/resnet_architecture/sub_mobile_resnet_generator.py) implements our final compressed ResNet-based generator.
    * [super_mobile_resnet_generator.py](../models/modules/resnet_architecture/super_mobile_resnet_generator.py) implements our "once-for-all" ResNet-based generator.
  * [spade_architecture](../models/modules/spade_architecture) directory contains some SPADE-based generator Pytorch modules for GauGAN:
    * [mobile_spade_generator.py](../models/modules/spade_architecture/mobile_spade_generator.py) implements the MobileNet-Style [spade_generator.py](../models/modules/spade_architecture/spade_generator.py).
    * [normalization.py](../models/modules/spade_architecture/normalization.py) implements different kinds of SPADE modules for SPADE-based generators.
    * [spade_generator.py](../models/modules/spade_architecture/spade_generator.py) implements the original spade generator as in [SPADE repository](https://github.com/NVlabs/SPADE).
    * [sub_mobile_spade_generator.py](../models/modules/spade_architecture/sub_mobile_spade_generator.py) implements our final compressed SPADE-based generator.
    * [super_mobile_spade_generator.py](../models/modules/spade_architecture/super_mobile_spade_generator.py) implements our "once-for-all" SPADE-based generator.
  * [spade_modules](../models/modules/spade_modules) contains some Pytorch-module wrapped modules which implement the logic for the GauGAN testing and training.
    * [base_spade_distiller_modules.py](../models/modules/spade_modules/base_spade_distiller_modules.py) is the base class of [spade_distiller_modules.py](../models/modules/spade_modules/spade_distiller_modules.py) and [spade_supernet_modules.py](../models/modules/spade_modules/spade_supernet_modules.py), which implements some logic of distillation and "once-for-all" training of GauGAN.
    * [spade_distiller_modules.py](../models/modules/spade_modules/spade_distiller_modules.py) implements the logic of the distillation for GauGAN.
    * [spade_model_modules.py](../models/modules/spade_modules/spade_model_modules.py) implements the logic of the original GauGAN training and testing. It is also the base class of [base_spade_distiller_modules.py](../models/modules/spade_modules/base_spade_distiller_modules.py).
    * [spade_supernet_modules.py](../models/modules/spade_modules/spade_supernet_modules.py) implements the logic of the "once-for-all" training for GauGAN.
  * [sync_batchnorm](../models/modules/sync_batchnorm) implements the Syncronized BatchNorm, which is from [vacancy/Synchronized-BatchNorm-PyTorch](https://github.com/vacancy/Synchronized-BatchNorm-PyTorch).
  * [discriminators.py](../models/modules/discriminators.py) defines some discriminator architectures. 
  * [loss.py](../models/modules/loss.py) defines some loss modules. 
  * [mobile_modules.py](../models/modules/mobile_modules.py) defines some basic mobile modules.
  * [super_modules.py](../models/modules/super_modules.py) defines some basic supernet ("once-for-all") modules.
* [\_\_init\_\_.py](../models/__init__.py)  implements the interface between this package and training and test scripts.  `train.py` and `test.py` call `model = create_model(opt)` to create a model given the option `opt`. You also need to call `model.setup(opt)` to properly initialize the model.
* [base_model.py](../models/base_model.py) implements an abstract base class ([ABC](https://docs.python.org/3/library/abc.html)) for models. It also includes commonly used helper functions (e.g., `setup`, `test`, `update_learning_rate`, `save_networks`, `load_networks`), which can be later used in subclasses.
* [cycle_gan_model.py](../models/cycle_gan_model.py) implements the [CycleGAN model](https://junyanz.github.io/CycleGAN/), for learning image-to-image translation  without paired data.  The model training requires `--dataset_mode unaligned` dataset. By default, it uses a ResNet generator, a `--netD n_layer --n_layers_D 3` discriminator (PatchGAN  introduced by pix2pix), and a least-square GANs [objective](https://arxiv.org/abs/1611.04076) (`--gan_mode lsgan`).
* [munit_model.py](../models/munit_model.py) implements the original [MUNIT](https://github.com/NVlabs/MUNIT) training.
* [munit_test_model.py](../models/munit_test_model.py) implements the test logic of [MUNIT](https://github.com/NVlabs/MUNIT).
* [networks.py](../models/networks.py) module implements normalization layers, initialization methods, and optimization schedulers (i.e., learning rate policy).
* [pix2pix_model.py](../models/pix2pix_model.py) implements the [pix2pix model](https://phillipi.github.io/pix2pix/), for learning a mapping from input images to output images given paired data. The model training requires `--dataset_mode aligned` dataset. By default, it uses a ResNet generator, a `--netD n_layer --n_layers_D 3` discriminator (PatchGAN), and  a `--gan_mode hinge` GAN loss.
* [spade_model.py](../models/spade_model.py) implements training and testing logic of [GauGAN model](https://nvlabs.github.io/SPADE/), a state-of-the-art paired image-to-image translation model. It uses a SPADE-based generator.
* [test_model.py](../models/test_model.py) implements a model that can be used to generate results in one direction for CycleGAN and pix2pix models.

### [supernets](../supernets)

[supernets](../supernets) directory contains modules related to "once-for-all" network training and fine-tuning for different model architectures (currently only the ResNet architecture).

* [\_\_init\_\_.py](../supernets/__init__.py)  implements the interface between this package and "once-for-all" network training scripts.  `train_supernet.py` calls `from supernets import create_supernet` and `supernet = create_supernet(opt)` to create a "once-for-all" network given the option `opt`. You also need to call `supernet.setup(opt)` to properly initialize the "once-for-all" network.
* [resnet_supernet.py](../models/resnet_supernet.py) is a subclass of [base_resnet_distiller.py](../models/base_resnet_distiller.py) implements an class for the "once-for-all" training for ResNet-based architectures.
* [spade_supernet.py](../models/spade_supernet.py) is a subclass of [base_spade_distiller.py](../models/base_spade_distiller.py) implements an class for the "once-for-all" training for SPADE-based architectures.

### [options](../options)

[options](../options) directory includes our option modules: training options, distill options, search options, "once-for-all" training options, test options, and basic options (the base class of all other options).

* [base_options.py](../options/base_options.py) includes options that are used in both training and test. It also implements a few helper functions such as parsing, printing, and saving the options. It also gathers additional options defined in `modify_commandline_options` functions in both dataset class and model class.
* [distill_options.py](../options/distill_options.py) includes options that are only used during distillation.
* [search_options.py](../options/searchl_options.py) includes options that are only used during searching.
* [supernet_options.py](../options/supernet_options.py) includes options that are only used during "once-for-all" network training and fine-tuning.
* [test_options.py](../options/test_options.py) includes options that are only used during test time.
* [train_options.py](../options/train_options.py) includes options that are only used during the original model training time.

### [util](../util)

[util](../utils) directory includes a miscellaneous collection of useful helper functions.
* [html.py](../utils/html.py) implements a module that saves images into a single HTML file.  It consists of functions such as `add_header` (add a text header to the HTML file), `add_images` (add a row of images to the HTML file), `save` (save the HTML to the disk). It is based on Python library `dominate`, a Python library for creating and manipulating HTML documents using a DOM API.
* [image_pool.py](../utils/image_pool.py) implements an image buffer that stores previously generated images. This buffer enables us to update discriminators using a history of generated images rather than the ones produced by the latest generators. The original idea was discussed in this [paper](http://openaccess.thecvf.com/content_cvpr_2017/papers/Shrivastava_Learning_From_Simulated_CVPR_2017_paper.pdf). The size of the buffer is controlled by the flag `--pool_size`.
* [logger.py](../utils/logger.py) provides a class for logging the training information. It also implements interfaces to the tensorboard.
* [util.py](../utils/util.py) consists of simple helper functions such as `tensor2im` (convert a tensor array to a numpy image array) and `load_network` (load a network from a specific checkpoint).
* [weight_transfer.py](../utils/weight_transfer.py) implements a function to transfer the weights of teacher network to a small student network of the same architecture. It functions as pruning.

