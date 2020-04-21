# Overview of Code Structure
To help users better understand and use our codebase, we briefly overview the functionality and implementation of each package and each module. Please see the documentation in each file for more details. 

## Scripts
* [test.py](../test.py) is a general-purpose test script. Once you have get your model weight, you can use this script to test your model. It will load a saved model from `--restore_G_path` and save the results to `--results_dir`.
* [train.py](../train.py) is a general-purpose original model training script. It works for various models (with option `--model`: e.g., `pix2pix`, `cyclegan`) and different datasets (with option `--dataset_mode`: e.g., `aligned`, `unaligned`). See the main [README](../README.md) and [training/test  tips](tips.md) for more details.
* [distill.py](../distill.py) is a script for distillation. Currently, the distiller only supports the ResNet models (with option `--distiller resnet`). You could specify the teacher model with options `--teacher_netG` and `--teacher_ngf`, and load the pretrained teacher weight with `--restore_teacher_G_path`. Similarly, You could specify the student model with options `--student_netG` and `--student_ngf`, and load the pretrained student weight with `--restore_student_G_path`. We also support pruning before distillation. You just need to specify the model you would like to prune with options `--pretrained_netG` and `--pretrained_ngf`, and load the weight with `restore_pretrain_G_path`. 
* [train_supernet.py](../train_supernet.py) is a script for the "once-for-all" network training and finetuning. Currently, the "once-for-all" network training only supports the ResNet models (with option `--supernet resnet`). Like distillation, you could specify the teacher model with options `--teacher_netG` and `--teacher_ngf`, and load the pre-trained teacher weight with `--restore_teacher_G_path`. Similarly, You could specify the student model with options `--student_netG` and `--student_ngf`, and load the pre-trained student weight with `--restore_student_G_path`. Moreover, you need to specify the candidate subnet set with option `--config_set` when training a supernet. When we are fine-tuning a specific subnet, you need to specify the chosen subnet configuration with option `--config_str`.
* [search.py](../search.py) is a script for evaluating all candidate subnets. Once you have get your supernet weight, you can use this script to evaluate the performance of candidate subnets. It will load a saved supernet model from `--resotre_G_path` and save the evaluation results to `--output_path`. See the main [README](../README.md) for the saved result format.
* [search_multi.py](../search_multi.py) is multi-GPU-evaluation supporting version of [search.py](../search.py). The usage is almost the same of [search.py](../search.py). All you need to do is specify the gpus you would like to use (with option `--gpu_ids`). **(Warning: we sometimes observe there's a deadlock using search_multi.py after evaluating all candidate subnets when the evaluation takes too long. )**
* [select_arch.py](../export.py) is an auxiliary script to parse the output pickle by the [search.py](../search.py) and select the architecture configurations you want.
* [export.py](../export.py) is an auxiliary script to extract a specific subnet for a supernet and export it. You need specify the  supernet model with `--ngf` and the model weight with `--input_path`. To extract the specific subnet, you need to provide the subnet configuration with `--config_str` and the exported model will be saved to `--output_path`.
* [get_real_stat.py](../get_real_stat.py) is an auxiliary script to get the statistical information of the ground-truth images to compute FID. You need to specify the dataset with options `--dataroot`, `--dataset_mode` and the direction you would like to train with  `--direction`. 
* [trainer.py](../trainer.py) is a script the implements the training logic for [train.py](../train.py), [distill.py](../distill.py) and [train_supernet.py](../train_supernet.py).

## Directories

### [data](../data)

[data](../data) directory contains all the modules related to data loading and preprocessing. To add a custom dataset class called `dummy`, you need to add a file called `dummy_dataset.py` and define a subclass `DummyDataset` inherited from `BaseDataset`. You need to implement four functions: `__init__` (initialize the class, you need to first call `BaseDataset.__init__(self, opt)`), `__len__` (return the size of dataset), `__getitem__`　(get a data point), and optionally `modify_commandline_options` (add dataset-specific options and set default options). Now you can use the dataset class by specifying flag `--dataset_mode dummy`. See our template dataset [class](../data/template_dataset.py) for an example.   Below we explain each file in details.

* [\_\_init\_\_.py](../data/__init__.py) implements the interface between this package and training and test scripts. Other scripts will call `from data import create_dataset` and `dataset = create_dataset(opt)` to create a dataset for training given the option `opt`. They can also call `from data import create_eval_dataset` and `dataset = create_eval_dataset(opt)` to create a dataset for evaluation given the option `opt`.
* [base_dataset.py](../data/base_dataset.py) implements an abstract base class ([ABC](https://docs.python.org/3/library/abc.html)) for datasets. It also includes common transformation functions (e.g., `get_transform`, `__scale_width`), which can be later used in subclasses.
* [image_folder.py](../data/image_folder.py) implements an image folder class. We modify the official PyTorch image folder [code](https://github.com/pytorch/vision/blob/master/torchvision/datasets/folder.py) so that this class can load images from both the current directory and its subdirectories.
* [template_dataset.py](../data/template_dataset.py) provides a dataset template with detailed documentation. Check out this file if you plan to implement your own dataset.
* [aligned_dataset.py](../data/aligned_dataset.py) includes a dataset class that can load image pairs. It assumes a single image directory `/path/to/data/train`, which contains image pairs in the form of {A,B}. See [here](https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix/blob/master/docs/tips.md#prepare-your-own-datasets-for-pix2pix) on how to prepare aligned datasets. During test time, you need to prepare a directory `/path/to/data/test` as test data.
* [unaligned_dataset.py](../data/unaligned_dataset.py) includes a dataset class that can load unaligned/unpaired datasets. It assumes that two directories to host training images from domain A `/path/to/data/trainA` and from domain B `/path/to/data/trainB` respectively. Then you can train the model with the dataset flag `--dataroot /path/to/data`. Similarly, you need to prepare two directories `/path/to/data/testA` and `/path/to/data/testB` during test time.
* [single_dataset.py](../data/single_dataset.py) includes a dataset class that can load a set of single images specified by the path `--dataroot /path/to/data`. It can be used for generating CycleGAN results only for one side with the model option `--model test`.


### [models](../models)

[models](../models) directory contains modules related to objective functions, optimizations, and network architectures. To add a custom model class called `dummy`, you need to add a file called `dummy_model.py` and define a subclass `DummyModel` inherited from `BaseModel`. You need to implement four functions: `__init__` (initialize the class; you need to first call `BaseModel.__init__(self, opt)`), `set_input` (unpack data from dataset and apply preprocessing), `forward` (generate intermediate results), `optimize_parameters` (calculate loss, gradients, and update network weights), and optionally `modify_commandline_options` (add model-specific options and set default options). Now you can use the model class by specifying flag `--model dummy`. See our template model [class](../models/template_model.py) for an example.  Below we explain each file in details.

* [\_\_init\_\_.py](../models/__init__.py)  implements the interface between this package and training and test scripts.  `train.py` and `test.py` call `from models import create_model` and `model = create_model(opt)` to create a model given the option `opt`. You also need to call `model.setup(opt)` to properly initialize the model.
* [base_model.py](../models/base_model.py) implements an abstract base class ([ABC](https://docs.python.org/3/library/abc.html)) for models. It also includes commonly used helper functions (e.g., `setup`, `test`, `update_learning_rate`, `save_networks`, `load_networks`), which can be later used in subclasses.
* [pix2pix_model.py](../models/pix2pix_model.py) implements the pix2pix [model](https://phillipi.github.io/pix2pix/), for learning a mapping from input images to output images given paired data. The model training requires `--dataset_mode aligned` dataset. By default, it uses a ResNet generator, a `--netD n_layer --n_layers_D 3` discriminator (PatchGAN), and  a `--gan_mode hinge` GAN loss (standard cross-entropy objective).
* [cycle_gan_model.py](../models/cycle_gan_model.py) implements the CycleGAN [model](https://junyanz.github.io/CycleGAN/), for learning image-to-image translation  without paired data.  The model training requires `--dataset_mode unaligned` dataset. By default, it uses a ResNet generator, a `--netD n_layer --n_layers_D 3` discriminator (PatchGAN  introduced by pix2pix), and a least-square GANs [objective](https://arxiv.org/abs/1611.04076) (`--gan_mode lsgan`).
* [spade_model.py](../models/spade_model.py) implements training and testing (currently only testing) GauGAN [model](https://nvlabs.github.io/SPADE/), a state-of-the-art paired image-to-image translation model. By default, it uses a SPADE generator.
* [networks.py](../models/networks.py) module implements normalization layers, initialization methods, and optimization scheduler (i.e., learning rate policy).
* [test_model.py](../models/test_model.py) implements a model that can be used to generate results in one direction for CycleGAN and pix2pix modesl.
* [modules](../models/modules) directory contains many pytorch nn.Module networks and loss moduels.

### [distillers](../distillers)

[distillers](../distillers) directory contains modules related to distillation for different model architectures (currently only the Resnet architecture).

* [\_\_init\_\_.py](../distillers/__init__.py)  implements the interface between this package and distill scripts.  `distill.py` calls `from distillers import create_distiller` and `distiller = create_distiller(opt)` to create a distiller given the option `opt`. You also need to call `distiller.setup(opt)` to properly initialize the model.
* [base_resnet_distiller.py](../models/base_resnet_distiller.py) implements an abstract base class for the distiller for ResNet architectures. It also includes commonly used helper functions for intermediate distillation, which can be later used in subclasses.
* [resnet_distiller.py](../models/resnet_distiller.py) is a subclass of [BaseResnetDistiller](../models/base_resnet_distiller.py) implements an class for the distiller of ResNet architectures.

### [supernets](../supernets)

[supernets](../supernets) directory contains modules related to "once-for-all" network training and fine-tuning for different model architectures (currently only the ResNet architecture).

* [\_\_init\_\_.py](../supernets/__init__.py)  implements the interface between this package and "once-for-all" network training scripts.  `train_supernet.py` calls `from supernets import create_supernet` and `supernet = create_supernet(opt)` to create a "once-for-all" network given the option `opt`. You also need to call `supernet.setup(opt)` to properly initialize the "once-for-all" network.
* [resnet_supernet.py](../models/resnet_supernet.py) is a subclass of [BaseResnetDistiller](../models/base_resnet_distiller.py) implements an class for the "once-for-all" network for ResNet architectures.

### [configs](../configs)

[configs](../configs) directory contains modules related to the configuration set used in "once-for-all" network training.

* [\_\_init\_\_.py](../configs/__init__.py) contains an encoding and a decoding function of the configuration description string.
* [resnet_configs.py](../configs/resnet_configs.py) is a module that implements a configuration set class that used in training Resnet-based "once-for-all" network. 
* [single_configs.py](../configs/single_configs.py) is a module that implements a configuration set class that only contains a single configuration. Usually, it is used in fine-tuning.


### [options](../options)

[options](../options) directory includes our option modules: training options, test options, and basic options (used in both training and test). `TrainOptions` and `TestOptions` are both subclasses of `BaseOptions`. They will reuse the options defined in `BaseOptions`.

* [\_\_init\_\_.py](../options/__init__.py)  is required to make Python treat the directory `options` as containing packages,
* [base_options.py](../options/base_options.py) includes options that are used in both training and test. It also implements a few helper functions such as parsing, printing, and saving the options. It also gathers additional options defined in `modify_commandline_options` functions in both dataset class and model class.
* [train_options.py](../options/train_options.py) includes options that are only used during training time.
* [test_options.py](../options/test_options.py) includes options that are only used during test time.
* [distill_options.py](../options/distill_options.py) includes options that are only used during distillation.
* [supernet_options.py](../options/supernet_options.py) includes options that are only used during "once-for-all" network training and fine-tuning.


### [util](../util)

[util](../utils) directory includes a miscellaneous collection of useful helper functions.

* [\_\_init\_\_.py](../utils/__init__.py) is required to make Python treat the directory `util` as containing packages,
* [logger.py](../utils/logger.py) provides a class for logging the training information. It also implements interfaces to the tensorboard.
* [html.py](../utils/html.py) implements a module that saves images into a single HTML file.  It consists of functions such as `add_header` (add a text header to the HTML file), `add_images` (add a row of images to the HTML file), `save` (save the HTML to the disk). It is based on Python library `dominate`, a Python library for creating and manipulating HTML documents using a DOM API.
* [image_pool.py](../utils/image_pool.py) implements an image buffer that stores previously generated images. This buffer enables us to update discriminators using a history of generated images rather than the ones produced by the latest generators. The original idea was discussed in this [paper](http://openaccess.thecvf.com/content_cvpr_2017/papers/Shrivastava_Learning_From_Simulated_CVPR_2017_paper.pdf). The size of the buffer is controlled by the flag `--pool_size`.
* [util.py](../utils/util.py) consists of simple helper functions such as `tensor2im` (convert a tensor array to a numpy image array), `load_netword` (load a network from a specific checkpoint), and `mkdirs` (create multiple directories).
* [weight_transfer.py](../utils/weight_transfer.py) implements a function to transfer the weights of teacher network to a small student network of the same architecture.

