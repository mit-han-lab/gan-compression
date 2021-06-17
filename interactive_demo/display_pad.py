import os

import torch
import wget
from PyQt5.QtWidgets import *
from torch import nn

from hparams import *
from util import load_network, pixmap2tensor, pixmap2tvm, tensor2pixmap, tvm2pixmap


class DisplayPad(QLabel):

    def __init__(self, canvas, opt):
        super(DisplayPad, self).__init__()
        self.mode = 'rectangle'
        self.canvas = canvas
        self.background_color = QColor(Qt.white)
        self.setPixmap(QPixmap(*CANVAS_DIMENSIONS))
        self.pixmap().fill(self.background_color)
        self.opt = opt
        prefix = 'https://hanlab.mit.edu/files/gan_compression/pretrained/demo'
        os.makedirs('checkpoints', exist_ok=True)
        if opt.model == 'tvm':
            import tvm
            from tvm.contrib.graph_executor import GraphModule
            if not os.path.exists('checkpoints/tvm.tar'):
                wget.download('%s/tvm.tar' % prefix, 'checkpoints/tvm.tar')
            lib = tvm.runtime.load_module('checkpoints/tvm.tar')
            device = tvm.cuda()
            gmod = GraphModule(lib['default'](device))

            def executor(input):
                gmod.set_input(0, input)
                gmod.run()
                return gmod.get_output(0)

            self.model = executor

        elif opt.model == 'compressed':
            from models.sub_mobile_resnet_generator import SubMobileResnetGenerator
            self.model = SubMobileResnetGenerator(3, 3, norm_layer=nn.InstanceNorm2d, n_blocks=9,
                                                  config={'channels': [24, 24, 40, 56, 24, 56, 16, 40]})
            if not os.path.exists('checkpoints/compressed.pth'):
                wget.download('%s/compressed.pth' % prefix, 'checkpoints/compressed.pth')
            load_network(self.model, 'checkpoints/compressed.pth')
        elif opt.model == 'legacy':
            from models.legacy_sub_mobile_resnet_generator import LegacySubMobileResnetGenerator
            self.model = LegacySubMobileResnetGenerator(3, 3, norm_layer=nn.InstanceNorm2d, n_blocks=9,
                                                        config={'channels': [32, 32, 40, 48, 16, 32]})
            if not os.path.exists('checkpoints/legacy.pth'):
                wget.download('%s/legacy.pth' % prefix, 'checkpoints/legacy.pth')
            load_network(self.model, 'checkpoints/legacy.pth')
        else:
            from models.resnet_generator import ResnetGenerator
            self.model = ResnetGenerator(3, 3, ngf=64, norm_layer=nn.InstanceNorm2d, n_blocks=9)
            if not os.path.exists('checkpoints/full.pth'):
                wget.download('%s/full.pth' % prefix, 'checkpoints/full.pth')
            load_network(self.model, 'checkpoints/full.pth')
        if not opt.use_cpu and opt.model != 'tvm':
            self.model = self.model.cuda()
        self.update()

    def update(self):
        pixmap = self.canvas.pixmap()
        if self.opt.model == 'tvm':
            tensor = pixmap2tvm(pixmap)
            tensor = self.model(tensor)
            output_map = tvm2pixmap(tensor)
        else:
            tensor = pixmap2tensor(pixmap)
            if not self.opt.use_cpu:
                tensor = tensor.cuda()
            with torch.no_grad():
                tensor = self.model(tensor)
            if not self.opt.use_cpu:
                tensor = tensor.cpu()
            output_map = tensor2pixmap(tensor)
        self.setPixmap(output_map)
        super(DisplayPad, self).update()
