import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

class BasicBlock(nn.Module):
    def __init__(self, in_planes, out_planes, kernel_size, stride=1):
        super(BasicBlock, self).__init__()
        padding = (kernel_size-1)/2
        self.layers = nn.Sequential()
        self.layers.add_module('Conv', nn.Conv2d(in_planes, out_planes, \
            kernel_size=kernel_size, stride=stride, padding=padding, bias=False))
        self.layers.add_module('BatchNorm', nn.BatchNorm2d(out_planes))
        self.layers.add_module('ReLU',      nn.ReLU(inplace=True))

    def forward(self, x):
        return self.layers(x)

class GlobalAvgPool(nn.Module):
    def __init__(self):
        super(GlobalAvgPool, self).__init__()

    def forward(self, feat):
        assert(feat.size(2) == feat.size(3))
        feat_avg = F.avg_pool2d(feat, feat.size(2)).view(-1, feat.size(1))
        return feat_avg

class Flatten(nn.Module):
    def __init__(self):
        super(Flatten, self).__init__()

    def forward(self, feat):
        return feat.view(feat.size(0), -1)

class Classifier(nn.Module):
    def __init__(self, opt):
        super(Classifier, self).__init__()
        nChannels      = opt['nChannels']
        num_classes    = opt['num_classes']
        self.cls_type  = opt['cls_type']

        self.classifier = nn.Sequential()

	if self.cls_type == 'MultLayer':
            nFeats = min(num_classes*20, 2048)
            self.classifier.add_module('Flatten',     Flatten())
            self.classifier.add_module('Liniear_1',   nn.Linear(nChannels, nFeats, bias=False))
            self.classifier.add_module('BatchNorm_1', nn.BatchNorm2d(nFeats))
            self.classifier.add_module('ReLU_1',      nn.ReLU(inplace=True))
            self.classifier.add_module('Liniear_2',   nn.Linear(nFeats, nFeats, bias=False))
            self.classifier.add_module('BatchNorm2d', nn.BatchNorm2d(nFeats))
            self.classifier.add_module('ReLU_2',      nn.ReLU(inplace=True))
            self.classifier.add_module('Liniear_F',   nn.Linear(nFeats, num_classes))
        elif self.cls_type == 'NIN_ConvBlock3':
            self.classifier.add_module('Block3_ConvB1',  BasicBlock(nChannels, 192, 3))
            self.classifier.add_module('Block3_ConvB2',  BasicBlock(192, 192, 1))
            self.classifier.add_module('Block3_ConvB3',  BasicBlock(192, 192, 1))
            self.classifier.add_module('GlobalAvgPool',  GlobalAvgPool())
            self.classifier.add_module('Liniear_F',      nn.Linear(192, num_classes))
        elif self.cls_type == 'Alexnet_conv5' or self.cls_type == 'Alexnet_conv4':
            if self.cls_type == 'Alexnet_conv4':
                block5 = nn.Sequential(
                    nn.Conv2d(256, 256, kernel_size=3, padding=1),
                    nn.BatchNorm2d(256),
                    nn.ReLU(inplace=True),
                )
                self.classifier.add_module('ConvB5', block5)
            self.classifier.add_module('Pool5', nn.MaxPool2d(kernel_size=3, stride=2))
            self.classifier.add_module('Flatten', Flatten())
            self.classifier.add_module('Linear1', nn.Linear(256*6*6, 4096, bias=False))
            self.classifier.add_module('BatchNorm1', nn.BatchNorm1d(4096))
            self.classifier.add_module('ReLU1', nn.ReLU(inplace=True))
            self.classifier.add_module('Liniear2', nn.Linear(4096, 4096, bias=False))
            self.classifier.add_module('BatchNorm2', nn.BatchNorm1d(4096))
            self.classifier.add_module('ReLU2', nn.ReLU(inplace=True))
            self.classifier.add_module('LinearF', nn.Linear(4096, num_classes))
        else:
            raise ValueError('Not recognized classifier type: %s' % self.cls_type)

        self.initilize()

    def forward(self, feat):
        return self.classifier(feat)

    def initilize(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                fin = m.in_features
                fout = m.out_features
                std_val = np.sqrt(2.0/fout)
                m.weight.data.normal_(0.0, std_val)
                if m.bias is not None:
                    m.bias.data.fill_(0.0)

def create_model(opt):
    return Classifier(opt)
