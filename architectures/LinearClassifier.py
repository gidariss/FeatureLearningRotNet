import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

class Flatten(nn.Module):
    def __init__(self):
        super(Flatten, self).__init__()

    def forward(self, feat):
        return feat.view(feat.size(0), -1)

class Classifier(nn.Module):
    def __init__(self, opt):
        super(Classifier, self).__init__()
        nChannels = opt['nChannels']
        num_classes = opt['num_classes']
        pool_size = opt['pool_size']
        pool_type = opt['pool_type'] if ('pool_type' in opt) else 'max'
        nChannelsAll = nChannels * pool_size * pool_size

        self.classifier = nn.Sequential()
        if pool_type == 'max':
            self.classifier.add_module('MaxPool', nn.AdaptiveMaxPool2d((pool_size, pool_size)))
        elif pool_type == 'avg':
            self.classifier.add_module('AvgPool', nn.AdaptiveAvgPool2d((pool_size, pool_size)))
	self.classifier.add_module('BatchNorm', nn.BatchNorm2d(nChannels, affine=False))
        self.classifier.add_module('Flatten', Flatten())
        self.classifier.add_module('LiniearClassifier', nn.Linear(nChannelsAll, num_classes))
        self.initilize()

    def forward(self, feat):
        return self.classifier(feat)

    def initilize(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                fin = m.in_features
                fout = m.out_features
                std_val = np.sqrt(2.0/fout)
                m.weight.data.normal_(0.0, std_val)
                if m.bias is not None:
                    m.bias.data.fill_(0.0)

def create_model(opt):
    return Classifier(opt)
