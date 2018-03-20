import torch
import torch.nn as nn
import imp
import os


current_path = os.path.abspath(__file__)
filepath_to_linear_classifier_definition = os.path.join(os.path.dirname(current_path), 'LinearClassifier.py')
LinearClassifier = imp.load_source('',filepath_to_linear_classifier_definition).create_model


class MClassifier(nn.Module):
    def __init__(self, opts):
        super(MClassifier, self).__init__()
        self.classifiers = nn.ModuleList([LinearClassifier(opt) for opt in opts])
        self.num_classifiers = len(opts)


    def forward(self, feats):
        assert(len(feats) == self.num_classifiers)
        return [self.classifiers[i](feat) for i, feat in enumerate(feats)]


def create_model(opt):
    return MClassifier(opt)

