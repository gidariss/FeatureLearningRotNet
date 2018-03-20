from __future__ import print_function
from PIL import Image
import os
import os.path
import numpy as np
import sys
import imp
from tqdm import tqdm
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.optim 
import torchnet as tnt


import numbers

class FastConfusionMeter(object):
    def __init__(self, k, normalized = False):
        #super(FastConfusionMeter, self).__init__()
        self.conf = np.ndarray((k,k), dtype=np.int32)
        self.normalized = normalized
        self.reset()

    def reset(self):
        self.conf.fill(0)

    def add(self, output, target):
        output = output.cpu().squeeze().numpy()
        target = target.cpu().squeeze().numpy()
        
        if np.ndim(output) == 1:
            output = output[None]
            
        onehot = np.ndim(target) != 1
        assert output.shape[0] == target.shape[0], \
                'number of targets and outputs do not match'
        assert output.shape[1] == self.conf.shape[0], \
                'number of outputs does not match size of confusion matrix'
        assert not onehot or target.shape[1] == output.shape[1], \
                'target should be 1D Tensor or have size of output (one-hot)'
        if onehot:
            assert (target >= 0).all() and (target <= 1).all(), \
                    'in one-hot encoding, target values should be 0 or 1'
            assert (target.sum(1) == 1).all(), \
                    'multi-label setting is not supported'

        target = target.argmax(1) if onehot else target
        pred = output.argmax(1)
        
        target    = target.astype(np.int32)
        pred      = pred.astype(np.int32)
        conf_this = np.bincount(target * self.conf.shape[0] + pred,minlength=np.prod(self.conf.shape))
        conf_this = conf_this.astype(self.conf.dtype).reshape(self.conf.shape)
        self.conf += conf_this

    def value(self):
        if self.normalized:
            conf = self.conf.astype(np.float32)
            return conf / conf.sum(1).clip(min=1e-12)[:,None]
        else:
            return self.conf

def getConfMatrixResults(matrix):
    assert(len(matrix.shape)==2 and matrix.shape[0]==matrix.shape[1])
    
    count_correct = np.diag(matrix)
    count_preds   = matrix.sum(1)
    count_gts     = matrix.sum(0)
    epsilon       = np.finfo(np.float32).eps
    accuracies    = count_correct / (count_gts + epsilon)
    IoUs          = count_correct / (count_gts + count_preds - count_correct + epsilon)
    totAccuracy   = count_correct.sum() / (matrix.sum() + epsilon)
    
    num_valid     = (count_gts > 0).sum()
    meanAccuracy  = accuracies.sum() / (num_valid + epsilon)
    meanIoU       = IoUs.sum() / (num_valid + epsilon)
    
    result = {'totAccuracy': round(totAccuracy,4), 'meanAccuracy': round(meanAccuracy,4), 'meanIoU': round(meanIoU,4)}
    if num_valid == 2:
        result['IoUs_bg'] = round(IoUs[0],4)
        result['IoUs_fg'] = round(IoUs[1],4)
        
    return result
    
class AverageConfMeter(object):
    def __init__(self):
        self.reset()
        
    def reset(self):
        self.val = np.asarray(0, dtype=np.float64)
        self.avg = np.asarray(0, dtype=np.float64)
        self.sum = np.asarray(0, dtype=np.float64)
        self.count = 0
        
    def update(self, val):
        self.val = val
        if self.count == 0:
            self.sum = val.copy().astype(np.float64)
        else:
            self.sum += val.astype(np.float64)
        
        self.count += 1
        self.avg = getConfMatrixResults(self.sum)

class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0.0
        self.sum = 0.0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += float(val * n)
        self.count += n
        self.avg = round(self.sum / self.count,4)

class LAverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = []
        self.avg = []
        self.sum = []
        self.count = 0

    def update(self, val):
        self.val = val
        self.count += 1
        if len(self.sum) == 0:
            assert(self.count == 1)
            self.sum = [v for v in val]
            self.avg = [round(v,4) for v in val]
        else:
            assert(len(self.sum) == len(val))
            for i, v in enumerate(val):
                self.sum[i] += v
                self.avg[i] = round(self.sum[i] / self.count,4)

class DAverageMeter(object):
    def __init__(self):
        self.reset()
    
    def reset(self):
        self.values = {}

    def update(self, values):
        assert(isinstance(values, dict))
        for key, val in values.items():
            if isinstance(val, (float, int)):
                if not (key in self.values):
                    self.values[key] = AverageMeter()
                self.values[key].update(val)
            elif isinstance(val, (tnt.meter.ConfusionMeter,FastConfusionMeter)):            
                if not (key in self.values):
                    self.values[key] = AverageConfMeter()
                self.values[key].update(val.value())
            elif isinstance(val, AverageConfMeter):
                if not (key in self.values):
                    self.values[key] = AverageConfMeter()
                self.values[key].update(val.sum)                
            elif isinstance(val, dict):
                if not (key in self.values):
                    self.values[key] = DAverageMeter()
                self.values[key].update(val)
            elif isinstance(val, list):
                if not (key in self.values):
                    self.values[key] = LAverageMeter()
                self.values[key].update(val)                
                
    def average(self):
        average = {}
        for key, val in self.values.items():
            if isinstance(val, type(self)):
                average[key] = val.average()
            else:
                average[key] = val.avg
                
        return average
        
    def __str__(self):
        ave_stats = self.average()
        return ave_stats.__str__()
