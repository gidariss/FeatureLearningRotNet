import torch
from torch.autograd import Variable
import utils
import time

from . import Algorithm
from pdb import set_trace as breakpoint


def accuracy(output, target, topk=(1,)):
    """Computes the precision@k for the specified values of k"""
    maxk = max(topk)
    batch_size = target.size(0)

    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))

    res = []
    for k in topk:
        correct_k = correct[:k].view(-1).float().sum(0)
        res.append(correct_k.mul_(100.0 / batch_size))
    return res

class FeatureClassificationModel(Algorithm):
    def __init__(self, opt):
        self.out_feat_keys = opt['out_feat_keys']
        Algorithm.__init__(self, opt)

    def allocate_tensors(self):
        self.tensors = {}
        self.tensors['dataX'] = torch.FloatTensor()
        self.tensors['labels'] = torch.LongTensor()

    def train_step(self, batch):
        return self.process_batch(batch, do_train=True)

    def evaluation_step(self, batch):
        return self.process_batch(batch, do_train=False)

    def process_batch(self, batch, do_train=True):
        #*************** LOAD BATCH (AND MOVE IT TO GPU) ********
        start = time.time()
        self.tensors['dataX'].resize_(batch[0].size()).copy_(batch[0])
        self.tensors['labels'].resize_(batch[1].size()).copy_(batch[1])
        dataX = self.tensors['dataX']
        labels = self.tensors['labels']
        batch_load_time = time.time() - start
        #********************************************************

        #********************************************************
        start = time.time()
        out_feat_keys = self.out_feat_keys
        finetune_feat_extractor = self.optimizers['feat_extractor'] is not None
        if do_train: # zero the gradients
            self.optimizers['classifier'].zero_grad() 
            if finetune_feat_extractor:
                self.optimizers['feat_extractor'].zero_grad()
            else:
                self.networks['feat_extractor'].eval()
        #********************************************************

        #***************** SET TORCH VARIABLES ******************
        dataX_var = Variable(dataX, volatile=((not do_train) or (not finetune_feat_extractor)))
        labels_var = Variable(labels, requires_grad=False)
        #********************************************************

        #************ FORWARD PROPAGATION ***********************
        feat_var = self.networks['feat_extractor'](dataX_var, out_feat_keys=out_feat_keys)
        if not finetune_feat_extractor:
            if isinstance(feat_var, (list, tuple)):
                for i in range(len(feat_var)):
                    feat_var[i] = Variable(feat_var[i].data, volatile=(not do_train))
            else:
                feat_var = Variable(feat_var.data, volatile=(not do_train))
        pred_var = self.networks['classifier'](feat_var)
        #********************************************************

        #*************** COMPUTE LOSSES *************************
        record = {}
        if isinstance(pred_var, (list, tuple)):
            loss_total = None
            for i in range(len(pred_var)):
                loss_this = self.criterions['loss'](pred_var[i], labels_var)
                loss_total = loss_this if (loss_total is None) else (loss_total + loss_this)
                record['prec1_c'+str(1+i)] = accuracy(pred_var[i].data, labels, topk=(1,))[0][0]
                record['prec5_c'+str(1+i)] = accuracy(pred_var[i].data, labels, topk=(5,))[0][0]
        else:
            loss_total = self.criterions['loss'](pred_var, labels_var)
            record['prec1'] = accuracy(pred_var.data, labels, topk=(1,))[0][0]
            record['prec5'] = accuracy(pred_var.data, labels, topk=(5,))[0][0]
        record['loss'] = loss_total.data[0]
        #********************************************************

        #****** BACKPROPAGATE AND APPLY OPTIMIZATION STEP *******
        if do_train:
            loss_total.backward()
            self.optimizers['classifier'].step()
            if finetune_feat_extractor:
                self.optimizers['feat_extractor'].step()
        #********************************************************
        batch_process_time = time.time() - start
        total_time = batch_process_time + batch_load_time
        record['load_time'] = 100*(batch_load_time/total_time)
        record['process_time'] = 100*(batch_process_time/total_time)

        return record
