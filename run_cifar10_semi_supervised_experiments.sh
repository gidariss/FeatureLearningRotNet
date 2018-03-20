#!/bin/bash
echo "Run semi supervised experiments"

# Train a conv-based classifier on top of the feature maps of the 2nd conv. block of a NIN-based RotNet model 
# trained on the entire training set of CIFAR10.

# Use K=5000 training examples per category (which is equal to using the entire training set).
# CUDA_VISIBLE_DEVICES=2 python main.py --exp=CIFAR10_ConvClassifier_on_RotNet_NIN4blocks_Conv2_feats 
# Use K=1000 training examples per category.
CUDA_VISIBLE_DEVICES=0 python main.py --exp=CIFAR10_ConvClassifier_on_RotNet_NIN4blocks_Conv2_feats_K1000
# Use K=400 training examples per category.
CUDA_VISIBLE_DEVICES=0 python main.py --exp=CIFAR10_ConvClassifier_on_RotNet_NIN4blocks_Conv2_feats_K400
# Use K=100 training examples per category.
CUDA_VISIBLE_DEVICES=0 python main.py --exp=CIFAR10_ConvClassifier_on_RotNet_NIN4blocks_Conv2_feats_K100
# Use K=400 training examples per category.
CUDA_VISIBLE_DEVICES=0 python main.py --exp=CIFAR10_ConvClassifier_on_RotNet_NIN4blocks_Conv2_feats_K20

# Train fully supervised NIN models using subsets of the CIFAR10 training set.
# Use K=5000 training examples per category (which is equal to using the entire training set).
CUDA_VISIBLE_DEVICES=0 python main.py --exp=CIFAR10_supervised_NIN # 
# Use K=1000 training examples per category.
CUDA_VISIBLE_DEVICES=0 python main.py --exp=CIFAR10_supervised_NIN_K1000
# Use K=400 training examples per category.
CUDA_VISIBLE_DEVICES=0 python main.py --exp=CIFAR10_supervised_NIN_K400
# Use K=100 training examples per category.
CUDA_VISIBLE_DEVICES=0 python main.py --exp=CIFAR10_supervised_NIN_K100
# Use K=20 training examples per category.
CUDA_VISIBLE_DEVICES=0 python main.py --exp=CIFAR10_supervised_NIN_K20
