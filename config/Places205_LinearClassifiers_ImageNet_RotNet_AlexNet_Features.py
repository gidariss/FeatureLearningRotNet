batch_size   = 192

config = {}
# set the parameters related to the training and testing set
data_train_opt = {} 
data_train_opt['batch_size'] = batch_size
data_train_opt['unsupervised'] = False
data_train_opt['epoch_size'] = None
data_train_opt['random_sized_crop'] = False
data_train_opt['dataset_name'] = 'places205'
data_train_opt['split'] = 'train'

data_test_opt = {}
data_test_opt['batch_size'] = batch_size
data_test_opt['unsupervised'] = False
data_test_opt['epoch_size'] = None
data_test_opt['random_sized_crop'] = False
data_test_opt['dataset_name'] = 'places205'
data_test_opt['split'] = 'val'

config['data_train_opt'] = data_train_opt
config['data_test_opt']  = data_test_opt
config['max_num_epochs'] = 35


networks = {}

pretrained = './experiments/ImageNet_RotNet_AlexNet/model_net_epoch50'
networks['feat_extractor'] = {'def_file': 'architectures/AlexNet.py', 'pretrained': pretrained, 'opt': {'num_classes': 4},  'optim_params': None} 

net_opt_cls = [None] * 5
net_opt_cls[0] = {'pool_type':'max', 'nChannels':64, 'pool_size':12, 'num_classes': 205}
net_opt_cls[1] = {'pool_type':'max', 'nChannels':192, 'pool_size':7, 'num_classes': 205}
net_opt_cls[2] = {'pool_type':'max', 'nChannels':384, 'pool_size':5, 'num_classes': 205}
net_opt_cls[3] = {'pool_type':'max', 'nChannels':256, 'pool_size':6, 'num_classes': 205}
net_opt_cls[4] = {'pool_type':'max', 'nChannels':256, 'pool_size':6, 'num_classes': 205}
out_feat_keys = ['conv1', 'conv2', 'conv3', 'conv4', 'conv5']
net_optim_params_cls = {'optim_type': 'sgd', 'lr': 0.1, 'momentum':0.9, 'weight_decay': 5e-4, 'nesterov': True, 'LUT_lr':[(5, 0.01),(15, 0.002),(25, 0.0004),(35, 0.00008)]}
networks['classifier']  = {'def_file': 'architectures/MultipleLinearClassifiers.py', 'pretrained': None, 'opt': net_opt_cls, 'optim_params': net_optim_params_cls}

config['networks'] = networks

criterions = {}
criterions['loss'] = {'ctype':'CrossEntropyLoss', 'opt':None}
config['criterions'] = criterions
config['algorithm_type'] = 'FeatureClassificationModel'
config['out_feat_keys'] = out_feat_keys

