batch_size   = 192

config = {}
# set the parameters related to the training and testing set
data_train_opt = {} 
data_train_opt['batch_size'] = batch_size
data_train_opt['unsupervised'] = True
data_train_opt['epoch_size'] = None
data_train_opt['random_sized_crop'] = True
data_train_opt['dataset_name'] = 'imagenet'
data_train_opt['split'] = 'train'

data_test_opt = {}
data_test_opt['batch_size'] = batch_size
data_test_opt['unsupervised'] = True
data_test_opt['epoch_size'] = None
data_test_opt['random_sized_crop'] = False
data_test_opt['dataset_name'] = 'imagenet'
data_test_opt['split'] = 'val'

config['data_train_opt'] = data_train_opt
config['data_test_opt']  = data_test_opt
config['max_num_epochs'] = 50

net_opt = {}
net_opt['num_classes'] = 4
net_opt['num_stages']  = 4

networks = {}
net_optim_params = {'optim_type': 'sgd', 'lr': 0.1, 'momentum':0.9, 'weight_decay': 5e-4, 'nesterov': True, 'LUT_lr':[(15, 0.01),(30, 0.001),(45, 0.0001),(50, 0.00001)]}
networks['model'] = {'def_file': 'architectures/AlexNet.py', 'pretrained': None, 'opt': net_opt,  'optim_params': net_optim_params} 
config['networks'] = networks

criterions = {}
criterions['loss'] = {'ctype':'CrossEntropyLoss', 'opt':None}
config['criterions'] = criterions
config['algorithm_type'] = 'ClassificationModel'
