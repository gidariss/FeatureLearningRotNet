batch_size   = 128

config = {}
# set the parameters related to the training and testing set
data_train_opt = {} 
data_train_opt['batch_size'] = batch_size
data_train_opt['unsupervised'] = False
data_train_opt['epoch_size'] = 10 * 5000
data_train_opt['random_sized_crop'] = False
data_train_opt['dataset_name'] = 'cifar10'
data_train_opt['split'] = 'train'
data_train_opt['num_imgs_per_cat'] = 400

data_test_opt = {}
data_test_opt['batch_size'] = batch_size
data_test_opt['unsupervised'] = False
data_test_opt['epoch_size'] = None
data_test_opt['random_sized_crop'] = False
data_test_opt['dataset_name'] = 'cifar10'
data_test_opt['split'] = 'test'

config['data_train_opt'] = data_train_opt
config['data_test_opt']  = data_test_opt
config['max_num_epochs'] = 200

net_opt = {}
net_opt['num_classes'] = 10
net_opt['num_stages']  = 3

networks = {}
net_optim_params = {'optim_type': 'sgd', 'lr': 0.1, 'momentum':0.9, 'weight_decay': 5e-4, 'nesterov': True, 'LUT_lr':[(60, 0.1),(120, 0.02),(160, 0.004),(200, 0.0008)]}
networks['model'] = {'def_file': 'architectures/NetworkInNetwork.py', 'pretrained': None, 'opt': net_opt,  'optim_params': net_optim_params} 
config['networks'] = networks

criterions = {}
criterions['loss'] = {'ctype':'CrossEntropyLoss', 'opt':None}
config['criterions'] = criterions
config['algorithm_type'] = 'ClassificationModel'
