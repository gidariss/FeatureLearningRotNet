import torch
import torchvision
import torchvision.transforms as transforms

import os
import imp
import argparse
import numpy as np
from PIL import Image
import caffe

def pytorch_init_network(net_def_file, opt):
    return imp.load_source("",net_def_file).create_model(opt)

def pytorch_load_pretrained(network, pretrained_path):
    print('==> Load pretrained parameters from file %s:' % (pretrained_path))
    assert(os.path.isfile(pretrained_path))
    pretrained_model = torch.load(pretrained_path)
    network.load_state_dict(pretrained_model['network'])

def merge_linear_with_bn(linearlayer, batchnorm):
    filters = linearlayer.weight.data    
    bias    = linearlayer.bias.data if (linearlayer.bias is not None) else 0
    
    epsilon = batchnorm.eps
    gamma   = batchnorm.weight.data
    beta    = batchnorm.bias.data
    mu      = batchnorm.running_mean
    var     = batchnorm.running_var
    
    scale_f = gamma / torch.sqrt(var+epsilon)
    print('==> Merge linear filters of size {} with batch norm layer'.format(filters.size()))
    if filters.dim() == 4:
        mergedfilters = filters.clone() * scale_f.view(-1, 1, 1, 1).expand_as(filters)
    else:
        assert(filters.dim()==2)
        mergedfilters = filters.clone() * scale_f.view(-1, 1).expand_as(filters)
    
    mergedbias    = (bias-mu) * scale_f + beta
    
    return mergedfilters, mergedbias

def copy_weights_to_linear_layer(linearlayer, weights, bias):
    linearlayer.weight.data.copy_(weights)
    linearlayer.bias.data.copy_(bias)

def copy_params_to_the_no_bn_net(net_with_bn, net_without_bn):
    net_with_bn_layers_dict = dict(zip(net_with_bn.all_feat_names,
                                       net_with_bn._feature_blocks))
    net_without_bn_layers_dict = dict(zip(net_without_bn.all_feat_names,
                                          net_without_bn._feature_blocks))

    # 1st conv. block
    mfilters, mbias = merge_linear_with_bn(net_with_bn_layers_dict['conv1'][0],
                                           net_with_bn_layers_dict['conv1'][1])
    copy_weights_to_linear_layer(net_without_bn_layers_dict['conv1'][0],
                                 mfilters, mbias)
    # 2nd conv. block
    mfilters, mbias = merge_linear_with_bn(net_with_bn_layers_dict['conv2'][0],
                                           net_with_bn_layers_dict['conv2'][1])
    copy_weights_to_linear_layer(net_without_bn_layers_dict['conv2'][0],
                                 mfilters, mbias)

    # 3rd conv. block
    mfilters, mbias = merge_linear_with_bn(net_with_bn_layers_dict['conv3'][0],
                                           net_with_bn_layers_dict['conv3'][1])
    copy_weights_to_linear_layer(net_without_bn_layers_dict['conv3'][0],
                                 mfilters, mbias)

    # 4th conv. block
    mfilters, mbias = merge_linear_with_bn(net_with_bn_layers_dict['conv4'][0],
                                           net_with_bn_layers_dict['conv4'][1])
    copy_weights_to_linear_layer(net_without_bn_layers_dict['conv4'][0],
                                 mfilters, mbias)
    
    # 5th conv. block
    mfilters, mbias = merge_linear_with_bn(net_with_bn_layers_dict['conv5'][0],
                                           net_with_bn_layers_dict['conv5'][1])
    copy_weights_to_linear_layer(net_without_bn_layers_dict['conv5'][0],
                                 mfilters, mbias)

    # hidden linear layers
    mfilters, mbias = merge_linear_with_bn(net_with_bn_layers_dict['fc_block'][1],
                                           net_with_bn_layers_dict['fc_block'][2])
    copy_weights_to_linear_layer(net_without_bn_layers_dict['fc_block'][1],
                                 mfilters, mbias)      
    mfilters, mbias = merge_linear_with_bn(net_with_bn_layers_dict['fc_block'][4],
                                           net_with_bn_layers_dict['fc_block'][5])
    copy_weights_to_linear_layer(net_without_bn_layers_dict['fc_block'][3],
                                 mfilters, mbias)

    # classifier
    mfilters = net_with_bn_layers_dict['classifier'][0].weight.data
    mbias = net_with_bn_layers_dict['classifier'][0].bias.data
    copy_weights_to_linear_layer(net_without_bn_layers_dict['classifier'][0],
                                 mfilters, mbias)


def printNetParamSizes(net_caffe):
    for param_name in net_caffe.params.keys():
        print("Parameter {0}".format(param_name))
        for idx, param in enumerate(net_caffe.params[param_name]):
            print("\t Shape[{0}] : {1}".format(idx, param.data.shape))
 
def print_data_stats(adata):
    adata = np.abs(adata)
    print("Abs: [mu: {0}, std: {1}, min: {2} max: {3}]".format(
          adata.mean(), adata.std(), adata.min(), adata.max() ))

    
def copy_params_pytorch2caffe(net_pytorch, net_caffe, std_pixel_values):

    net_pytorch_layers_dict = dict(zip(net_pytorch.all_feat_names,
                                       net_pytorch._feature_blocks))

    # 1st conv. block
    filters = net_pytorch_layers_dict['conv1'][0].weight.data.cpu().numpy()
    bias = net_pytorch_layers_dict['conv1'][0].bias.data.cpu().numpy()
    print("Layer {0} copy params with shape: {1} and {2}".format(
          'conv1', filters.shape, bias.shape))
    net_caffe.params['conv1'][0].data[:,:,:,:] = filters
    net_caffe.params['conv1'][1].data[:] = bias
    
    assert(net_caffe.params['conv1'][0].data.shape[1] == 3)
    net_caffe.params['conv1'][0].data[:,0,:,:] /= (std_pixel_values[0]*255.0)
    net_caffe.params['conv1'][0].data[:,1,:,:] /= (std_pixel_values[1]*255.0)
    net_caffe.params['conv1'][0].data[:,2,:,:] /= (std_pixel_values[2]*255.0)
        
    filters = net_caffe.params['conv1'][0].data.copy()
    net_caffe.params['conv1'][0].data[:,:,:,:] = filters[:,::-1,:,:]
    
    print("Conv 1 weight:")
    print_data_stats(net_caffe.params['conv1'][0].data)
    print("Conv 1 bias:")
    print_data_stats(net_caffe.params['conv1'][1].data)
    
    #print(net_caffe.params['conv1'][0].data.mean())
    
    # 2nd conv. block
    filters = net_pytorch_layers_dict['conv2'][0].weight.data.cpu().numpy()
    bias = net_pytorch_layers_dict['conv2'][0].bias.data.cpu().numpy()
    print("Layer {0} copy params with shape: {1} and {2}".format(
          'conv2', filters.shape, bias.shape))
    net_caffe.params['conv2'][0].data[:,:,:,:] = filters
    net_caffe.params['conv2'][1].data[:] = bias 
    
    print("Conv 2 weight:")
    print_data_stats(net_caffe.params['conv2'][0].data)
    print("Conv 2 bias:")
    print_data_stats(net_caffe.params['conv2'][1].data)    
    
    # 3rd conv. block
    filters = net_pytorch_layers_dict['conv3'][0].weight.data.cpu().numpy()
    bias = net_pytorch_layers_dict['conv3'][0].bias.data.cpu().numpy()
    print("Layer {0} copy params with shape: {1} and {2}".format(
          'conv3', filters.shape, bias.shape))    
    net_caffe.params['conv3'][0].data[:,:,:,:] = filters
    net_caffe.params['conv3'][1].data[:] = bias     
 
    print("Conv 3 weight:")
    print_data_stats(net_caffe.params['conv3'][0].data)
    print("Conv 3 bias:")
    print_data_stats(net_caffe.params['conv3'][1].data)

    # 4th conv. block
    filters = net_pytorch_layers_dict['conv4'][0].weight.data.cpu().numpy()
    bias = net_pytorch_layers_dict['conv4'][0].bias.data.cpu().numpy()
    print("Layer {0} copy params with shape: {1} and {2}".format(
          'conv4', filters.shape, bias.shape))        
    net_caffe.params['conv4'][0].data[:,:,:,:] = filters
    net_caffe.params['conv4'][1].data[:] = bias     
    
    print("Conv 4 weight:")
    print_data_stats(net_caffe.params['conv4'][0].data)
    print("Conv 4 bias:")
    print_data_stats(net_caffe.params['conv4'][1].data)    
    
    # 5th conv. block
    filters = net_pytorch_layers_dict['conv5'][0].weight.data.cpu().numpy()
    bias = net_pytorch_layers_dict['conv5'][0].bias.data.cpu().numpy()
    print("Layer {0} copy params with shape: {1} and {2}".format(
          'conv5', filters.shape, bias.shape))            
    net_caffe.params['conv5'][0].data[:,:,:,:] = filters
    net_caffe.params['conv5'][1].data[:] = bias 

    print("Conv 5 weight:")
    print_data_stats(net_caffe.params['conv5'][0].data)
    print("Conv 5 bias:")
    print_data_stats(net_caffe.params['conv5'][1].data)      
    
    # fully connected: fc6 
    filters = net_pytorch_layers_dict['fc_block'][1].weight.data.cpu().numpy()
    bias = net_pytorch_layers_dict['fc_block'][1].bias.data.cpu().numpy()
    print("Layer {0} copy params with shape: {1} and {2}".format(
          'fc6', filters.shape, bias.shape))
    net_caffe.params['fc6'][0].data[:,:] = filters
    net_caffe.params['fc6'][1].data[:] = bias     

    # fully connected: fc7
    filters = net_pytorch_layers_dict['fc_block'][3].weight.data.cpu().numpy()
    bias = net_pytorch_layers_dict['fc_block'][3].bias.data.cpu().numpy()
    print("Layer {0} copy params with shape: {1} and {2}".format(
          'fc7', filters.shape, bias.shape))                
    net_caffe.params['fc7'][0].data[:,:] = filters
    net_caffe.params['fc7'][1].data[:] = bias


class normalize_numpy(object):
    def __init__(self, mean, std):
        self.mean = mean
        self.std = std

    def __call__(self, array):
        assert(len(array.shape)==3)
        assert(array.shape[0] == len(self.mean))
        assert(array.shape[0] == len(self.std))
        for c in range(array.shape[0]):
            array[c,:,:] -= self.mean[c]
            array[c,:,:] /= self.std[c]
        return array


def prind_data_diffs(data1, data2):
    print("Data 1 shape {0} Data 2 shape {1}".format(data1.shape, data2.shape))
    abs_diff = np.abs(data1 - data2)
    max_diff = abs_diff.max()
    mu_diff  = abs_diff.mean()  
    print("Max diff {0} Mu diff {1} Max1 {2} Max2 {3} Mu1 {4} Mu2 {5}".format(
           max_diff, mu_diff, data1.max(), data2.max(), data1.mean(), data2.mean()))

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--src', type=str, required=True, default='', help='Source location of the Alexnet model saved in pytorch format')
    parser.add_argument('--dst', type=str, required=True, default='', help='Destination location where the Alexnet model will be saved in caffe format')
    args_opt = parser.parse_args()

    pytorch_alexnet_def = './architectures/AlexNet.py'
    src_pytorch_alexnet_params = args_opt.src
    pytorch_alexnet_without_bn_def = './extras/AlexNet_without_BN.py'
    dst_caffe_alexnet_params = args_opt.dst

    net_opt = {'num_classes': 4}
    net_with_bn = pytorch_init_network(pytorch_alexnet_def, net_opt)
    net_without_bn = pytorch_init_network(pytorch_alexnet_without_bn_def, net_opt)
    pytorch_load_pretrained(net_with_bn, src_pytorch_alexnet_params)
    net_with_bn.eval()
    net_without_bn.eval()

    # Merge the batch norm layers to the linear layers
    copy_params_to_the_no_bn_net(net_with_bn, net_without_bn)


    caffe.set_mode_cpu()
    caffe_model_def = './extras/AlexNet.prototxt'
    net_caffe = caffe.Net(caffe_model_def, caffe.TEST)

    mean_pix = [0.485, 0.456, 0.406]
    std_pix = [0.229, 0.224, 0.225]
    mean_pix_caffe = [0.406*255.0, 0.456*255.0, 0.485*255.0]

    # Copy weights to the caffe model
    net_pytorch = net_without_bn
    copy_params_pytorch2caffe(net_pytorch, net_caffe, std_pix)

    transform = []
    transform.append(transforms.Scale(256))
    transform.append(transforms.CenterCrop(224))
    transform.append(lambda x: np.asarray(x))
    transform = transforms.Compose(transform)        

    torch_transform = [transforms.ToTensor(), transforms.Normalize(mean=mean_pix, std=std_pix)]
    torch_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=mean_pix, std=std_pix),
        lambda x: x.view([1,] + list(x.size())),
    ])

    caffe_transform = transforms.Compose([
        lambda x: x.transpose(2,0,1),
        lambda x: x.astype(np.float),
        lambda x: x[::-1,:,:],
        normalize_numpy(mean=mean_pix_caffe, std=[1.0, 1.0, 1.0]),
        lambda x: x.reshape((1,)+x.shape),
    ])

    image_file = './extras/cat.jpg'
    img = transform(Image.open(image_file))
    img_torch = torch_transform(img)
    img_caffe = caffe_transform(img)

    net_caffe.blobs['data'].data[...] = img_caffe
    out_caffe = net_caffe.forward()
    
    img_torch_var = torch.autograd.Variable(img_torch, volatile=True)
    out_pytorch = net_pytorch(img_torch_var, ['fc_block',])

    data_caffe = out_caffe['fc7'].copy()
    data_pytorch = out_pytorch.data.cpu().numpy()
    abs_diff = np.abs(data_pytorch - data_caffe)
    max_diff = abs_diff.max()

    print('==> Maximum data elements difference between torch and caffe: {}'.format(max_diff))

    if not os.path.isfile(dst_caffe_alexnet_params):
        print('==> Saving caffe alexnet model at {0}'.format(dst_caffe_alexnet_params))
        net_caffe.save(dst_caffe_alexnet_params)
    else:
        print('==> File {0} already exists'.format(dst_caffe_alexnet_params))

