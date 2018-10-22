import imp
import argparse
import numpy as np
from PIL import Image
import math
import caffe
import math
import os

def printNetParamSizes(net):
    for param_name in net.params.keys():
        print("Parameter {0}".format(param_name))
        for idx, param in enumerate(net.params[param_name]):
            print("\t Shape[{0}] : {1}".format(idx, param.data.shape))
 
def print_data_stats(adata):
    adata = np.abs(adata)
    print("Abs: [mu: {0}, std: {1}, min: {2} max: {3}]".format(\
        adata.mean(), adata.std(), adata.min(), adata.max() ))

def copy_params_fc2fcn(net_fc, net_fcn):
    # 1st conv. block
    filters = net_fc.params['conv1'][0].data
    bias = net_fc.params['conv1'][1].data
    print("Layer {0} copy params with shape: {1} and {2}".format('conv1', filters.shape, bias.shape))
    net_fcn.params['conv1'][0].data[:,:,:,:] = filters
    net_fcn.params['conv1'][1].data[:] = bias
    print("Conv 1 weight:")
    print_data_stats(net_fcn.params['conv1'][0].data)
    print("Conv 1 bias:")
    print_data_stats(net_fcn.params['conv1'][1].data)     
    
    # 2nd conv. block
    filters = net_fc.params['conv2'][0].data
    bias = net_fc.params['conv2'][1].data
    print("Layer {0} copy params with shape: {1} and {2}".format('conv2', filters.shape, bias.shape))
    net_fcn.params['conv2'][0].data[:,:,:,:] = filters
    net_fcn.params['conv2'][1].data[:] = bias 
    print("Conv 2 weight:")
    print_data_stats(net_fcn.params['conv2'][0].data)
    print("Conv 2 bias:")
    print_data_stats(net_fcn.params['conv2'][1].data)    
    
    # 3rd conv. block
    filters = net_fc.params['conv3'][0].data
    bias = net_fc.params['conv3'][1].data    
    print("Layer {0} copy params with shape: {1} and {2}".format('conv3', filters.shape, bias.shape))    
    net_fcn.params['conv3'][0].data[:,:,:,:] = filters
    net_fcn.params['conv3'][1].data[:] = bias     
    print("Conv 3 weight:")
    print_data_stats(net_fcn.params['conv3'][0].data)
    print("Conv 3 bias:")
    print_data_stats(net_fcn.params['conv3'][1].data)

    # 4th conv. block 
    filters = net_fc.params['conv4'][0].data
    bias = net_fc.params['conv4'][1].data      
    print("Layer {0} copy params with shape: {1} and {2}".format('conv4', filters.shape, bias.shape))        
    net_fcn.params['conv4'][0].data[:,:,:,:] = filters
    net_fcn.params['conv4'][1].data[:] = bias     
    print("Conv 4 weight:")
    print_data_stats(net_fcn.params['conv4'][0].data)
    print("Conv 4 bias:")
    print_data_stats(net_fcn.params['conv4'][1].data)    
    
    # 5th conv. block
    filters = net_fc.params['conv5'][0].data
    bias = net_fc.params['conv5'][1].data     
    print("Layer {0} copy params with shape: {1} and {2}".format('conv5', filters.shape, bias.shape))            
    net_fcn.params['conv5'][0].data[:,:,:,:] = filters
    net_fcn.params['conv5'][1].data[:] = bias 
    print("Conv 5 weight:")
    print_data_stats(net_fcn.params['conv5'][0].data)
    print("Conv 5 bias:")
    print_data_stats(net_fcn.params['conv5'][1].data)      

    # 6th conv. block
    filters = net_fc.params['fc6'][0].data
    bias = net_fc.params['fc6'][1].data     
    print("Layer {0} copy params with shape: {1} and {2}".format('fc6', filters.shape, bias.shape))   
    fcn_shape = net_fcn.params['fc6'][0].data.shape
    print(fcn_shape)
    net_fcn.params['fc6'][0].data[:,:,:,:] = filters.reshape(fcn_shape)
    net_fcn.params['fc6'][1].data[:] = bias 
    print("fc 6 weight:")
    print_data_stats(net_fcn.params['fc6'][0].data)
    print("fc 6 bias:")
    print_data_stats(net_fcn.params['fc6'][1].data)     
    
    # 7th conv. block
    filters = net_fc.params['fc7'][0].data
    bias = net_fc.params['fc7'][1].data     
    print("Layer {0} copy params with shape: {1} and {2}".format('fc7', filters.shape, bias.shape))   
    fcn_shape = net_fcn.params['fc7'][0].data.shape
    print(fcn_shape)
    net_fcn.params['fc7'][0].data[:,:,:,:] = filters.reshape(fcn_shape)
    net_fcn.params['fc7'][1].data[:] = bias 
    print("fc 7 weight:")
    print_data_stats(net_fcn.params['fc7'][0].data)
    print("fc 7 bias:")
    print_data_stats(net_fcn.params['fc7'][1].data)



if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--src', type=str, required=True, default='', help='Source location of the Alexnet model params saved in caffe format')
    parser.add_argument('--dst', type=str, required=True, default='', help='Destination location where the Alexnet-fcn model params will be saved in caffe format')
    args_opt = parser.parse_args()

    caffe_alexnet_model_def = './extras/AlexNet.prototxt'
    caffe_alexnet_fcn_model_def = './extras/AlexNet_fcn.prototxt'

    src_caffe_alexnet_params = args_opt.src 
    dst_caffe_alexnet_fcn_params = args_opt.dst

    caffe.set_mode_cpu()
    caffe_model_def_fcn = './caffe_models/AlexNet_NoGroups_NoLRN_FCN.prototxt'
    net_caffe_fcn = caffe.Net(caffe_alexnet_fcn_model_def, caffe.TEST)
    net_caffe = caffe.Net(caffe_alexnet_model_def, src_caffe_alexnet_params, caffe.TEST)

    copy_params_fc2fcn(net_caffe, net_caffe_fcn)

    if not os.path.isfile(dst_caffe_alexnet_fcn_params):
        print('==> Saving caffe alexnet-fcn model at {0}'.format(dst_caffe_alexnet_fcn_params))
        net_caffe_fcn.save(dst_caffe_alexnet_fcn_params)
    else:
        print('==> File {0} already exists'.format(dst_caffe_alexnet_fcn_params))
