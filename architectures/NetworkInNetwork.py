import math
import torch
import torch.nn as nn
import torch.nn.functional as F

class BasicBlock(nn.Module):
    def __init__(self, in_planes, out_planes, kernel_size):
        super(BasicBlock, self).__init__()
        padding = (kernel_size-1)/2
        self.layers = nn.Sequential()
        self.layers.add_module('Conv', nn.Conv2d(in_planes, out_planes, \
            kernel_size=kernel_size, stride=1, padding=padding, bias=False))
        self.layers.add_module('BatchNorm', nn.BatchNorm2d(out_planes))
        self.layers.add_module('ReLU',      nn.ReLU(inplace=True))

    def forward(self, x):
        return self.layers(x)

        feat = F.avg_pool2d(feat, feat.size(3)).view(-1, self.nChannels)

class GlobalAveragePooling(nn.Module):
    def __init__(self):
        super(GlobalAveragePooling, self).__init__()

    def forward(self, feat):
        num_channels = feat.size(1)
        return F.avg_pool2d(feat, (feat.size(2), feat.size(3))).view(-1, num_channels)

class NetworkInNetwork(nn.Module):
    def __init__(self, opt):
        super(NetworkInNetwork, self).__init__()

        num_classes = opt['num_classes']
        num_inchannels = opt['num_inchannels'] if ('num_inchannels' in opt) else 3
        num_stages = opt['num_stages'] if ('num_stages' in opt) else 3
        use_avg_on_conv3 = opt['use_avg_on_conv3'] if ('use_avg_on_conv3' in opt) else True


        assert(num_stages >= 3)
        nChannels  = 192
        nChannels2 = 160
        nChannels3 = 96

        blocks = [nn.Sequential() for i in range(num_stages)]
        # 1st block
        blocks[0].add_module('Block1_ConvB1', BasicBlock(num_inchannels, nChannels, 5))
        blocks[0].add_module('Block1_ConvB2', BasicBlock(nChannels,  nChannels2, 1))
        blocks[0].add_module('Block1_ConvB3', BasicBlock(nChannels2, nChannels3, 1))
        blocks[0].add_module('Block1_MaxPool', nn.MaxPool2d(kernel_size=3,stride=2,padding=1))

        # 2nd block
        blocks[1].add_module('Block2_ConvB1',  BasicBlock(nChannels3, nChannels, 5))
        blocks[1].add_module('Block2_ConvB2',  BasicBlock(nChannels,  nChannels, 1))
        blocks[1].add_module('Block2_ConvB3',  BasicBlock(nChannels,  nChannels, 1))
        blocks[1].add_module('Block2_AvgPool', nn.AvgPool2d(kernel_size=3,stride=2,padding=1))

        # 3rd block
        blocks[2].add_module('Block3_ConvB1',  BasicBlock(nChannels, nChannels, 3))
        blocks[2].add_module('Block3_ConvB2',  BasicBlock(nChannels, nChannels, 1))
        blocks[2].add_module('Block3_ConvB3',  BasicBlock(nChannels, nChannels, 1))

        if num_stages > 3 and use_avg_on_conv3:
            blocks[2].add_module('Block3_AvgPool', nn.AvgPool2d(kernel_size=3,stride=2,padding=1))
        for s in range(3, num_stages):
            blocks[s].add_module('Block'+str(s+1)+'_ConvB1',  BasicBlock(nChannels, nChannels, 3))
            blocks[s].add_module('Block'+str(s+1)+'_ConvB2',  BasicBlock(nChannels, nChannels, 1))
            blocks[s].add_module('Block'+str(s+1)+'_ConvB3',  BasicBlock(nChannels, nChannels, 1))

        # global average pooling and classifier
        blocks.append(nn.Sequential())
        blocks[-1].add_module('GlobalAveragePooling',  GlobalAveragePooling())
        blocks[-1].add_module('Classifier', nn.Linear(nChannels, num_classes))

        self._feature_blocks = nn.ModuleList(blocks)
        self.all_feat_names = ['conv'+str(s+1) for s in range(num_stages)] + ['classifier',]
        assert(len(self.all_feat_names) == len(self._feature_blocks))

    def _parse_out_keys_arg(self, out_feat_keys):

    	# By default return the features of the last layer / module.
    	out_feat_keys = [self.all_feat_names[-1],] if out_feat_keys is None else out_feat_keys

        if len(out_feat_keys) == 0:
    		raise ValueError('Empty list of output feature keys.')
        for f, key in enumerate(out_feat_keys):
    		if key not in self.all_feat_names:
    			raise ValueError('Feature with name {0} does not exist. Existing features: {1}.'.format(key, self.all_feat_names))
    		elif key in out_feat_keys[:f]:
    			raise ValueError('Duplicate output feature key: {0}.'.format(key))

    	# Find the highest output feature in `out_feat_keys
    	max_out_feat = max([self.all_feat_names.index(key) for key in out_feat_keys])

    	return out_feat_keys, max_out_feat

    def forward(self, x, out_feat_keys=None):
    	"""Forward an image `x` through the network and return the asked output features.

    	Args:
    	  x: input image.
    	  out_feat_keys: a list/tuple with the feature names of the features
                that the function should return. By default the last feature of
                the network is returned.

    	Return:
            out_feats: If multiple output features were asked then `out_feats`
                is a list with the asked output features placed in the same
                order as in `out_feat_keys`. If a single output feature was
                asked then `out_feats` is that output feature (and not a list).
    	"""
    	out_feat_keys, max_out_feat = self._parse_out_keys_arg(out_feat_keys)
    	out_feats = [None] * len(out_feat_keys)

    	feat = x
    	for f in range(max_out_feat+1):
    		feat = self._feature_blocks[f](feat)
    		key = self.all_feat_names[f]
    		if key in out_feat_keys:
    			out_feats[out_feat_keys.index(key)] = feat

        out_feats = out_feats[0] if len(out_feats)==1 else out_feats
    	return out_feats


    def weight_initialization(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                if m.weight.requires_grad:
                    n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                    m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                if m.weight.requires_grad:
                    m.weight.data.fill_(1)
                if m.bias.requires_grad:
                    m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                if m.bias.requires_grad:
                    m.bias.data.zero_()

def create_model(opt):
    return NetworkInNetwork(opt)

if __name__ == '__main__':
    size = 32
    opt = {'num_classes':4, 'num_stages': 5}

    net = create_model(opt)
    x = torch.autograd.Variable(torch.FloatTensor(1,3,size,size).uniform_(-1,1))

    out = net(x, out_feat_keys=net.all_feat_names)
    for f in range(len(out)):
        print('Output feature {0} - size {1}'.format(
            net.all_feat_names[f], out[f].size()))


    out = net(x)
    print('Final output: {0}'.format(out.size()))
