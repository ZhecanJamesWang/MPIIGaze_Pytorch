import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class Model(nn.Module):
    def __init__(self, weight_file):
        super(Model, self).__init__()
        # global __weights_dict
        self.__weights_dict = np.load(weight_file, encoding='bytes').item()


        self.data_bn = self.__batch_normalization(2, 'data_bn', num_features=3, eps=9.99999974738e-06, momentum=0.0)
        self.conv1 = self.__conv(2, name='conv1', in_channels=3, out_channels=64, kernel_size=(7, 7), stride=(2, 2),
                                 groups=1, bias=True)
        self.conv1_bn = self.__batch_normalization(2, 'conv1_bn', num_features=64, eps=9.99999974738e-06, momentum=0.0)
        self.layer_64_1_conv1 = self.__conv(2, name='layer_64_1_conv1', in_channels=64, out_channels=64,
                                            kernel_size=(3, 3), stride=(1, 1), groups=1, bias=False)
        self.layer_64_1_bn2 = self.__batch_normalization(2, 'layer_64_1_bn2', num_features=64, eps=9.99999974738e-06,
                                                         momentum=0.0)
        self.layer_64_1_conv2 = self.__conv(2, name='layer_64_1_conv2', in_channels=64, out_channels=64,
                                            kernel_size=(3, 3), stride=(1, 1), groups=1, bias=False)
        self.layer_128_1_bn1 = self.__batch_normalization(2, 'layer_128_1_bn1', num_features=64, eps=9.99999974738e-06,
                                                          momentum=0.0)
        self.layer_128_1_conv1 = self.__conv(2, name='layer_128_1_conv1', in_channels=64, out_channels=128,
                                             kernel_size=(3, 3), stride=(2, 2), groups=1, bias=False)
        self.layer_128_1_conv_expand = self.__conv(2, name='layer_128_1_conv_expand', in_channels=64, out_channels=128,
                                                   kernel_size=(1, 1), stride=(2, 2), groups=1, bias=False)
        self.layer_128_1_bn2 = self.__batch_normalization(2, 'layer_128_1_bn2', num_features=128, eps=9.99999974738e-06,
                                                          momentum=0.0)
        self.layer_128_1_conv2 = self.__conv(2, name='layer_128_1_conv2', in_channels=128, out_channels=128,
                                             kernel_size=(3, 3), stride=(1, 1), groups=1, bias=False)
        self.layer_256_1_bn1 = self.__batch_normalization(2, 'layer_256_1_bn1', num_features=128, eps=9.99999974738e-06,
                                                          momentum=0.0)
        self.layer_256_1_conv1 = self.__conv(2, name='layer_256_1_conv1', in_channels=128, out_channels=256,
                                             kernel_size=(3, 3), stride=(2, 2), groups=1, bias=False)
        self.layer_256_1_conv_expand = self.__conv(2, name='layer_256_1_conv_expand', in_channels=128, out_channels=256,
                                                   kernel_size=(1, 1), stride=(2, 2), groups=1, bias=False)
        self.layer_256_1_bn2 = self.__batch_normalization(2, 'layer_256_1_bn2', num_features=256, eps=9.99999974738e-06,
                                                          momentum=0.0)
        self.layer_256_1_conv2 = self.__conv(2, name='layer_256_1_conv2', in_channels=256, out_channels=256,
                                             kernel_size=(3, 3), stride=(1, 1), groups=1, bias=False)
        self.layer_512_1_bn1 = self.__batch_normalization(2, 'layer_512_1_bn1', num_features=256, eps=9.99999974738e-06,
                                                          momentum=0.0)
        self.layer_512_1_conv_expand = self.__conv(2, name='layer_512_1_conv_expand', in_channels=256, out_channels=512,
                                                   kernel_size=(1, 1), stride=(2, 2), groups=1, bias=False)
        self.layer_512_1_conv1 = self.__conv(2, name='layer_512_1_conv1', in_channels=256, out_channels=512,
                                             kernel_size=(3, 3), stride=(2, 2), groups=1, bias=False)
        self.layer_512_1_bn2 = self.__batch_normalization(2, 'layer_512_1_bn2', num_features=512, eps=9.99999974738e-06,
                                                          momentum=0.0)
        self.layer_512_1_conv2 = self.__conv(2, name='layer_512_1_conv2', in_channels=512, out_channels=512,
                                             kernel_size=(3, 3), stride=(1, 1), groups=1, bias=False)
        self.last_bn = self.__batch_normalization(2, 'last_bn', num_features=512, eps=9.99999974738e-06, momentum=0.0)
        self.score_1 = self.__dense(name='score_1', in_features=512, out_features=1000, bias=True)
        self.fc1 = self.__dense(name='fc1', in_features=512, out_features=256, bias=True)
        self.fc2 = self.__dense(name='fc2', in_features=258, out_features=128, bias=True)
        self.fc3 = self.__dense(name='fc3', in_features=128, out_features=2, bias=True)




    def forward(self, x):
        data_bn = self.data_bn(x)
        conv1_pad = F.pad(data_bn, (3, 3, 3, 3))
        conv1 = self.conv1(conv1_pad)
        conv1_bn = self.conv1_bn(conv1)
        conv1_relu = F.relu(conv1_bn)
        conv1_pool_pad = F.pad(conv1_relu, (0, 1, 0, 1), value=float('-inf'))
        conv1_pool = F.max_pool2d(conv1_pool_pad, kernel_size=(3, 3), stride=(2, 2), padding=0, ceil_mode=False)
        layer_64_1_conv1_pad = F.pad(conv1_pool, (1, 1, 1, 1))
        layer_64_1_conv1 = self.layer_64_1_conv1(layer_64_1_conv1_pad)
        layer_64_1_bn2 = self.layer_64_1_bn2(layer_64_1_conv1)
        layer_64_1_relu2 = F.relu(layer_64_1_bn2)
        layer_64_1_conv2_pad = F.pad(layer_64_1_relu2, (1, 1, 1, 1))
        layer_64_1_conv2 = self.layer_64_1_conv2(layer_64_1_conv2_pad)
        layer_64_1_sum = layer_64_1_conv2 + conv1_pool
        layer_128_1_bn1 = self.layer_128_1_bn1(layer_64_1_sum)
        layer_128_1_relu1 = F.relu(layer_128_1_bn1)
        layer_128_1_conv1_pad = F.pad(layer_128_1_relu1, (1, 1, 1, 1))
        layer_128_1_conv1 = self.layer_128_1_conv1(layer_128_1_conv1_pad)
        layer_128_1_conv_expand = self.layer_128_1_conv_expand(layer_128_1_relu1)
        layer_128_1_bn2 = self.layer_128_1_bn2(layer_128_1_conv1)
        layer_128_1_relu2 = F.relu(layer_128_1_bn2)
        layer_128_1_conv2_pad = F.pad(layer_128_1_relu2, (1, 1, 1, 1))
        layer_128_1_conv2 = self.layer_128_1_conv2(layer_128_1_conv2_pad)
        layer_128_1_sum = layer_128_1_conv2 + layer_128_1_conv_expand
        layer_256_1_bn1 = self.layer_256_1_bn1(layer_128_1_sum)
        layer_256_1_relu1 = F.relu(layer_256_1_bn1)
        layer_256_1_conv1_pad = F.pad(layer_256_1_relu1, (1, 1, 1, 1))
        layer_256_1_conv1 = self.layer_256_1_conv1(layer_256_1_conv1_pad)
        layer_256_1_conv_expand = self.layer_256_1_conv_expand(layer_256_1_relu1)
        layer_256_1_bn2 = self.layer_256_1_bn2(layer_256_1_conv1)
        layer_256_1_relu2 = F.relu(layer_256_1_bn2)
        layer_256_1_conv2_pad = F.pad(layer_256_1_relu2, (1, 1, 1, 1))
        layer_256_1_conv2 = self.layer_256_1_conv2(layer_256_1_conv2_pad)
        layer_256_1_sum = layer_256_1_conv2 + layer_256_1_conv_expand
        layer_512_1_bn1 = self.layer_512_1_bn1(layer_256_1_sum)
        layer_512_1_relu1 = F.relu(layer_512_1_bn1)
        layer_512_1_conv_expand = self.layer_512_1_conv_expand(layer_512_1_relu1)
        layer_512_1_conv1_pad = F.pad(layer_512_1_relu1, (1, 1, 1, 1))
        layer_512_1_conv1 = self.layer_512_1_conv1(layer_512_1_conv1_pad)
        layer_512_1_bn2 = self.layer_512_1_bn2(layer_512_1_conv1)
        layer_512_1_relu2 = F.relu(layer_512_1_bn2)
        layer_512_1_conv2_pad = F.pad(layer_512_1_relu2, (1, 1, 1, 1))
        layer_512_1_conv2 = self.layer_512_1_conv2(layer_512_1_conv2_pad)
        layer_512_1_sum = layer_512_1_conv2 + layer_512_1_conv_expand
        last_bn = self.last_bn(layer_512_1_sum)
        last_relu = F.relu(last_bn)
        global_pool = F.avg_pool2d(last_relu, kernel_size=(7, 7), stride=(1, 1), padding=(0,), ceil_mode=False)
        score_0 = global_pool.view(global_pool.size(0), -1)
        # score_1 = self.score_1(score_0)
        # prob = F.softmax(score_1)
        # return prob
        x = self.fc1(score_0)
        x = torch.cat([x, y], dim=1)
        x = self.fc2(x)
        x = self.fc3(x)
        x = self.fc4(x)


    # @staticmethod
    def __batch_normalization(self, dim, name, **kwargs):
        if   dim == 1:  layer = nn.BatchNorm1d(**kwargs)
        elif dim == 2:  layer = nn.BatchNorm2d(**kwargs)
        elif dim == 3:  layer = nn.BatchNorm3d(**kwargs)
        else:           raise NotImplementedError()

        if 'scale' in self.__weights_dict[name]:
            layer.state_dict()['weight'].copy_(torch.from_numpy(self.__weights_dict[name]['scale']))
        else:
            layer.weight.data.fill_(1)

        if 'bias' in self.__weights_dict[name]:
            layer.state_dict()['bias'].copy_(torch.from_numpy(self.__weights_dict[name]['bias']))
        else:
            layer.bias.data.fill_(0)

        layer.state_dict()['running_mean'].copy_(torch.from_numpy(self.__weights_dict[name]['mean']))
        layer.state_dict()['running_var'].copy_(torch.from_numpy(self.__weights_dict[name]['var']))
        return layer

    # @staticmethod
    def __conv(self, dim, name, **kwargs):
        if   dim == 1:  layer = nn.Conv1d(**kwargs)
        elif dim == 2:  layer = nn.Conv2d(**kwargs)
        elif dim == 3:  layer = nn.Conv3d(**kwargs)
        else:           raise NotImplementedError()

        layer.state_dict()['weight'].copy_(torch.from_numpy(self.__weights_dict[name]['weights']))
        if 'bias' in self.__weights_dict[name]:
            layer.state_dict()['bias'].copy_(torch.from_numpy(self.__weights_dict[name]['bias']))
        return layer

    # @staticmethod
    def __dense(self, name, **kwargs):
        print "loading: ", name
        layer = nn.Linear(**kwargs)
        if name in self.__weights_dict:
            layer.state_dict()['weight'].copy_(torch.from_numpy(self.__weights_dict[name]['weights']))
        if 'bias' in self.__weights_dict[name]:
            layer.state_dict()['bias'].copy_(torch.from_numpy(self.__weights_dict[name]['bias']))
        return layer


if __name__ == '__main__':
    Model("resnet10_weights.npy")
