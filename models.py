import torch
import torch.nn as nn
import torch.nn.functional as F

from yaml import load, dump

def create_model(filename, n_inputs, n_outputs, weight_initilization='identity'):
    """
    Creates Neural Network model.

    Arguments:
        filename: filename of model configuration
        weight_initilization: weight intilization type
    """

    model = None
    model_info = load(open(filename, 'r'))
    model_type = model_info['name']
    params = model_info['params']
    params[0]['params']['in_channels'] = n_inputs
    params[-1]['params']['out_channels'] = n_outputs
    if 'meshnet' in model_type:
        model = MeshNet(params, weight_initilization=weight_initilization)
    else:
        assert False, 'The model {} isn\'t specifed'.format(model_type)
    print (model)
    return model, params


def weight_init(model, weight_initilization):
    """
    Initialize weights of the Neural Network.

    Arguments:
        model: Neural Network model
        weight_initilization: weight intilization type
    """

    if weight_initilization == 'xavier_uniform':
        for m in model.modules():
            if isinstance(m, nn.Conv3d):
                nn.init.xavier_uniform_(m.weight, gain=nn.init.calculate_gain('relu'))
                nn.init.constant_(m.bias, 0.)
    elif weight_initilization == 'xavier_normal':
        for m in model.modules():
            if isinstance(m, nn.Conv3d):
                nn.init.xavier_normal_(m.weight, gain=nn.init.calculate_gain('relu'))
                nn.init.constant_(m.bias, 0.)
    elif weight_initilization == 'identity':
        for m in model.modules():
            if isinstance(m, nn.Conv3d):
                temp = torch.FloatTensor(m.weight.size())
                nn.init.xavier_uniform_(temp, gain=nn.init.calculate_gain('relu'))
                temp[:, :, 0, 0, 0] += 1
                m.weight = torch.nn.Parameter(temp)
                nn.init.constant_(m.bias, 0.)
    elif weight_initilization == 'kaiming_uniform':
        for m in model.modules():
            if isinstance(m, nn.Conv3d):
                nn.init.kaiming_uniform_(m.weight, nonlinearity='relu')
                nn.init.constant_(m.bias, 0.)
    elif weight_initilization == 'kaiming_normal':
        for m in model.modules():
            if isinstance(m, nn.Conv3d):
                nn.init.kaiming_normal_(m.weight, nonlinearity='relu')
                nn.init.constant_(m.bias, 0.)
    else:
        assert False, '{} initilization isn\'t defined'.format(weight_initilization)


class MeshNet(nn.Module):
    """
    MeshNet Neural Network

    Arguments:
        config: config of the neural network
        bn_before: apply batch normalization before activation function
        weight_initilization: weight intilization type
    """

    def __init__(self, config, bn_before=True, 
        weight_initilization='xavier_uniform'):
        super(MeshNet, self).__init__()
        self.model = nn.Sequential()
        for i, p in enumerate(config):
            if i != len(config) - 1:
                self.model.add_module('conv_{}'.format(i), nn.Conv3d(**p['params']))
                if bn_before:
                    self.model.add_module('bn_{}'.format(i), 
                        nn.BatchNorm3d(p['params']['out_channels']))
                self.model.add_module('relu_{}'.format(i), nn.ReLU(inplace=True))
                if not bn_before:
                    self.model.add_module('bn_{}'.format(i), 
                        nn.BatchNorm3d(p['params']['out_channels']))
                if p['dropout'] > 0:
                    self.model.add_module('dp_{}'.format(i), 
                        nn.Dropout3d(p=p['dropout'], inplace=True))
            else:
                self.model.add_module('conv_{}'.format(i), nn.Conv3d(**p['params']))

        # weight initilization
        weight_init(self.model, weight_initilization)


    def forward(self, x):
        """
        Forward propagation.

        Arguments:
            x: input
        """
        x = self.model(x)
        return x
