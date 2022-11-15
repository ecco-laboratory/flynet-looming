# %% 
# imports
# Copied from https://github.com/bryanyzhu/two-stream-pytorch on 2022-11-04

import torch.nn as nn
import torch.utils.model_zoo as model_zoo

import math
import collections
import torch

__all__ = ['VGG', 'flow_vgg16', 'rgb_vgg16', 'rgb_vgg16_bn']


model_urls = {'vgg16': 'https://download.pytorch.org/models/vgg16-397923af.pth'}

# %%
# define VGG class

class VGG(nn.Module):

    def __init__(self, features, type, num_classes=1000):
        super(VGG, self).__init__()
        self.features = features

        # MT: I found these comments in the flow and rgb files respectively
        # But other than these... it looks like the vgg class in both files is the same
        # Change the dropout value to 0.9 and 0.8 for flow model
        # Change the dropout value to 0.9 and 0.9 for rgb model
        final_classifier_dropout = 0.9 if type == 'rgb' else 0.8
        self.classifier = nn.Sequential(
            nn.Linear(512 * 7 * 7, 4096),
            nn.ReLU(True),
            nn.Dropout(p=0.9),
            nn.Linear(4096, 4096),
            nn.ReLU(True),
            nn.Dropout(p=final_classifier_dropout),
        )
        
        self.fc_action = nn.Linear(4096, num_classes)
        self._initialize_weights()

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        x = self.fc_action(x)
        return x

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                n = m.weight.size(1)
                m.weight.data.normal_(0, 0.01)
                m.bias.data.zero_()

# %%
# define helper functions, I think

def make_layers(cfg, type, batch_norm=False):
    layers = []
    # Again, adding this because make_layers otherwise looked the same
    # between flow and rgb vgg16 files
    in_channels = 3 if type == 'rgb' else 20
    for v in cfg:
        if v == 'M':
            layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
        else:
            conv2d = nn.Conv2d(in_channels, v, kernel_size=3, padding=1)
            if batch_norm:
                layers += [conv2d, nn.BatchNorm2d(v), nn.ReLU(inplace=True)]
            else:
                layers += [conv2d, nn.ReLU(inplace=True)]
            in_channels = v
    return nn.Sequential(*layers)


cfg = {
    'A': [64, 'M', 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
    'B': [64, 64, 'M', 128, 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
    'D': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512, 'M', 512, 512, 512, 'M'],
    'E': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 256, 'M', 512, 512, 512, 512, 'M', 512, 512, 512, 512, 'M'],
}

# Only used for the flow model, it looks like
def change_key_names(old_params, in_channels):
    new_params = collections.OrderedDict()
    layer_count = 0
    for layer_key in old_params.keys():
        if layer_count < 26:
            if layer_count == 0:
                # Specifically for the first layer I think it's meant to reshape
                # from 3ch weights to 20ch weights
                rgb_weight = old_params[layer_key]
                rgb_weight_mean = torch.mean(rgb_weight, dim=1)
                # MT: Added this in order to re-insert the channel dimension
                # which gets dropped when meaned
                # so that the listed repeat dims will line up with the actual dims
                rgb_weight_mean = rgb_weight_mean.unsqueeze(dim=1)
                flow_weight = rgb_weight_mean.repeat(1,in_channels,1,1)
                new_params[layer_key] = flow_weight
                layer_count += 1
                # print(layer_key, new_params[layer_key].size())
            else:
                new_params[layer_key] = old_params[layer_key]
                layer_count += 1
                # print(layer_key, new_params[layer_key].size())

    return new_params

# %%
# Define functions to fully build up a model instance

def flow_vgg16(pretrained=False, **kwargs):
    """VGG 16-layer model (configuration "D")

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = VGG(make_layers(cfg['D'], type='flow'), type='flow', **kwargs)
    # TODO: hardcoded for now for 10 optical flow images, set it as an argument later 
    in_channels = 20            
    if pretrained:
        # model.load_state_dict(model_zoo.load_url(model_urls['vgg16']))
        pretrained_dict = model_zoo.load_url(model_urls['vgg16'])
        model_dict = model.state_dict()

        new_pretrained_dict = change_key_names(pretrained_dict, in_channels)
        # 1. filter out unnecessary keys
        new_pretrained_dict = {k: v for k, v in new_pretrained_dict.items() if k in model_dict}
        # 2. overwrite entries in the existing state dict
        model_dict.update(new_pretrained_dict)
        # 3. load the new state dict
        model.load_state_dict(model_dict)

    return model

def rgb_vgg16(pretrained=False, **kwargs):
    """VGG 16-layer model (configuration "D")

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = VGG(make_layers(cfg['D'], type='rgb'), type='rgb', **kwargs)
    if pretrained:
        # model.load_state_dict(model_zoo.load_url(model_urls['vgg16']))
        pretrained_dict = model_zoo.load_url(model_urls['vgg16'])
        model_dict = model.state_dict()

        # 1. filter out unnecessary keys
        pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict}
        # 2. overwrite entries in the existing state dict
        model_dict.update(pretrained_dict) 
        # 3. load the new state dict
        model.load_state_dict(model_dict)

    return model


def rgb_vgg16_bn(**kwargs):
    """VGG 16-layer model (configuration "D") with batch normalization
       No pretrained model available.
    """
    return VGG(make_layers(cfg['D'], type='rgb', batch_norm=True), type='rgb', **kwargs)
