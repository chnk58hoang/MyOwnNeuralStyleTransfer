from torch import nn
from torchvision.models import vgg19, VGG19_Weights
from collections import namedtuple
import torch


class VGG19_Model(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        # Get the feature module of the pretrained vgg19
        vgg_pretrained_features = vgg19(VGG19_Weights.DEFAULT).features

        self.layer_names = ['relu1_1', 'relu2_1', 'relu3_1', 'relu4_1', 'conv4_2', 'relu5_1']

        self.content_feature_map_index = 4

        self.style_feature_maps_indices = list(range(len(self.layer_names)))
        self.style_feature_maps_indices.remove(self.content_feature_map_index)
        self.offset = 1

        self.slice1 = torch.nn.Sequential()
        self.slice2 = torch.nn.Sequential()
        self.slice3 = torch.nn.Sequential()
        self.slice4 = torch.nn.Sequential()
        self.slice5 = torch.nn.Sequential()
        self.slice6 = torch.nn.Sequential()

        for x in range(1 + self.offset):
            self.slice1.add_module(str(x), vgg_pretrained_features[x])
        for x in range(1 + self.offset, 6 + self.offset):
            self.slice2.add_module(str(x), vgg_pretrained_features[x])
        for x in range(6 + self.offset, 11 + self.offset):
            self.slice3.add_module(str(x), vgg_pretrained_features[x])
        for x in range(11 + self.offset, 20 + self.offset):
            self.slice4.add_module(str(x), vgg_pretrained_features[x])
        for x in range(20 + self.offset, 22):
            self.slice5.add_module(str(x), vgg_pretrained_features[x])
        for x in range(22, 29 + self.offset):
            self.slice6.add_module(str(x), vgg_pretrained_features[x])

        for param in self.parameters():
            param.requires_grad = False

    def forward(self, x):
        x = self.slice1(x)
        layer1_1 = x
        x = self.slice2(x)
        layer2_1 = x
        x = self.slice3(x)
        layer3_1 = x
        x = self.slice4(x)
        layer4_1 = x
        x = self.slice5(x)
        conv4_2 = x
        x = self.slice6(x)
        layer5_1 = x
        vgg_outputs = namedtuple("VggOutputs", self.layer_names)
        out = vgg_outputs(layer1_1, layer2_1, layer3_1, layer4_1, conv4_2, layer5_1)
        return out
