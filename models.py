from turtle import forward
import torch
import torch.nn as nn
from typing import Optional


def CalculateOutSize(model, channels, samples):
    '''
    Calculate the output based on input size.
    model is from nn.Module and inputSize is a array.
    '''
    device = next(model.parameters()).device
    x = torch.rand(1, 1, channels, samples).to(device)
    out = model(x)
    return out.shape[-1]


def LoadModel(model_name, Chans, Samples):
    if model_name == 'EEGNet':
        model = EEGNet(Chans=Chans,
                       Samples=Samples,
                       kernLenght=64,
                       F1=4,
                       D=2,
                       F2=8,
                       dropoutRate=0.25)
    elif model_name == 'DeepCNN':
        model = DeepConvNet(Chans=Chans, Samples=Samples, dropoutRate=0.5)
    elif model_name == 'ShallowCNN':
        model = ShallowConvNet(Chans=Chans, Samples=Samples, dropoutRate=0.5)
    else:
        raise 'No such model'
    return model


class EEGNet(nn.Module):
    """
    :param
    """
    def __init__(self,
                 Chans: int,
                 Samples: int,
                 kernLenght: int,
                 F1: int,
                 D: int,
                 F2: int,
                 dropoutRate: Optional[float] = 0.5):
        super(EEGNet, self).__init__()

        self.Chans = Chans
        self.Samples = Samples
        self.kernLenght = kernLenght
        self.F1 = F1
        self.D = D
        self.F2 = F2
        self.dropoutRate = dropoutRate

        self.block1 = nn.Sequential(
            nn.ZeroPad2d((self.kernLenght // 2 - 1,
                          self.kernLenght - self.kernLenght // 2, 0,
                          0)),  # left, right, up, bottom
            nn.Conv2d(in_channels=1,
                      out_channels=self.F1,
                      kernel_size=(1, self.kernLenght),
                      stride=1,
                      bias=False),
            nn.BatchNorm2d(num_features=self.F1),
            # DepthwiseConv2d
            nn.Conv2d(in_channels=self.F1,
                      out_channels=self.F1 * self.D,
                      kernel_size=(self.Chans, 1),
                      groups=self.F1,
                      bias=False),
            nn.BatchNorm2d(num_features=self.F1 * self.D),
            nn.ELU(),
            nn.AvgPool2d((1, 4)),
            nn.Dropout(p=self.dropoutRate))

        self.block2 = nn.Sequential(
            nn.ZeroPad2d((7, 8, 0, 0)),
            # SeparableConv2d
            nn.Conv2d(in_channels=self.F1 * self.D,
                      out_channels=self.F1 * self.D,
                      kernel_size=(1, 16),
                      stride=1,
                      groups=self.F1 * self.D,
                      bias=False),
            nn.BatchNorm2d(num_features=self.F1 * self.D),
            nn.Conv2d(in_channels=self.F1 * self.D,
                      out_channels=self.F2,
                      kernel_size=(1, 1),
                      stride=1,
                      bias=False),
            nn.BatchNorm2d(num_features=self.F2),
            nn.ELU(),
            nn.AvgPool2d((1, 8)),
            nn.Dropout(self.dropoutRate))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        output = self.block1(x)
        output = self.block2(output)
        output = output.reshape(output.size(0), -1)

        return output

    def MaxNormConstraint(self):
        for n, p in self.block1.named_parameters():
            if n == '3.weight':
                p.data = torch.renorm(p.data, p=2, dim=0, maxnorm=1.0)


class DeepConvNet(nn.Module):
    def __init__(self,
                 Chans: int,
                 Samples: int,
                 dropoutRate: Optional[float] = 0.5):
        super(DeepConvNet, self).__init__()

        self.Chans = Chans
        self.Samples = Samples
        self.dropoutRate = dropoutRate

        self.block1 = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=25, kernel_size=(1, 5)),
            nn.Conv2d(in_channels=25, out_channels=25, kernel_size=(Chans, 1)),
            nn.BatchNorm2d(num_features=25), nn.ELU(),
            nn.MaxPool2d(kernel_size=(1, 2), stride=(1, 2)),
            nn.Dropout(self.dropoutRate))

        self.block2 = nn.Sequential(
            nn.Conv2d(in_channels=25, out_channels=50, kernel_size=(1, 5)),
            nn.BatchNorm2d(num_features=50), nn.ELU(),
            nn.MaxPool2d(kernel_size=(1, 2), stride=(1, 2)),
            nn.Dropout(self.dropoutRate))

        self.block3 = nn.Sequential(
            nn.Conv2d(in_channels=50, out_channels=100, kernel_size=(1, 5)),
            nn.BatchNorm2d(num_features=100), nn.ELU(),
            nn.MaxPool2d(kernel_size=(1, 2), stride=(1, 2)),
            nn.Dropout(self.dropoutRate))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        output = self.block1(x)
        output = self.block2(output)
        output = self.block3(output)
        output = output.reshape(output.size(0), -1)
        return output

    def MaxNormConstraint(self):
        for block in [self.block1, self.block2, self.block3]:
            for n, p in block.named_parameters():
                if hasattr(n, 'weight') and (
                        not n.__class__.__name__.startswith('BatchNorm')):
                    p.data = torch.renorm(p.data, p=2, dim=0, maxnorm=2.0)


class Activation(nn.Module):
    def __init__(self, type):
        super(Activation, self).__init__()
        self.type = type

    def forward(self, input):
        if self.type == 'square':
            output = input * input
        elif self.type == 'log':
            output = torch.log(torch.clamp(input, min=1e-6))
        else:
            raise Exception('Invalid type !')

        return output


class ShallowConvNet(nn.Module):
    def __init__(
        self,
        Chans: int,
        Samples: int,
        dropoutRate: Optional[float] = 0.5,
    ):
        super(ShallowConvNet, self).__init__()

        self.Chans = Chans
        self.Samples = Samples
        self.dropoutRate = dropoutRate

        self.block1 = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=40, kernel_size=(1, 13)),
            nn.Conv2d(in_channels=40,
                      out_channels=40,
                      kernel_size=(self.Chans, 1)),
            nn.BatchNorm2d(num_features=40), 
            nn.ELU(), #Activation('square'),
            nn.AvgPool2d(kernel_size=(1, 35), stride=(1, 7)),
            nn.ELU(), # Activation('log'), 
            nn.Dropout(self.dropoutRate))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        output = self.block1(x)
        output = output.reshape(output.size(0), -1)
        return output

    def MaxNormConstraint(self):
        for n, p in self.block1.named_parameters():
            if hasattr(n, 'weight') and (
                    not n.__class__.__name__.startswith('BatchNorm')):
                p.data = torch.renorm(p.data, p=2, dim=0, maxnorm=2.0)


class Classifier(nn.Module):
    def __init__(self, input_dim, n_classes):
        super(Classifier, self).__init__()

        self.input_dim = input_dim
        self.n_classes = n_classes

        self.block = nn.Sequential(
            nn.Linear(in_features=self.input_dim,
                      out_features=self.n_classes,
                      bias=True))

    def forward(self, feature):
        output = self.block(feature)

        return output

    def MaxNormConstraint(self):
        for n, p in self.block.named_parameters():
            if n == '0.weight':
                p.data = torch.renorm(p.data, p=2, dim=0, maxnorm=0.25)


class Discriminator(nn.Module):
    def __init__(self, input_dim, n_subjects):
        super(Discriminator, self).__init__()

        self.input_dim = input_dim
        self.n_subjects = n_subjects

        self.block = nn.Sequential(
            nn.Linear(in_features=self.input_dim, out_features=50, bias=True),
            nn.Linear(in_features=50, out_features=self.n_subjects, bias=True))

    def forward(self, feature):
        output = self.block(feature)

        return output


import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import numpy as np
import random

"""
    Neural Network: CLSTM
    Detail: The input first cross CNN model ,then the output of CNN as the input of LSTM
"""


class LSTM_CNN(nn.Module):
    def __init__(self, channels, input_size, hidden_size, num_layers, 
                 time_kernel_size=16, spatial_num=8, drop_out=0.5):
        super(LSTM_CNN, self).__init__()

        drop_out = 0.5
        self.channels = channels
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        
        self.lstm = nn.LSTM(self.input_size, self.hidden_size, self.num_layers, batch_first=True)
        
        # self.block_1 = nn.Sequential(
        #     nn.ZeroPad2d((time_kernel_size // 2 - 1, time_kernel_size // 2, 0, 0)),
        #     nn.Conv2d(1, 8, (1, time_kernel_size), bias=False),
        #     nn.BatchNorm2d(8),
        #     nn.ELU(),
        #     nn.AvgPool2d((1, 4))
        # )
        
        self.block_2 = nn.Sequential(
            nn.Conv2d(1, spatial_num, (channels, 1), bias=False),
            nn.BatchNorm2d(spatial_num),
            nn.ELU(),
            nn.AvgPool2d((1, 4)),
            nn.Dropout(drop_out)
        )
        
        
    def forward(self, x):
        # x: (N, C, T)  N: batch_size; C: channels; T: times
        x = x.reshape(-1, self.channels, self.input_size)
        N, C, T = x.shape
        x = x.reshape(N * C, T // self.input_size, self.input_size)
        lstm_out, _ = self.lstm(x, None)
        
        # x: (N, 1, C, H)  H: hidden_size
        x = lstm_out[:, -1, :].reshape(N, 1, C, self.hidden_size)
        
        # x = self.block_1(x)
        x = self.block_2(x)
        
        x = x.view(x.size(0), -1)

        return x