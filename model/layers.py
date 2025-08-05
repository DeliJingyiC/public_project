#This code is adopted from
#https://github.com/ivanvovk/WaveGrad
import torch

from model.base import BaseModule


class Conv1dWithInitialization(BaseModule):

    def __init__(self, **kwargs):
        super(Conv1dWithInitialization, self).__init__()
        self.conv1d = torch.nn.Conv1d(**kwargs)
        torch.nn.init.orthogonal_(self.conv1d.weight.data, gain=1)

    def forward(self, x):
        # print('conv signal')
        return torch.nan_to_num(self.conv1d(x), posinf=65504, neginf=-65504)
