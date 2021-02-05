from __future__ import division
from __future__ import print_function

import argparse
import torch
from torch.autograd import gradcheck

from lltm import LLTMFunction

parser = argparse.ArgumentParser()
parser.add_argument('-b', '--batch-size', type=int, default=3)
parser.add_argument('-f', '--features', type=int, default=17)
parser.add_argument('-s', '--state-size', type=int, default=5)
options = parser.parse_args()

device = torch.device("cuda") 

kwargs = {'dtype': torch.float64,
          'device': device,
          'requires_grad': True}

X = torch.randn(options.batch_size, options.features)
h = torch.randn(options.batch_size, options.state_size)
C = torch.randn(options.batch_size, options.state_size)
W = torch.randn(3 * options.state_size, options.features + options.state_size)
b = torch.randn(1, 3 * options.state_size)

variables = [X, W, b, h, C]


if gradcheck(LLTMFunction.apply, variables):
    print('Ok')
