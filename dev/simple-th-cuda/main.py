from __future__ import division
from __future__ import print_function

import argparse
import torch
from torch.autograd import gradcheck

parser = argparse.ArgumentParser()
parser.add_argument('-b', '--batch-size', type=int, default=3)
parser.add_argument('-f', '--features', type=int, default=17)
parser.add_argument('-s', '--state-size', type=int, default=5)
parser.add_argument('-c', '--cuda', action='store_true')
options = parser.parse_args()

from lltm import LLTMFunction
options.cuda = True

device = torch.device("cuda")
dtype = torch.float32

kwargs = {'dtype': dtype,
          'device': device,
          'requires_grad': True}
          
X = torch.randn(options.batch_size, options.features, **kwargs)
h = torch.randn(options.batch_size, options.state_size, **kwargs)
C = torch.randn(options.batch_size, options.state_size, **kwargs)
rnn = LLTM(options.features, options.state_size).to(device, dtype)

# Force CUDA initialization
new_h, new_C = rnn(X, (h, C))
(new_h.sum() + new_C.sum()).backward()


