import math
from torch import nn
from torch.autograd import Function
import torch

import lltm_cuda

import sys
import numpy as np

torch.manual_seed(42)

def sprint(obj, obj_name="Object", complete=False, exit=False):
    print("Printing out", obj_name)
    print(type(obj))

    if (isinstance(obj, torch.Tensor)):
        obj = obj.cpu().detach().numpy()

    if (isinstance(obj, np.ndarray)):
        print(obj.shape)

    if (complete):
        print(obj)

    if(exit):
        sys.exit()


class LLTMFunction(Function):
    @staticmethod
    def forward(ctx, input, weights, bias, old_h, old_cell):
        outputs = lltm_cuda.forward(input, weights, bias, old_h, old_cell)
        # outputs =  {new_h, new_cell, input_gate, output_gate, candidate_cell, X, gates};
        new_h, new_cell = outputs[:2]
        outputs[-1] = outputs[-1].unflatten(1, (3, 128))
        variables = outputs[1:] + [weights]
        ctx.save_for_backward(*variables)

        return new_h, new_cell

    @staticmethod
    def backward(ctx, grad_h, grad_cell):
        var_list = []

        '''
        grad_h:
        torch.Size([16, 128])
        grad_cell:
        torch.Size([16, 128])
    
        ctx:
        0:  torch.Size([16, 128]) 
        1: torch.Size([16, 128])
        2: torch.Size([16, 128])
        3: torch.Size([16, 128]) 
        4: torch.Size([16, 160]) "X"
        5: torch.Size([16, 384]) this neeeded an extra dim "gates"
        6: torch.Size([384, 160])

        '''
        for idx,var in enumerate(ctx.saved_variables):

            if (False): #idx in [5]):
                var_list.append(var.unsqueeze_(0))
            else:
                var_list.append(var)
        

        outputs = lltm_cuda.backward(
            grad_h.contiguous(), grad_cell.contiguous(), *var_list)
        d_old_h, d_input, d_weights, d_bias, d_old_cell, d_gates = outputs
        return d_input, d_weights, d_bias, d_old_h, d_old_cell


class LLTM(nn.Module):
    def __init__(self, input_features, state_size):
        super(LLTM, self).__init__()
        self.input_features = input_features
        self.state_size = state_size
        self.weights = nn.Parameter(
            torch.Tensor(3 * state_size, input_features + state_size))
        self.bias = nn.Parameter(torch.Tensor(1, 3 * state_size))
        self.reset_parameters()

    def reset_parameters(self):
        stdv = 1.0 / math.sqrt(self.state_size)
        for weight in self.parameters():
            weight.data.uniform_(-stdv, +stdv)

    def forward(self, input, state):
        return LLTMFunction.apply(input, self.weights, self.bias, *state)

    def backward(self, grad_h, grad_cell):
        return LLTMFunction.apply(grad_h, grad_cell)
