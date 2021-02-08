import math
import torch

from torch.utils.cpp_extension import load

import numpy as np
import sys

distana_cuda = load(
    'distana_cuda', ['cuda/distana_cuda.cpp', 'cuda/distana_cuda_kernel.cu'], verbose=True)

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


class DISTANAFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input, pre_weights, lstm_weights, post_weights, old_h, old_cell):
        sprint(input,"input")
        sprint(pre_weights,"pre_weights")
        sprint(lstm_weights,"lstm_weights")
        sprint(post_weights,"post_weights")
        sprint(old_h,"old_h", exit=True)

        outputs = distana_cuda.forward(input, weights, bias, old_h, old_cell)
        new_h, new_cell = outputs[:2]
        input_gate = outputs[2]

        variables = outputs[1:] + [weights]
        ctx.save_for_backward(*variables)

        return new_h, new_cell

    @staticmethod
    def backward(ctx, grad_h, grad_cell):
        # When you call contiguous(), it actually makes a copy of tensor 
        # so the order of elements would be same 
        # as if tensor of same shape created from scratch.
        outputs = distana_cuda.backward(
            grad_h.contiguous(), grad_cell.contiguous(), *ctx.saved_variables)

        d_old_h, d_input, d_weights, d_bias, d_old_cell, d_gates = outputs


        return d_input, d_weights, d_bias, d_old_h, d_old_cell


class DISTANA(torch.nn.Module):

    def __init__(self, params):

        super(DISTANA, self).__init__()
        self.params = params

        '''

        Information in self.params, e.g. 
        nn.Linear: in_features = params.pk_dyn_in_size +
                        (params.pk_lat_in_size * params.pk_neighbors

        '''

        self.pre_weights = torch.nn.Parameter(
            torch.Tensor(params.pk_dyn_in_size +
                        (params.pk_lat_in_size * params.pk_neighbors), 
                            params.pk_pre_layer_size))

        self.lstm_weights = torch.nn.Parameter(
            torch.Tensor(params.pk_pre_layer_size, 
                         params.pk_num_lstm_cells))

        self.post_weights = torch.nn.Parameter(
            torch.Tensor(params.pk_num_lstm_cells, 
                         params.pk_dyn_out_size +
                         (params.pk_lat_out_size * params.pk_neighbors)))

        self.reset_parameters()

    def reset_parameters(self):
        stdv = 1.0 / math.sqrt(self.params.pk_num_lstm_cells)
        for weight in self.parameters():
            weight.data.uniform_(-stdv, +stdv)

    def forward(self, input, state):
        return DISTANAFunction.apply(input, self.pre_weights, self.lstm_weights, self.post_weights, *state)

    def backward(self, grad_h, grad_cell):
        return LLTMFunction.apply(grad_h, grad_cell)
