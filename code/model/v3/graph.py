import math
import os
import torch as th
import numpy as np

from torch.utils.cpp_extension import load

# Load/Compile the cuda extension 
# in the include folder is the Python generated config.h file,
# that contains the parameter/config values
cpp = os.path.join('model', 'v3', 'graph.cpp')
cu = os.path.join('model', 'v3', 'graph.cu')
include = os.path.join('model', 'v3', 'include')

# compile the C++ and CUDA code on the fly when this script is imported
graph_cuda = load(
    'graph', [cpp, cu],
    extra_include_paths=[include], verbose = False)

class GraphFunction(th.autograd.Function):
    '''
    Custom Pytorch autograd Function that wraps the CUDA code for the graph connections.
    '''
    connections = None

    @staticmethod
    def forward(ctx, dyn_in, lat_in):

        rearranged_in = graph_cuda.forward(dyn_in.contiguous(), 
            lat_in.contiguous(), GraphFunction.connections)[0]

        return rearranged_in

    @staticmethod
    def backward(ctx, grad_rearranged_in):

        d_dyn_in, d_lat_in = graph_cuda.backward(grad_rearranged_in.contiguous(),
            GraphFunction.connections)

        return d_dyn_in, d_lat_in


class Graph(th.nn.Module):
    def __init__(self, connections):
        super(Graph, self).__init__()

        GraphFunction.connections = connections


    def forward(self, dyn_in, lat_in):
        '''
            Implementing the lateral connections (graph edges) of DISTANA
            
            :param input_flat: 
                The input for the PKs where dynamical input is concatenated with flattened dynamical input.
                Size is [B, PK, DYN + N*LAT] with batch size B, amount of PKs PK, dynamical input size DYN,
                Neighbors N and lateral input size LAT.

        '''
        return GraphFunction.apply(dyn_in, lat_in)


