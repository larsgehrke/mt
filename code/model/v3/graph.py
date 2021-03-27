import math
import os
import torch as th
import numpy as np

from torch.utils.cpp_extension import load

from tools.debug import sprint

# Load/Compile the cuda extension 
# in the include folder is the config.h file,
# that saves the configuration values
cpp = os.path.join('model', 'v3', 'graph.cpp')
cu = os.path.join('model', 'v3', 'graph.cu')
include = os.path.join('model', 'v3', 'include')

graph_cuda = load(
    'graph', [cpp, cu],
    extra_include_paths=[include], verbose = False)

class GraphFunction(th.autograd.Function):


    @staticmethod
    def forward(ctx, dyn_in, lat_in):

        rearranged_in = graph_cuda.forward(dyn_in.contiguous(), 
            lat_in.contiguous(), Connections.getInstance().get())[0]

        return rearranged_in

    @staticmethod
    def backward(ctx, grad_rearranged_in):

        d_dyn_in, d_lat_in = graph_cuda.backward(grad_rearranged_in.contiguous(),
            Connections.getInstance().get())

        return d_dyn_in, d_lat_in


class Graph(th.nn.Module):
    def __init__(self, connections):
        super(Graph, self).__init__()

        Connections.createInstance()
        Connections.getInstance().set(connections)


    def forward(self, dyn_in, lat_in):
        '''
            Implementing the lateral connections (graph edges) of DISTANA
            
            :param input_flat: 
                The input for the PKs where dynamical input is concatenated with flattened dynamical input.
                Size is [B, PK, DYN + N*LAT] with batch size B, amount of PKs PK, dynamical input size DYN,
                Neighbors N and lateral input size LAT.

        '''
        return GraphFunction.apply(dyn_in, lat_in)

class Connections():
    instance = None

    def __init__(self):
        self.connections = None

    def get(self):
        return self.connections

    def set(self, connections):
        self.connections = connections.contiguous()

    @staticmethod
    def createInstance():
        if Connections.instance is None:
            Connections.instance = Connections()

    @staticmethod
    def getInstance():
        return Connections.instance
        


