'''

    Quality Assurance: Test the cuda implementation of the graph connections from model v3

    This unit test checks if the graph connections implemented in the CUDA code of model v3 
    are correctly implemented. Two groups of characteristics are checked: 
    The amount of connections per node and the position of the connections in the lateral array.

'''
from model.v3.graph import Graph
import torch as th
import numpy as np

import time

from tools.debug import sprint

device_str = 'cpu'

def test_graph():

    print("This is a unit test for the cuda graph connection implementation.")
    print("Testing it with 16x16 PK grid, 8 neighbors and 1 lateral vector size.")

    success = True

    pk_rows, pk_cols, pk_neighbors, pk_neighbor_size = 16, 16, 8, 1
    total = pk_rows * pk_cols
    
    g = Graph(_build_connections())

    dyn_in = th.zeros(size=(1, total, 1),
                              device=device_str)

    lat_in = th.ones(size=(1, total, 1),
                              device=device_str)

    # Most of the nodes have 8 incoming values
    expect = th.zeros((total, 1)) + 8


    # Most of the edge nodes have only 5 incoming values
    for i in range(total):
        # first row, last row, left and right column
        if i < pk_cols or i > total - pk_cols or i%pk_cols == 0 or (i+1)%pk_cols == 0:
            expect[i] = 5

    # The 4 corners have only 3 incoming values
    expect[0] = 3
    expect[pk_cols-1] = 3
    expect[total - pk_cols] = 3
    expect[total-1] = 3

    expect = th.unsqueeze(expect, 0)
    print(expect.shape)

    out = g.forward(dyn_in, lat_in)
    d_dyn_in, d_lat_in = g.backward(out)

   
    print(f"d_dyn_in.shape: {d_dyn_in.shape}")
    print(f"d_lat_in.shape: {d_lat_in.shape}")
    print("\nd_dyn_in: ")
    print(d_dyn_in)
    print("\nd_lat_in")
    print(d_lat_in)

    # start = time.time()
    # out = g.forward(dyn_in, lat_in)
    # stop = time.time()
    # print("forward pass took " + str(stop-start) + " seconds")

    # out_all = out.cpu().detach().numpy()
    # # lateral output
    # out = out_all[:,:,1:]

    
    
    

    

