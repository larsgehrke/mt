'''

    Quality Assurance: Test the cuda implementation of the graph connections from model v3

    This unit test checks if the graph connections implemented in the CUDA code of model v3 
    are correctly implemented. Two groups of characteristics are checked: 
    The amount of connections per node and the position of the connections in the lateral array.

'''
from model.v3.graph import GraphFunction
import torch as th
import numpy as np

import time

from tools.debug import sprint

device_str = 'cuda'

def test_graph():

    print("This is a unit test for the cuda graph connection implementation.")
    print("Testing it with 16x16 PK grid, 8 neighbors and 1 lateral vector size.")

    success = True

    pk_rows, pk_cols, pk_neighbors, pk_neighbor_size = 16, 16, 8, 1
    total = pk_rows * pk_cols
    
    GraphFunction.connections = _build_connections()

    dyn_in = th.zeros(size=(1, total, 1),
                              device=device_str)
    dyn_in[0,0,0] = 42
    dyn_in[0,3,0] = 42
    dyn_in[0,7,0] = 42
    dyn_in[0,19,0] = 42
    dyn_in[0,44,0] = 42


    lat_in = th.ones(size=(1, total, 1),
                              device=device_str)
    lat_in[0,0,0] = 2
    lat_in[0,5,0] = 55
    lat_in[0,10,0] = 12    

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

    out = GraphFunction.forward(None,dyn_in, lat_in)
    d_dyn_in, d_lat_in = GraphFunction.backward(None,out)

   
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


def _build_connections(rows=16, cols=16):

    # Initialize an adjacency matrix for the PK-TK connections
    pk_adj_mat = th.zeros(size=(2,
                                rows * cols,
                                rows * cols),
                            device=device_str)

    
    # Old encoding:
    # direction_dict = {"top": 1, "left top": 2, "left": 3, "left bottom": 4,
    #                   "bottom": 5, "right bottom": 6, "right": 7,
    #                   "right top": 8}
    
    # Define a dictionary that maps directions to numbers
    direction_dict = {"left top": 1, "top": 2, "right top": 3, "left": 4, "right": 5,
                        "left bottom": 6, "bottom": 7, "right bottom": 8}
    

    # Running index to pass a distinct id to each PK
    pk_id_running = 0

    # Iterate over all PK rows and columns to create PK instances
    for pk_row in range(rows):
        for pk_col in range(cols):

            # Find the neighboring PKs to which this PK is connected
            neighbors = {"top": [pk_row - 1, pk_col],
                         "left top": [pk_row - 1, pk_col - 1],
                         "left": [pk_row, pk_col - 1],
                         "left bottom": [pk_row + 1, pk_col - 1],
                         "bottom": [pk_row + 1, pk_col],
                         "right bottom": [pk_row + 1, pk_col + 1],
                         "right": [pk_row, pk_col + 1],
                         "right top": [pk_row - 1, pk_col + 1]}

            # Set the values of the PK adjacency matrix on true that
            # represent a connection between the connected PKs
            for neighbor_direction in neighbors:

                # Get the row and column index of the current neighbor
                neighbor_row, neighbor_col = neighbors[neighbor_direction]

                # If the neighbor lies within the defined field, define
                # it as neighbor in the adjacency matrix
                if (0 <= neighbor_row < rows) and \
                   (0 <= neighbor_col < cols):

                    # Determine the index of the neighbor
                    neighbor_idx = neighbor_row * cols + neighbor_col

                    # Set the corresponding entry in the adjacency matrix to
                    # one
                    pk_adj_mat[0, pk_id_running, neighbor_idx] = 1
                    pk_adj_mat[1, pk_id_running, neighbor_idx] = \
                        direction_dict[neighbor_direction]

            pk_id_running += 1

    #
    # Set up vectors that describe which lateral output goes to which
    # lateral input
    a = np.where(pk_adj_mat[0].cpu().detach().numpy() == 1)

    # PKs that are to be considered in the lateral update
    pos0 = th.from_numpy(a[0]).to(device=device_str, dtype=th.long)
    # PK lateral outputs that will be sent to the lateral inputs
    coming_from = th.from_numpy(a[1]).to(device=device_str,
                                              dtype=th.long)
    # PK lateral input neurons that will get inputs from the previous time
    # step's lateral output
    going_to = (pk_adj_mat[1][a] - 1).to(device=device_str,
                                                   dtype=th.long)


    return _prepare_connections(pos0, coming_from, going_to)

def _prepare_connections(pos0, coming_from, going_to):

    bincount = th.bincount(pos0)
    length = th.max(bincount).item()

    connections = th.zeros((16*16, length, 2)).to(device = device_str)-1

    idx_counts = th.zeros((16*16)).to(device = device_str, dtype = th.long)

    for k,v in enumerate(pos0):
        
        pk_idx = v.item()
       
        position_idx = idx_counts[pk_idx]
        connections[pk_idx][position_idx][0] = coming_from[k]
        connections[pk_idx][position_idx][1] = going_to[k]
        idx_counts[pk_idx] += 1

    return connections
    

    
    
    

    

