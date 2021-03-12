'''

    Quality Assurance: Test the cuda implementation of the graph connections 

'''

from model.th2.graph import Graph
import torch as th
import numpy as np

import time

from tools.debug import sprint

def test_graph():

    print("This is a unit test for the cuda graph connection implementation.")
    print("Testing it with 16x16 PK grid, 8 neighbors and 1 lateral vector size.")

    pk_rows, pk_cols, pk_neighbors, pk_neighbor_size = 16, 16, 8, 1
    total = pk_rows * pk_cols

    g = Graph(pk_rows,pk_cols)

    dyn_in = th.zeros(size=(8, total, 1),
                              device="cuda")

    lat_in = th.ones(size=(8, total, 1),
                              device="cuda")

    # Most of the nodes have 8 incoming values
    expect = np.zeros((total, 1)) + 8


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


    start = time.time()
    out = g.forward(dyn_in, lat_in)
    stop = time.time()
    print("forward pass took " + str(stop-start) + " seconds")

    out = out.cpu().detach().numpy()
    
    for b in range(8):
        
        
        if np.sum(sum(out[b])-sum(expect)) == 0:
            print("Test successful for batch " + str(b))
        else:
            print("Test not successful for batch " + str(b))

            # Visualize PK grid
            s = ""
            print(f"\n\n ==== BATCH {str(b)} ==== \n")
            for y in range(pk_rows):
              for x in range(pk_cols):
                s += str(np.sum(out[b][y*pk_cols + x])) + " "
              s += "\n"

            print(s)
            sprint(out, "out")

        # print("out[0][0]")
        # print(out[0][0].cpu().detach().numpy())
        # print("out[0][5]")
        # print(out[0][5].cpu().detach().numpy())
        # print("out[0][15]")
        # print(out[0][15].cpu().detach().numpy())
        # print("out[0][16]")
        # print(out[0][16].cpu().detach().numpy())
        # print("out[0][17]")
        # print(out[0][17].cpu().detach().numpy())
        # print("out[0][255]")
        # print(out[0][255].cpu().detach().numpy())
        
    
    print(stop-start)
    
    

    

