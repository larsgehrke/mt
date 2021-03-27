'''

    Quality Assurance: Test the cuda implementation of the graph connections 

'''

from model.v2.graph import Graph
import torch as th
import numpy as np

import time

from tools.debug import sprint

def test_graph():

    print("This is a unit test for the cuda graph connection implementation.")
    print("Testing it with 16x16 PK grid, 8 neighbors and 1 lateral vector size.")

    success = True

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

    out_all = out.cpu().detach().numpy()
    # lateral output
    out = out_all[:,:,1:]

    if np.sum(out_all[:,:,0]) == 0:
        print("SUCCESS: Test successful for dynamical input")
    else:
        print("FAILURE: Test not successful for dynamical input")
        success = False
    
    for b in range(8):

        print(" === BATCH "+ str(b) + " ===")
        
        current = np.reshape(np.array([np.sum(x) for x in out[b,:]]), (256,1))
        
        if np.sum(current-expect) == 0:
            print("SUCCESS:Test successful for sum of lateral input in batch " + str(b))
        else:
            success = False
            print("FAILURE: Test not successful for batch " + str(b))
            print(sum(out[b]))
            print(sum(expect))

            # Visualize PK grid
            s = ""
            print(f"\n\n ==== BATCH {str(b)} ==== \n")
            for y in range(pk_rows):
              for x in range(pk_cols):
                s += str(np.sum(out[b][y*pk_cols + x])) + " "
              s += "\n"

            print(s)
            sprint(out, "out")

        if np.all(out[b, 0] == np.array([0,0,0,0,1,0,1,1])):
            print("SUCCESS:Test successful for node 0 in batch " + str(b))
        else:
            success = False
            print("FAILURE: Test not successful for node 0 in batch " + str(b))

        if np.all(out[b, 5] == np.array([0,0,0,1,1,1,1,1])):
            print("SUCCESS:Test successful for node 5 in batch " + str(b))
        else:
            success = False
            print("FAILURE: Test not successful for node 5 in batch " + str(b))


        if np.all(out[b, 15] == np.array([0,0,0,1,0,1,1,0])):
            print("SUCCESS:Test successful for node 15 in batch " + str(b))
        else:
            success = False
            print("FAILURE: Test not successful for node 15 in batch " + str(b))


        if np.all(out[b, 16] == np.array([0,1,1,0,1,0,1,1])):
            print("SUCCESS:Test successful for node 16 in batch " + str(b))
        else:
            success = False
            print("FAILURE: Test not successful for node 16 in batch " + str(b))

        if np.all(out[b, 17] == np.array([1,1,1,1,1,1,1,1])):
            print("SUCCESS:Test successful for node 17 in batch " + str(b))
        else:
            success = False
            print("FAILURE: Test not successful for node 17 in batch " + str(b))

        if np.all(out[b, 255] == np.array([1,1,0,1,0,0,0,0])):
            print("SUCCESS:Test successful for node 255 in batch " + str(b))
        else:
            success = False
            print("FAILURE: Test not successful for node 255 in batch " + str(b))

        print()


    print()
    print("IN TOTAL:")
    if success:    
        print("===> SUCCESS")
    else:
        print("===> FAILURE")
        
    
    

    

