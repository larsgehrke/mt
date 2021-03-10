'''

    Quality Assurance: Test the cuda implementation of the graph connections 

'''

from model.th2.graph import Graph
import torch as th
import numpy as np

import time

from tools.debug import sprint

def test_graph():

    pk_rows, pk_cols, pk_neighbors, pk_neighbor_size = 16, 16, 4, 1

    g = Graph(pk_rows,pk_cols)

    dyn_in = th.zeros(size=(8, 256, 1),
                              device="cuda")

    lat_in = th.ones(size=(8, 256, 1),
                              device="cuda")

    
    start = time.time()
    out = g.forward(dyn_in, lat_in)
    stop = time.time()

    s = ""

    for y in range(pk_rows):
      for x in range(pk_cols):
        s += str(np.sum(out[0][y*pk_cols + x].cpu().detach().numpy())) + " "
      s += "\n"

    print(s)

    print(out[0][0].cpu().detach().numpy())
    print(out[0][5].cpu().detach().numpy())
    print(out[0][15].cpu().detach().numpy())
    print(out[0][16].cpu().detach().numpy())
    print(out[0][17].cpu().detach().numpy())
    print(out[0][255].cpu().detach().numpy())
    
    sprint(out, "out")
    print(stop-start)
    
    

    

