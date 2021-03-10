'''

    Quality Assurance: Test the cuda implementation of the graph connections 

'''

from model.th2.graph import Graph
import torch as th

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
      print()

    sprint(out, "out")
    print(stop-start)
    
    

    

