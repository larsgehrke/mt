'''

    Quality Assurance: Test the cuda implementation of the graph connections 

'''

from model.th2.graph import Graph
import torch as th

import time

from tools.debug import sprint

def test_graph():

    pk_rows, pk_cols, pk_neighbors, pk_neighbor_size = 16, 16, 8, 1

    g = Graph(pk_rows,pk_cols)

    dyn_in = th.zeros(size=(8, 256, 1),
                              device="cuda")+55

    lat_in = th.ones(size=(8, 256, 1),
                              device="cuda")

    
    start = time.time()
    out = g.forward(dyn_in, lat_in)
    stop = time.time()
    print(out)
    print(out[0][20][0])
    sprint(out, "out")
    print(stop-start)
    
    

    

