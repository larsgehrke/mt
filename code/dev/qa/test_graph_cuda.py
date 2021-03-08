'''

    Quality Assurance: Test the cuda implementation of the graph connections 

'''

from model.th2.graph import Graph
import torch as th

from tools.debug import sprint

def test_graph():

    pk_rows, pk_cols, pk_dyn_size = 16, 16, 1

    g = Graph(pk_rows,pk_cols)

    input_zeros = th.zeros(size=(8,                              
                                 pk_rows * pk_cols,
                                 pk_dyn_size),
                              device="gpu")

    out = g.forward(input_zeros)
    print(out)
    sprint(out, "out")
    
    

    

