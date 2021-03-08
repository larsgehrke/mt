import math
import torch as th
import numpy as np

from torch.utils.cpp_extension import load

from tools.debug import sprint

graph_cuda = load(
    'graph', ['model/th2/graph.cpp', 'model/th2/graph.cu'],
    verbose=True)
# extra_include_paths=["cuda/include"]

class GraphFunction(th.autograd.Function):

    @staticmethod
    def forward(ctx, input_):

        rearranged_in = graph_cuda.forward(input_)

        return rearranged_in

    @staticmethod
    def backward(ctx, grad_rearranged_in):

        d_input_ = graph_cuda.backward(grad_rearranged_in.contiguous())

        return d_input_


class Graph(th.nn.Module):
    def __init__(self, pk_rows, pk_cols):
        super(Graph, self).__init__()

        # Variables for the PK-TK connections
        self.pos0 = None
        self.going_to = None
        self.coming_from = None

        self._build_connections(pk_rows, pk_cols)


    def forward(self, input_):
        '''
            Implementing the lateral connections (graph edges) of DISTANA
            
            :param input_flat: 
                The input for the PKs where dynamical input is concatenated with flattened dynamical input.
                Size is [B, PK, DYN + N*LAT] with batch size B, amount of PKs PK, dynamical input size DYN,
                Neighbors N and lateral input size LAT.

        '''

        # Set the appropriate lateral inputs to the lateral outputs from the
        # previous time step
        # output[:,self.pos0, self.going_to] = \
        # input_[:,self.coming_from, self.going_to]
        sprint(input_, "input_", exit=True)

        return GraphFunction.apply(input_)

    def _build_connections(self, rows, cols):


        # Initialize an adjacency matrix for the PK-TK connections
        pk_adj_mat = np.zeros((2, rows*cols,rows*cols))

        # Define a dictionary that maps directions to numbers
        # direction_dict = {"top": 1, "left": 2, "right": 3, "bottom": 4}
        direction_dict = {"top left": 1, "top": 2, "top right": 3,
                          "left": 4, "right": 5,
                          "bottom left": 6, "bottom": 7, "bottom right": 8}

        # Running index to pass a distinct id to each PK
        pk_id_running = 0

        # Iterate over all PK rows and columns to create PK instances
        for pk_row in range(rows):
            for pk_col in range(cols):

                # Find the neighboring PKs to which this PK is connected
                neighbors = {"top left": [pk_row - 1, pk_col - 1],
                             "top": [pk_row - 1, pk_col],
                             "top right": [pk_row - 1, pk_col + 1],
                             "left": [pk_row, pk_col - 1],
                             "right": [pk_row, pk_col + 1],
                             "bottom left": [pk_row + 1, pk_col - 1],
                             "bottom": [pk_row + 1, pk_col],
                             "bottom right": [pk_row + 1, pk_col + 1]}
                # neighbors = {"top": [pk_row - 1, pk_col],
                #              "left": [pk_row, pk_col - 1],
                #              "right": [pk_row, pk_col + 1],
                #              "bottom": [pk_row + 1, pk_col]}

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
        a = np.where(pk_adj_mat[0] == 1)

        # PK indexes that are to be considered in the lateral update
        self.pos0 = th.from_numpy(a[0]).to(dtype=th.long)
        # Define matrix from where the lateral inputs come and where they go
        self.coming_from = th.from_numpy(a[1]).to(dtype=th.long)
        self.going_to = th.from_numpy(pk_adj_mat[1][a] - 1).to(dtype=th.long)