import numpy as np
import torch as th

from tools.debug import sprint

import model.old2.prediction_kernel as pk


class KernelNetwork(th.nn.Module):
    """
    This class contains the kernelized network topology for the spatio-temporal
    propagation of information
    """

    def __init__(self, config, tensors):

        super(KernelNetwork, self).__init__()

        self.config = config
        self.tensors = tensors

        #
        # Prediction Kernels

        # Initialize the shared Prediction Kernel (PK) network that will do the
        # PK calculations 
        self.pk_net = pk.PredictionKernelNet(config)

        # Variables for the PK-TK connections
        self.pk_adj_mat = None
        self.pos0 = None
        self.coming_from = None
        self.going_to = None

        self._build_connections(config.pk_rows, config.pk_cols)

    def forward(self, dyn_in):
        """
        Runs the forward pass of all PKs and TKs, respectively, in parallel for
        a given input
        :param dyn_in: The dynamic input for the PKs
        :param pk_stat_in: (optional) The static input for the PKs
        :param tk_stat_in: (optional) The static input for the TKs
        """
        
        # Write the dynamic PK input to the corresponding tensor
        self.tensors.pk_dyn_in = dyn_in

        
        # Set the appropriate lateral inputs to the lateral outputs from the
        # previous time step
        self.tensors.pk_lat_in[self.pos0, self.going_to] =\
            self.tensors.pk_lat_out[self.coming_from]

        # sprint(self.tensors.pk_lat_in,"self.tensors.pk_lat_in")
        # sprint(self.going_to,"self.going_to")
        # sprint(self.pos0, "self.pos0")
        # sprint(self.coming_from, "self.coming_from")
        # sprint(self.tensors.pk_lat_in[self.pos0, self.going_to], "self.tensors.pk_lat_in[self.pos0, self.going_to]", exit=True)

        
        # Forward the PK inputs through the pk_net to get the outputs and hidden
        # states of these PKs
        pk_dyn_out, pk_lat_out, pk_lstm_c, pk_lstm_h = self.pk_net.forward(
            dyn_in=self.tensors.pk_dyn_in,
            lat_in=self.tensors.pk_lat_in,
            lstm_c=self.tensors.pk_lstm_c,
            lstm_h=self.tensors.pk_lstm_h
        )

        # Update the output and hidden state tensors of the PKs
        self.tensors.pk_dyn_out = pk_dyn_out
        self.tensors.pk_lat_out = pk_lat_out
        self.tensors.pk_lstm_c = pk_lstm_c
        self.tensors.pk_lstm_h = pk_lstm_h

    def reset(self):
        self.tensors.reset()

    def _build_connections(self, rows, cols):

        # Initialize an adjacency matrix for the PK-TK connections
        self.pk_adj_mat = th.zeros(size=(2,
                                         rows * cols,
                                         rows * cols),
                                   device=self.config.device)

        # Define a dictionary that maps directions to numbers
        direction_dict = {"top": 1, "left top": 2, "left": 3, "left bottom": 4,
                          "bottom": 5, "right bottom": 6, "right": 7,
                          "right top": 8}

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
                        self.pk_adj_mat[0, pk_id_running, neighbor_idx] = 1
                        self.pk_adj_mat[1, pk_id_running, neighbor_idx] = \
                            direction_dict[neighbor_direction]

                pk_id_running += 1

        #
        # Set up vectors that describe which lateral output goes to which
        # lateral input
        a = np.where(self.pk_adj_mat[0].cpu().detach().numpy() == 1)

        # PKs that are to be considered in the lateral update
        self.pos0 = th.from_numpy(a[0]).to(device=self.config.device, dtype=th.long)
        # PK lateral outputs that will be sent to the lateral inputs
        self.coming_from = th.from_numpy(a[1]).to(device=self.config.device,
                                                  dtype=th.long)
        # PK lateral input neurons that will get inputs from the previous time
        # step's lateral output
        self.going_to = (self.pk_adj_mat[1][a] - 1).to(device=self.config.device,
                                                       dtype=th.long)

