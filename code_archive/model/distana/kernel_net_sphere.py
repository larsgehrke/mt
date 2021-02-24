import numpy as np
import torch as th
import torch.nn as nn
import prediction_kernel
import configuration as cfg


class KernelNetwork(nn.Module):
    """
    This class contains the kernelized network topology for the spatio-temporal
    propagation of information
    """

    def __init__(self, params, tensors):

        super(KernelNetwork, self).__init__()

        self.params = params
        self.tensors = tensors

        #
        # Prediction Kernels

        # Initialize the shared Prediction Kernel (PK) network that will do the
        # PK calculations
        self.pk_net = prediction_kernel.PredictionKernelNet(params=params)

        # Initialize an adjacency matrix for the PK-TK connections
        pk_adj_mat = th.zeros(size=(2,
                                    cfg.PK_ROWS * cfg.PK_COLS,
                                    cfg.PK_ROWS * cfg.PK_COLS),
                              device=params.device)

        # Define a dictionary that maps directions to numbers
        direction_dict = {"top left": 1, "top": 2, "top right": 3,
                          "left": 4, "right": 5,
                          "bottom left": 6, "bottom": 7, "bottom right": 8}

        # Use this instead of the above eight-neighbors dictionary for a four-
        # neighborhood if desired
        # direction_dict = {"top": 1, "left": 2, "right": 3, "bottom": 4}

        # Running index to pass a distinct id to each PK
        pk_id_running = 0

        # Iterate over all PK rows and columns to create PK instances
        for pk_row in range(cfg.PK_ROWS):
            for pk_col in range(cfg.PK_COLS):

                # Find the neighboring PKs to which this PK is connected
                neighbors = {"top left": [pk_row - 1, pk_col - 1],
                             "top": [pk_row - 1, pk_col],
                             "top right": [pk_row - 1, pk_col + 1],
                             "left": [pk_row, pk_col - 1],
                             "right": [pk_row, pk_col + 1],
                             "bottom left": [pk_row + 1, pk_col - 1],
                             "bottom": [pk_row + 1, pk_col],
                             "bottom right": [pk_row + 1, pk_col + 1]}

                # Use this instead of the above eight-neighbors dictionary for a
                # four-neighborhood if desired. Don't forget to also change the
                # direction_dict accordingly!
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
                    if (0 <= neighbor_row < cfg.PK_ROWS) and \
                       (0 <= neighbor_col < cfg.PK_COLS):

                        # Determine the index of the neighbor
                        neighbor_idx = neighbor_row * cfg.PK_COLS + neighbor_col

                    elif neighbor_row == -1:
                        # The neighbor is at the top outside of the simulation
                        # field.

                        # Determine the index of the neighbor
                        # Get the index from the other half
                        neighbor_idx = (neighbor_col + (cfg.PK_COLS // 2))\
                                       % cfg.PK_COLS

                    elif neighbor_row == cfg.PK_ROWS:
                        # the neighbor is at the bottom outside of the
                        # simulation field.

                        # Determine the index of the neighbor
                        neighbor_idx = (cfg.PK_ROWS - 1) * cfg.PK_COLS + \
                                       (neighbor_col + (cfg.PK_COLS // 2))\
                                       % cfg.PK_COLS

                    else:
                        # When arriving here, the currently considered neighbor
                        # of the currently inspected PK is out of the simulation
                        # field. Thus, the neighbor is taken from the other end
                        # of the field to create a sphere over which information
                        # can be propagated endlessly.

                        # Compute the modulo values of the current neighbor row
                        # and col. This yields the inverse values for negatives,
                        # e.g. it results in 5 - which is the last column - for
                        # col=-1 (given the field is made up of 6 columns)
                        neighbor_row %= cfg.PK_ROWS
                        neighbor_col %= cfg.PK_COLS

                        # Determine the index of the neighbor
                        neighbor_idx = neighbor_row * cfg.PK_COLS + neighbor_col

                    # Set the corresponding entry in the adjacency matrix to
                    # one.
                    pk_adj_mat[0, pk_id_running, neighbor_idx] = 1

                    # Set the second dimension (which indicates to which
                    # position in the lateral vector this value will be
                    # written to) of the adjacency matrix
                    pk_adj_mat[1, pk_id_running, neighbor_idx] = \
                        direction_dict[neighbor_direction]

                # The linking of the current PK to all its neighbors has been
                # completed, thus increment the running PK id
                pk_id_running += 1

        #
        # Set up vectors that describe which lateral output goes to which
        # lateral input
        # produces two arrays (a[0]:x and a[1]:y positions where ==1 is true); 
        a = np.where(pk_adj_mat[0].cpu().detach().numpy() == 1)
        
        # PK indexes that are to be considered in the lateral update
        # a[0] = self.pos0, are x/rows in adjacent matrix with 1 entry
        # this is import to get the assigment per PK right
        # in teamwork with self.coming_from and self.going_to
        # it contains boring data: 0,0,0...(8x),1,1,1...(8x),...,255,255...(8x)
        self.pos0 = th.from_numpy(a[0]).to(dtype=th.long)
        # Define matrix from where the lateral inputs come and where they go
        # see pk_adj_mat[1, pk_id_running, neighbor_idx] 
        # -> rows is current pk and column is 'coming from' pk
        self.coming_from = th.from_numpy(a[1]).to(dtype=th.long)
        # Why -1? Because the direction initialization starts with 1,2,3...8
        # direction type: top, left, bottom... encoded as 0-7
        self.going_to = (pk_adj_mat[1][a] - 1).to(dtype=th.long)

    def forward(self, dyn_in, pk_stat_in=None, tk_stat_in=None):
        """
        Runs the forward pass of all PKs and TKs, respectively, in parallel for
        a given input
        :param dyn_in: The dynamic input for the PKs
        :param pk_stat_in: (optional) The static input for the PKs
        :param tk_stat_in: (optional) The static input for the TKs
        """

        # Write the dynamic PK input to the corresponding tensor
        if isinstance(dyn_in, th.Tensor):
            self.tensors.pk_dyn_in = dyn_in
        else:
            self.tensors.pk_dyn_in = th.from_numpy(
                dyn_in
            ).to(device=self.params.device)

        # Set the appropriate lateral inputs to the lateral outputs from the
        # previous time step
        self.tensors.pk_lat_in[self.pos0, self.going_to] = \
            self.tensors.pk_lat_out[self.coming_from, self.going_to]

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

    def reset(self, pk_num):
        self.tensors.reset(pk_num=pk_num)

    def detach(self):
        self.tensors.detach()
