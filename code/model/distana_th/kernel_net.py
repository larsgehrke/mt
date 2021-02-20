import numpy as np
import torch as th
import torch.nn as nn
import configuration as cfg

import helper_functions as helpers

import pk


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
        self.pk_net = pk.PK(batch_size=cfg.BATCH_SIZE,
                              amount_pks=cfg.PK_ROWS*cfg.PK_COLS, 
                              input_size = cfg.PK_NEIGHBORS + 1, 
                              lstm_size = cfg.PK_NUM_LSTM_CELLS,
                              device = params.device)

        # Initialize an adjacency matrix for the PK-TK connections
        self.pk_adj_mat = th.zeros(size=(2,
                                         cfg.PK_ROWS * cfg.PK_COLS,
                                         cfg.PK_ROWS * cfg.PK_COLS),
                                   device=params.device)

        # Define a dictionary that maps directions to numbers
        # direction_dict = {"top": 1, "left": 2, "right": 3, "bottom": 4}
        direction_dict = {"top left": 1, "top": 2, "top right": 3,
                          "left": 4, "right": 5,
                          "bottom left": 6, "bottom": 7, "bottom right": 8}

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

        # PK indexes that are to be considered in the lateral update
        self.pos0 = th.from_numpy(a[0]).to(dtype=th.long)
        # Define matrix from where the lateral inputs come and where they go
        self.coming_from = th.from_numpy(a[1]).to(dtype=th.long)
        self.going_to = (self.pk_adj_mat[1][a] - 1).to(dtype=th.long)

    def forward(self, dyn_in, pk_stat_in=None, tk_stat_in=None):
        """
        Runs the forward pass of all PKs and TKs, respectively, in parallel 
        for a given input

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
        self.tensors.pk_lat_in[:,self.pos0, self.going_to] = \
        self.tensors.pk_lat_out[:,self.coming_from, self.going_to]

        # helpers.sprint(self.tensors.pk_dyn_in, "self.tensors.pk_dyn_in")
        # helpers.sprint(self.tensors.pk_lat_in, "self.tensors.pk_lat_in")
        # helpers.sprint(self.tensors.pk_lstm_c, "self.tensors.pk_lstm_c")
        # helpers.sprint(self.tensors.pk_lstm_h, "self.tensors.pk_lstm_h", exit=True)

        # Insert Dim 10, 256, 1 -> 10, 256, 1,1 and concat with 10, 256, 8, 1
        # => 10, 256, 9, 1
        input_ = th.cat((th.unsqueeze(self.tensors.pk_dyn_in,2), self.tensors.pk_lat_in),2)

        # Forward the PK inputs through the pk_net to get the outputs and hidden
        # states of these PKs
        pk_output, pk_lstm_h, pk_lstm_c = self.pk_net.forward(
            input_= input_, # (10, 256, 9, 1)
            old_h= self.tensors.pk_lstm_h,  # (10, 256, 16)
            old_c= self.tensors.pk_lstm_c # (10, 256, 16)            
        )

        # Dynamic output
        pk_dyn_out = pk_output[:, :,  :self.params.pk_dyn_out_size]

        # Lateral output
        pk_lat_out = pk_output[:, : , self.params.pk_dyn_out_size:]

        

        #pk_lstm_h: (10, 256, 16)
        #pk_lstm_c: (10, 256, 16)

        #helpers.sprint(pk_dyn_out)
        #helpers.sprint(pk_lat_out)
        #helpers.sprint(pk_lstm_h, "kernel_net.pk_lstm_h")
        #print(pk_lstm_h[0][0][0])
        #helpers.sprint(pk_lstm_c, "kernel_net.pk_lstm_c", exit=True)
        

        # Update the output and hidden state tensors of the PKs
        self.tensors.pk_dyn_out = pk_dyn_out
        self.tensors.pk_lat_out = pk_lat_out
        self.tensors.pk_lstm_h = pk_lstm_h
        self.tensors.pk_lstm_c = pk_lstm_c    





    def forward_old(self, dyn_in, pk_stat_in=None, tk_stat_in=None):
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

    def reset(self):
        self.tensors.reset()

    def detach(self):
        self.tensors.detach()
