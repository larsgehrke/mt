import numpy as np
import torch as th

import model.v1a.pk as pk

from tools.debug import sprint

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
        self.pk_net = pk.PK(batch_size= config.batch_size_train,
                              amount_pks= config.amount_pks, 
                              input_size = config.pk_neighbors * config.pk_lat_size + config.pk_dyn_size, 
                              lstm_size = config.pk_num_lstm_cells,
                              device = config.device)

        # Variables for the PK-TK connections
        self.pk_adj_mat = None
        self.pos0 = None
        self.coming_from = None
        self.going_to = None

        self._build_connections(config.pk_rows, config.pk_cols)

    def forward(self, dyn_in):
        """
        Runs the forward pass of all PKs and TKs, respectively, in parallel 
        for a given input

        """

        # Write the dynamic PK input to the corresponding tensor
        self.tensors.pk_dyn_in = dyn_in

        # Set the appropriate lateral inputs to the lateral outputs from the
        # previous time step
        self.tensors.pk_lat_in[:,self.pos0, self.going_to] = \
        self.tensors.pk_lat_out[:,self.coming_from, self.going_to]

       
        # Flatten last two dims: B, PK, N, Lat -> B, PK, N*Lat and concat with B, PK, Dyn
        # => B, PK, N*Lat + Dyn
        input_ = th.cat((self.tensors.pk_dyn_in, th.flatten(self.tensors.pk_lat_in,start_dim=2)),2)


        # Forward the PK inputs through the pk_net to get the outputs and hidden
        # states of these PKs
        pk_output, pk_lstm_h, pk_lstm_c = self.pk_net.forward(
            input_flat= input_, # Size: [B, PK,  N*Lat + Dyn]
            old_h= self.tensors.pk_lstm_h,  # Size: [B, PK,  LSTM]
            old_c= self.tensors.pk_lstm_c # Size: [B, PK,  LSTM]
        )
        # pk_output: [B, PK, DYN + N*LAT]

        # Dynamic output
        pk_dyn_out = pk_output[:, :,  :self.config.pk_dyn_size]
        

        # Lateral output flattened
        pk_lat_out_flat = pk_output[:, :, self.config.pk_dyn_size:]

        # Batch Size is flexibel (note end of epoch)
        pk_lat_out = pk_lat_out_flat.view(size=(-1,
            self.config.amount_pks, self.config.pk_neighbors, self.config.pk_lat_size))


        # Update the output and hidden state tensors of the PKs
        self.tensors.pk_dyn_out = pk_dyn_out
        self.tensors.pk_lat_out = pk_lat_out
        self.tensors.pk_lstm_h = pk_lstm_h
        self.tensors.pk_lstm_c = pk_lstm_c    

    def reset(self, batch_size):
        self.tensors.set_batch_size_and_reset(batch_size)
        self.pk_net.set_batch_size(batch_size)

    def _build_connections(self, rows, cols):

        # Initialize an adjacency matrix for the PK-TK connections
        self.pk_adj_mat = th.zeros(size=(2,
                                         rows * cols,
                                         rows * cols),
                                   device=self.config.device)

        # Define a dictionary that maps directions to numbers
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

        # PK indexes that are to be considered in the lateral update
        self.pos0 = th.from_numpy(a[0]).to(dtype=th.long)
        # Define matrix from where the lateral inputs come and where they go
        self.coming_from = th.from_numpy(a[1]).to(dtype=th.long)
        self.going_to = (self.pk_adj_mat[1][a] - 1).to(dtype=th.long)

