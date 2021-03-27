import os
import numpy as np
import torch as th

from model.v3.pk import PK

from tools.debug import sprint
from tools.debug import Clock

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
        self.pk_net = PK(batch_size= config.batch_size_train,
                              amount_pks= config.amount_pks, 
                              input_size = config.pk_dyn_size + config.pk_neighbors * config.pk_lat_size, 
                              lstm_size = config.pk_num_lstm_cells,
                              output_size = config.pk_dyn_size + config.pk_lat_size, 
                              device = config.device)

        # Variables for the PK-TK connections
        self.pos0 = None
        self.going_to = None
        self.coming_from = None

        self._build_connections(config.pk_rows, config.pk_cols)
        
        if self.config.use_gpu:          
            self.graph = None
            self._compile_cuda_extension()

    def _compile_cuda_extension(self):
        cpp_config_file = os.path.join('model', 'v3', 'include','config.h')

        with open(cpp_config_file, 'w') as conf_file:
            conf_file.write("#define PK_ROWS " + str(self.config.pk_rows) + os.linesep)
            conf_file.write("#define PK_COLS " + str(self.config.pk_cols) + os.linesep)
            conf_file.write("#define LAT_SIZE " + str(self.config.pk_lat_size) + os.linesep)
            conf_file.write("#define DYN_SIZE " + str(self.config.pk_dyn_size) + os.linesep)

        # import the custom CUDA kernel
        from model.v3.graph import Graph

        connections = self._prepare_connections()

        self.graph = Graph(connections)

    def _graph_connections(self):
        '''
        Implementing the graph connections of DISTANA.
        '''

        if self.config.use_gpu: 
            # Use the custom CUDA kernel
            input_ =  self.graph.forward(self.tensors.pk_dyn_in, self.tensors.pk_lat_out)
        else:
            # Set the appropriate lateral inputs to the lateral outputs from the
            # previous time step
            self.tensors.pk_lat_in[:,self.pos0, self.going_to] = \
                self.tensors.pk_lat_out[:,self.coming_from]

            # Flatten last two dims: B, PK, N, Lat -> B, PK, N*Lat and concat with B, PK, Dyn
            # => B, PK, Dyn + N*Lat 
            lat_in_flat = th.flatten(self.tensors.pk_lat_in,start_dim=2)
            
            input_ = th.cat((self.tensors.pk_dyn_in, lat_in_flat),2)

        return input_

    def forward(self, dyn_in):
        """
        Runs the forward pass of all PKs and TKs, respectively, in parallel 
        for a given input

        """ 
        
        # Write the dynamic PK input to the corresponding tensor
        self.tensors.pk_dyn_in = dyn_in
       
        input_ = self._graph_connections()

        # Forward the PK inputs through the pk_net to get the outputs and hidden
        # states of these PKs
        pk_output, pk_lstm_h, pk_lstm_c = self.pk_net.forward(
            input_flat= input_, # Size: [B, PK,  N*Lat + Dyn]
            old_h= self.tensors.pk_lstm_h,  # Size: [B, PK, LSTM]
            old_c= self.tensors.pk_lstm_c # Size: [B, PK, LSTM]
        )
        # pk_output: [B, PK, DYN + N*LAT]

        # Dynamic output
        pk_dyn_out = pk_output[:, :,  :self.config.pk_dyn_size]

        # Lateral output flattened
        pk_lat_out = pk_output[:, :, self.config.pk_dyn_size:]

        # Update the output and hidden state tensors of the PKs
        self.tensors.pk_dyn_out = pk_dyn_out
        self.tensors.pk_lat_out = pk_lat_out
        self.tensors.pk_lstm_h = pk_lstm_h
        self.tensors.pk_lstm_c = pk_lstm_c 
        

    def reset(self, batch_size):
        self.tensors.set_batch_size_and_reset(batch_size)

    def _build_connections(self, rows, cols):

        # Initialize an adjacency matrix for the PK-TK connections
        pk_adj_mat = th.zeros(size=(2,
                                    rows * cols,
                                    rows * cols),
                                device=self.config.device)
        
        # Define a dictionary that maps directions to numbers
        direction_dict = {"top left": 1, "top": 2, "top right": 3, "left": 4, "right": 5,
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
                        pk_adj_mat[0, pk_id_running, neighbor_idx] = 1
                        pk_adj_mat[1, pk_id_running, neighbor_idx] = \
                            direction_dict[neighbor_direction]

                pk_id_running += 1

        #
        # Set up vectors that describe which lateral output goes to which
        # lateral input
        a = np.where(pk_adj_mat[0].cpu().detach().numpy() == 1)

        # PKs that are to be considered in the lateral update
        self.pos0 = th.from_numpy(a[0]).to(device=self.config.device, dtype=th.long)
        # PK lateral outputs that will be sent to the lateral inputs
        self.coming_from = th.from_numpy(a[1]).to(device=self.config.device,
                                                  dtype=th.long)
        # PK lateral input neurons that will get inputs from the previous time
        # step's lateral output
        self.going_to = (pk_adj_mat[1][a] - 1).to(device=self.config.device,
                                                       dtype=th.long)


    def _prepare_connections(self):

        bincount = th.bincount(self.pos0)
        length = th.max(bincount).item()

        connections = th.zeros((self.config.amount_pks, length, 2)).to(device = self.config.device)-1

        idx_counts = th.zeros((self.config.amount_pks)).to(device = self.config.device, 
                                                                dtype = th.long)

        for k,v in enumerate(self.pos0):
            
            pk_idx = v.item()
           
            position_idx = idx_counts[pk_idx]
            connections[pk_idx][position_idx][0] = self.coming_from[k]
            connections[pk_idx][position_idx][1] = self.going_to[k]
            idx_counts[pk_idx] += 1

        return connections









