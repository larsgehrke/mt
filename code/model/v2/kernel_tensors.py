import torch as th

class KernelTensors:
    """
    This class holds the tensors of the Kernel Network.
    """

    def __init__(self, config):
        self.config = config

        # set batch_size temporarily,
        # can be overwritten with self.set_batch_size_and_reset(batch_size)
        self.batch_size = config.batch_size_train 

        # Initialize the tensors by calling the reset method 
        self.reset()

    def set_batch_size_and_reset(self, batch_size):
        self.batch_size = batch_size
        self.reset()

    def reset(self):

        #
        # PK tensors

        # Inputs
        self.pk_dyn_in = th.zeros(size=(self.batch_size,
                                        self.config.amount_pks,
                                        self.config.pk_dyn_size),
                                  device=self.config.device)

        self.pk_lat_in = th.zeros(size=(self.batch_size,
                                        self.config.amount_pks,
                                        self.config.pk_neighbors,
                                        self.config.pk_lat_size),
                                  device=self.config.device)

        # LSTM states
        self.pk_lstm_c = th.zeros(size=(self.batch_size, 
                                        self.config.amount_pks, 
                                        self.config.pk_num_lstm_cells),
                                  device=self.config.device)

        self.pk_lstm_h = th.zeros(size=(self.batch_size, 
                                        self.config.amount_pks, 
                                        self.config.pk_num_lstm_cells),
                                  device=self.config.device)

        # Outputs
        self.pk_dyn_out = th.zeros(size=(self.batch_size, 
                                         self.config.amount_pks,
                                         self.config.pk_dyn_size),
                                   device=self.config.device)
        
        self.pk_lat_out = th.zeros(size=(self.batch_size, 
                                         self.config.amount_pks,
                                         self.config.pk_lat_size),
                                   device=self.config.device)

