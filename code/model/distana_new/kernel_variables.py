import torch as th
import configuration as cfg


class KernelParameters:
    """
    This class holds the parameters of the Kernel Network.
    """

    def __init__(self, amount_pks, device):

        #
        # System parameters
        self.device = device

        #
        # General network parameters
        self.seq_len = cfg.SEQ_LEN

        #
        # PK specific parameters
        self.amount_pks = amount_pks
        self.pk_neighbors = cfg.PK_NEIGHBORS

        # Input sizes (dimensions)
        self.pk_dyn_in_size = cfg.PK_DYN_IN_SIZE
        self.pk_lat_in_size = cfg.PK_LAT_IN_SIZE

        # Layer sizes (number of neurons per layer)
        self.pk_pre_layer_size = cfg.PK_PRE_LAYER_SIZE
        self.pk_num_lstm_cells = cfg.PK_NUM_LSTM_CELLS

        # Output sizes (dimensions)
        self.pk_dyn_out_size = cfg.PK_DYN_OUT_SIZE
        self.pk_lat_out_size = cfg.PK_LAT_OUT_SIZE


class KernelTensors:
    """
    This class holds the tensors of the Kernel Network.
    """

    def __init__(self, params):
        self.params = params

        # Initialize the tensors by calling the reset method (this may not be
        # clean code style, yet it spares lots of lines :p)
        self.reset()

    def reset(self):

        #
        # PK tensors

        batch = cfg.BATCH_SIZE
        pk_num = self.params.amount_pks

        # Inputs
        self.pk_dyn_in = th.zeros(size=(batch,
                                        pk_num,
                                        self.params.pk_dyn_in_size),
                                  device=self.params.device)
        self.pk_lat_in = th.zeros(size=(batch,
                                        pk_num,
                                        self.params.pk_neighbors,
                                        self.params.pk_lat_in_size),
                                  device=self.params.device)

        # LSTM states
        self.pk_lstm_c = th.zeros(size=(batch, pk_num, self.params.pk_num_lstm_cells),
                                  device=self.params.device,
                                  requires_grad=False)
        self.pk_lstm_h = th.zeros(size=(batch, pk_num, self.params.pk_num_lstm_cells),
                                  device=self.params.device,
                                  requires_grad=False)

        # Outputs
        self.pk_dyn_out = th.zeros(size=(batch, pk_num,
                                         self.params.pk_dyn_out_size),
                                   device=self.params.device)
        self.pk_lat_out = th.zeros(size=(batch, pk_num,
                                         self.params.pk_neighbors,
                                         self.params.pk_lat_out_size),
                                   device=self.params.device)


    def detach(self):
        self.pk_dyn_in = self.pk_dyn_in.detach()
        self.pk_lat_in = self.pk_lat_in.detach()

        # LSTM states
        self.pk_lstm_c = self.pk_lstm_c.detach()
        self.pk_lstm_h = self.pk_lstm_h.detach()

        # Outputs
        self.pk_dyn_out = self.pk_dyn_out.detach()
        self.pk_lat_out = self.pk_lat_out.detach()
