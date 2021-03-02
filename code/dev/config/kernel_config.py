
class KernelConfig:
    """
    This class holds the parameters of the Kernel Network.
    """

    def __init__(self, params):

        #
        # System parameters
        self.device = params["device"]

        self.batch_size = int(params["batch_size"])

        #
        # General network parameters
        self.seq_len = int(params["seq_len"])
        self.p_zero_input = int(params["p_zero_input"])
        self.data_noise = int(params["data_noise"])

        #
        # PK specific parameters
        self.amount_pks = int(params["pk_rows"]) * int(params["pk_cols"]) 
        self.pk_rows = int(params["pk_rows"])
        self.pk_cols = int(params["pk_cols"]) 
        self.pk_neighbors = params["pk_neighbors"]

        # Input sizes (dimensions)
        self.pk_dyn_in_size = int(params["pk_dyn_in_size"])
        self.pk_lat_in_size = int(params["pk_lat_in_size"])

        # Layer sizes (number of neurons per layer)
        self.pk_pre_layer_size = int(params["pk_pre_layer_size"])
        self.pk_num_lstm_cells = int(params["pk_num_lstm_cells"])

        # Output sizes (dimensions)
        self.pk_dyn_out_size = int(params["pk_dyn_out_size"])
        self.pk_lat_out_size = int(params["pk_lat_out_size"])