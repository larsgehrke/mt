
class KernelConfig:
    """
    This class holds the parameters of the Kernel Network.
    Thus the implementation of the model can depend on this class
    with specific class attributs instead of a mysterious dictionary object.
    The advantage is to have clear, transparent dependencies.
    """

    def __init__(self, params: dict):
        '''
        Initialisation of KernelConfig.
        :param params: parameters that are saved in this class
        '''

        #
        # System parameters
        self.device = params["device"]
        self.use_gpu = params["use_gpu"]

        self.batch_size_train = int(params["batch_size_train"])
        self.batch_size_test = int(params["batch_size_test"])

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
        
        # Hard coded Hyperparameter: PK Neighbors
        # needed e.g. for the calculation of the input tensor shapes
        self.pk_neighbors = 8

        # Input sizes (dimensions)
        self.pk_dyn_size = int(params["pk_dyn_size"])
        self.pk_lat_size = int(params["pk_lat_size"])

        # Layer sizes (number of neurons per layer)
        self.pk_pre_layer_size = int(params["pk_pre_layer_size"])
        self.pk_num_lstm_cells = int(params["pk_num_lstm_cells"])