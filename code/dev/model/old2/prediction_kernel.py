import torch as th
import torch.nn as nn


class PredictionKernelNet(nn.Module):
    """
    This class represents the shared-weights-network for all Prediction Kernels
    """

    def __init__(self, config):

        super(PredictionKernelNet, self).__init__()

        self.config = config

        #
        # Define the weights of the shared PK

        # Dynamic and lateral input preprocessing layer weights
        self.pre_weights = nn.Linear(
            in_features=config.pk_dyn_size +
                        (config.pk_lat_size * config.pk_neighbors),
            out_features=config.pk_pre_layer_size,
            bias=False
        ).to(device=self.config.device)

        # Central LSTM layer
        self.lstm = nn.LSTMCell(
            input_size=config.pk_pre_layer_size,
            hidden_size=config.pk_num_lstm_cells,
            bias=False
        ).to(device=self.config.device)

        # Postprocessing layer weights
        self.post_weights = nn.Linear(
            in_features=config.pk_num_lstm_cells,
            out_features=config.pk_dyn_size +
                         (config.pk_lat_size),
            bias=False
        ).to(device=self.config.device)


    def forward(self, dyn_in, lat_in, lstm_c=None, lstm_h=None):

        # Flatten the last two dimensions of the lateral input such that it has
        # the correct dimensionality for the forward pass
        lat_in = lat_in.view(
            size=(self.config.amount_pks,
                  self.config.pk_neighbors * self.config.pk_lat_size)
        )

        #
        # Forward the activities through the shared PK
        pre_act = th.tanh(
            self.pre_weights(th.cat(tensors=(dyn_in, lat_in), dim=1))
        )

        lstm_c, lstm_h = self.lstm(pre_act, (lstm_c, lstm_h))

        # Postprocessing layer activation
        post_act = th.tanh(self.post_weights(lstm_h))

        # Dynamic output
        dyn_out = post_act[:, :self.config.pk_dyn_size]

        # Lateral output
        lat_out = post_act[:, self.config.pk_dyn_size:]

        # Unflatten the last dimension of the lateral output such that it has
        # the correct dimensionality for the further processing
        lat_out = lat_out.view(size=(self.config.amount_pks,
                                     self.config.pk_lat_size))

        return dyn_out, lat_out, lstm_c, lstm_h
