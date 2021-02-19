import torch as th
import torch.nn as nn

import helper_functions as helpers

class PredictionKernelNet(nn.Module):
    """
    This class represents the shared-weights-network for all Prediction Kernels
    """

    def __init__(self, params):

        super(PredictionKernelNet, self).__init__()

        self.params = params

        #
        # Define the weights of the shared PK

        # Dynamic and lateral input preprocessing layer weights
        self.pre_weights = nn.Linear(
            in_features=params.pk_dyn_in_size +
                        (params.pk_lat_in_size * params.pk_neighbors),
            out_features=params.pk_pre_layer_size,
            bias=False
        ).to(device=self.params.device)

        # Central LSTM layer
        self.lstm = nn.LSTMCell(
            input_size=params.pk_pre_layer_size,
            hidden_size=params.pk_num_lstm_cells,
            bias=False
        ).to(device=self.params.device)

        # Postprocessing layer weights
        self.post_weights = nn.Linear(
            in_features=params.pk_num_lstm_cells,
            out_features=params.pk_dyn_out_size +
                         (params.pk_lat_out_size * params.pk_neighbors),
            bias=False
        ).to(device=self.params.device)

        # self.weights = nn.Linear(
        #     in_features=params.pk_dyn_in_size +
        #                 (params.pk_lat_in_size * params.pk_neighbors),
        #     out_features=params.pk_dyn_out_size + params.pk_lat_out_size,
        #     bias=False
        # )

    def forward(self, dyn_in, lat_in, stat_in=None, lstm_c=None,
                lstm_h=None):

        # Flatten the last two dimensions of the lateral input such that it has
        # the correct dimensionality for the forward pass
        lat_in = lat_in.view(
            size=(self.params.amount_pks,
                  self.params.pk_neighbors * self.params.pk_lat_in_size)
        )

        #
        # Forward the activities through the shared PK
        pre_act = th.tanh(
            self.pre_weights(th.cat(tensors=(dyn_in, lat_in), dim=1))
        )

        lstm_c, lstm_h = self.lstm(pre_act, (lstm_c, lstm_h))

        # Postprocessing layer activation
        post_act = th.tanh(self.post_weights(lstm_h_))


        # Dynamic output
        dyn_out = post_act[:, :self.params.pk_dyn_out_size]


        # Lateral output
        lat_out = post_act[:, self.params.pk_dyn_out_size:]
        

        # Unflatten the last dimension of the lateral output such that it has
        # the correct dimensionality for the further processing
        lat_out = lat_out.view(size=(self.params.amount_pks,
                                     self.params.pk_neighbors,
                                     self.params.pk_lat_out_size))

        return dyn_out, lat_out, lstm_c, lstm_h_
