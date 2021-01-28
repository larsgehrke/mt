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
            size=(self.params.pk_batches,
                  self.params.pk_neighbors * self.params.pk_lat_in_size)
        )

        helpers.sprint(dyn_in, "prediction_kernel.dyn_in")
        helpers.sprint(lat_in, "prediction_kernel.lat_in")

        #
        # Forward the activities through the shared PK
        pre_act = th.tanh(
            self.pre_weights(th.cat(tensors=(dyn_in, lat_in), dim=1))
        )

        helpers.sprint(pre_act, "prediction_kernel.pre_act")

        lstm_c, lstm_h = self.lstm(pre_act, (lstm_c, lstm_h))

        helpers.sprint(lstm_c, "prediction_kernel.lstm_c")
        helpers.sprint(lstm_h, "prediction_kernel.lstm_h")

        # Postprocessing layer activation
        post_act = th.tanh(self.post_weights(lstm_h))

        helpers.sprint(post_act, "prediction_kernel.post_act")

        # Dynamic output
        dyn_out = post_act[:, :self.params.pk_dyn_out_size]

        helpers.sprint(dyn_out, "prediction_kernel.dyn_out")

        # Lateral output
        lat_out = post_act[:, self.params.pk_dyn_out_size:]

        helpers.sprint(lat_out, "prediction_kernel.lat_out", exit=True)

        # Unflatten the last dimension of the lateral output such that it has
        # the correct dimensionality for the further processing
        lat_out = lat_out.view(size=(self.params.pk_batches,
                                     self.params.pk_neighbors,
                                     self.params.pk_lat_out_size))

        return dyn_out, lat_out, lstm_c, lstm_h