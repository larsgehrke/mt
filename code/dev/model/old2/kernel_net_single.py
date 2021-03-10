import numpy as np
import torch as th
import torch.nn as nn
import prediction_kernel
import configuration as cfg


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
        self.pk_net = prediction_kernel.PredictionKernelNet(params=params)

        

    def forward(self, dyn_in, pk_stat_in=None, is_first_of_seq=False):
        """
        Runs the forward pass of all PKs and TKs, respectively, in parallel for
        a given input
        :param dyn_in: The dynamic input for the PKs
        :param pk_stat_in: The static input for the PKs
        :param is_first_of_seq: Boolean indicating whether this is the first
         iteration of a new sequence
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
        self.tensors.pk_lat_in[self.pos0, self.going_to] =\
            self.tensors.pk_lat_out[self.coming_from].t()

        # Forward the PK inputs through the pk_net to get the outputs and hidden
        # states of these PKs
        pk_dyn_out, pk_lat_out, pk_lstm_c, pk_lstm_h = self.pk_net.forward(
            dyn_in=self.tensors.pk_dyn_in,
            lat_in=self.tensors.pk_lat_in,
            stat_in=pk_stat_in,
            lstm_c=self.tensors.pk_lstm_c,
            lstm_h=self.tensors.pk_lstm_h,
            is_first_of_seq=is_first_of_seq,
        )

        # Update the output and hidden state tensors of the PKs
        self.tensors.pk_dyn_out = pk_dyn_out
        self.tensors.pk_lat_out = pk_lat_out
        self.tensors.pk_lstm_c = pk_lstm_c
        self.tensors.pk_lstm_h = pk_lstm_h

    def reset(self):
        self.tensors.reset()
