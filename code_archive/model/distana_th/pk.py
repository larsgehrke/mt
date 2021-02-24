import math
import torch as th

import numpy as np
import sys
import os 

from debug import sprint

th.manual_seed(42)

class PK(th.nn.Module):
    def __init__(self, batch_size, amount_pks, input_size, lstm_size, device):
        super(PK, self).__init__()
        self.amount_pks = amount_pks
        self.input_size = input_size
        self.lstm_size = lstm_size

        self.set_batch_size(batch_size)

        # starting fc layer weights
        self.W_input = th.nn.Parameter(
            th.Tensor(input_size,4)) 

        # LSTM weights
        self.W_lstm = th.nn.Parameter(
            th.Tensor( 4 + lstm_size,4 * lstm_size))

        # ending fc layer weights
        self.W_output = th.nn.Parameter(
            th.Tensor(lstm_size,input_size))

        # 3 * state_size for input gate, output gate and candidate cell gate.
        # input_features + state_size because we will multiply with [input, h].
        #self.weights = th.nn.Parameter(
        #    th.empty(3 * state_size, input_features + state_size))
        #self.bias = th.nn.Parameter(th.empty(3 * state_size))
        self.reset_parameters()
        

    def reset_parameters(self):
        stdv = 1.0 / math.sqrt(self.lstm_size)
        for weight in self.parameters():
            weight.data.uniform_(-stdv, +stdv)

    def set_batch_size(self, batch_size):
        self.lstm_h = th.zeros(batch_size,self.amount_pks,self.lstm_size)
        self.lstm_c = th.zeros(batch_size,self.amount_pks,self.lstm_size)

        self.batch_size = batch_size


    def forward(self, input_, old_c, old_h):


        # Flatten the last two input dims
        input_flat = th.flatten(input_, start_dim=2)
        # => th.Size([10, 256, 9])

        x_t = th.tanh(th.matmul(input_flat, self.W_input))
        # => th.Size([10, 256, 16])

        X = th.cat([x_t, old_h], dim=2)
        # => th.Size([10, 256, 32])

        # Compute the input, output and candidate cell gates with one MM.
        gate_weights = th.matmul(X, self.W_lstm)

        # Split the combined gate weight matrix into its components.
        gates = gate_weights.chunk(4, dim=2)

        # LSTM forward pass
        f_t = th.sigmoid(gates[0])
        i_t = th.sigmoid(gates[1])
        o_t = th.sigmoid(gates[2])
        Ctilde_t = th.tanh(gates[3])

        C_t = f_t * old_c + i_t * Ctilde_t
        # => th.Size([10, 256, 16])

        h_t = th.tanh(C_t) * o_t
        # => th.Size([10, 256, 16])

        y_hat = th.tanh(th.matmul(h_t, self.W_output))
        # => th.Size([10, 256, 9])

        # Unflatten the last dimension of the lateral output such that it has
        # the correct dimensionality for the further processing
        y_hat_unflattened = y_hat.view(size=(self.batch_size,self.amount_pks,self.input_size,1))
        # => th.Size([10, 256, 9, 1])


        return y_hat_unflattened, h_t, C_t