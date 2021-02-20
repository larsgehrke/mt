import math
import torch as th

import numpy as np
import sys
import os 

from helper_functions import sprint

th.manual_seed(42)


class PK(th.nn.Module):
    def __init__(self, batch_size, amount_pks, input_size, lstm_size, device):
        super(PK, self).__init__()
        self.batch_size = batch_size
        self.amount_pks = amount_pks
        self.input_size = input_size
        self.lstm_size = lstm_size

        self.lstm_h = th.zeros(batch_size,amount_pks,lstm_size,
                                device = device)
        self.lstm_c = th.zeros(batch_size,amount_pks,lstm_size,
                                device = device)

        # starting fc layer weights
        self.W_input = th.nn.Parameter(
            th.Tensor(input_size,lstm_size)).to(device=device) 

        # LSTM weights
        self.W_f = th.nn.Parameter(
            th.Tensor(lstm_size,lstm_size)).to(device=device)

        self.W_i = th.nn.Parameter(
            th.Tensor(lstm_size,lstm_size)).to(device=device)

        self.W_o = th.nn.Parameter(
            th.Tensor(lstm_size,lstm_size)).to(device=device)

        self.W_c = th.nn.Parameter(
            th.Tensor(lstm_size,lstm_size)).to(device=device)


        self.Q_f = th.nn.Parameter(
            th.Tensor(lstm_size,lstm_size)).to(device=device)

        self.Q_i = th.nn.Parameter(
            th.Tensor(lstm_size,lstm_size)).to(device=device)

        self.Q_o = th.nn.Parameter(
            th.Tensor(lstm_size,lstm_size)).to(device=device)

        self.Q_c = th.nn.Parameter(
            th.Tensor(lstm_size,lstm_size)).to(device=device)

        # ending fc layer weights
        self.W_output = th.nn.Parameter(
            th.Tensor(lstm_size,input_size)).to(device=device)

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

    def forward(self, input_, old_c, old_h):


        # Flatten the last two input dims
        input_flat = th.flatten(input_, start_dim=2)
        # => th.Size([10, 256, 9])

        x_t = th.tanh(th.matmul(input_flat, self.W_input))
        # => th.Size([10, 256, 16])

        # LSTM forward pass
        f_t = th.sigmoid(th.matmul(x_t,self.W_f) + th.matmul(old_h,self.Q_f))
        i_t = th.sigmoid(th.matmul(x_t,self.W_i) + th.matmul(old_h,self.Q_i))
        o_t = th.sigmoid(th.matmul(x_t,self.W_o) + th.matmul(old_h,self.Q_o))
        Ctilde_t = th.tanh(th.matmul(x_t,self.W_c) + th.matmul(old_h,self.Q_c))

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