import math
import torch as th

from tools.debug import sprint


class PK(th.nn.Module):
    def __init__(self, batch_size, amount_pks, input_size, lstm_size, output_size, device):
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
            th.Tensor(lstm_size,output_size))


        self.reset_parameters()
        self.to(device)
        

    def reset_parameters(self):
        stdv = 1.0 / math.sqrt(self.lstm_size)
        for weight in self.parameters():
            weight.data.uniform_(-stdv, +stdv)

    def set_batch_size(self, batch_size):
        self.lstm_h = th.zeros(batch_size,self.amount_pks,self.lstm_size)
        self.lstm_c = th.zeros(batch_size,self.amount_pks,self.lstm_size)

        self.batch_size = batch_size


    def forward(self, input_flat, old_c, old_h):
        '''
            Forward propagation for all PKs in all batches in parallel.
            
            :param input_flat: 
                The input for the PKs where dynamical input is concatenated with flattened dynamical input.
                Size is [B, PK, DYN + N*LAT] with batch size B, amount of PKs PK, dynamical input size DYN,
                Neighbors N and lateral input size LAT.

                
            :param old_c: the LSTM cell values
            :param old_h: the LSTM h vector

        '''

        x_t = th.tanh(th.matmul(input_flat, self.W_input))
        # => th.Size([B, PK, lstm_size])

        X = th.cat([x_t, old_h], dim=2)
        # => th.Size([B, PK, 2*lstm_size])

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
        # => th.Size([B, PK, lstm_size])

        h_t = th.tanh(C_t) * o_t
        # => th.Size([B, PK, lstm_size])

        y_hat_ = th.matmul(h_t, self.W_output)
        y_hat = th.tanh(y_hat_)
        # => th.Size([B, PK, DYN + LAT])


        return y_hat, h_t, C_t