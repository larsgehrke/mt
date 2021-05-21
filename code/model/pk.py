import math
import torch as th

from tools.debug import sprint


class PK(th.nn.Module):
    '''
    Custom PyTorch class that implements the Prediction Kernel (PK) of DISTANA in a parallel fashion.
    Note that in DISTANA the PKs share their weights.
    All PKs from all batches should be processed in parallel. 
    The implementation of the forward pass is based on PyTorch´s tensor operations.
    The backward pass is automatically calculated from PyTorch´s autograd feature.
    '''

    def __init__(self,  
        input_size: int, 
        pre_layer_size: int,
        lstm_size: int, 
        output_size: int, 
        device: str):
        '''
        The initialisation of the PK module.
        :param input_size: The size of the input to each PK (dynamical and lateral)
        :param pk_pre_layer_size: The size of the input layer of each PK. 
        :param lstm_size: number of LSTM nodes in the PK.
        :param output_size: The size of the output of each PK (dynamical and lateral)
        :param device: PyTorch specific string to select either CPU('cpu') or GPU ('cuda') for the execution

        '''

        super(PK, self).__init__()
        self.lstm_size = lstm_size

        # starting fc layer weights
        self.W_input = th.nn.Parameter(
            th.Tensor(input_size,pre_layer_size)) 

        # LSTM weights
        self.W_lstm = th.nn.Parameter(
            th.Tensor( pre_layer_size + lstm_size, 4 * lstm_size))

        # ending fc layer weights
        self.W_output = th.nn.Parameter(
            th.Tensor(lstm_size,output_size))


        self.reset_parameters()
        self.to(device)
        

    def reset_parameters(self):
        stdv = 1.0 / math.sqrt(self.lstm_size)
        for weight in self.parameters():
            weight.data.uniform_(-stdv, +stdv)


    def forward(self, input_flat: th.Tensor, old_c: th.Tensor, old_h: th.Tensor) -> list:
        '''
            Forward propagation for all PKs in all batches in parallel.
            
            :param input_flat: 
                The input for the PKs where dynamical input is concatenated with flattened dynamical input.
                Size is [B, PK, DYN + N*LAT] with batch size B, amount of PKs PK, dynamical input size DYN,
                Neighbors N and lateral input size LAT.

                
            :param old_c: the LSTM cell values
            :param old_h: the LSTM h vector

            :return: 
                list[0]: network output for this time step (y_hat)
                list[1]: hidden vector of the LSTM for this time step (h_t)
                list[2]: cell state of the LSTM for this time step (C_t)

        '''
        #sprint(input_flat, "input_flat")
        #sprint(self.W_input, "self.W_input")

        # first fully connected layer
        x_t = th.tanh(th.matmul(input_flat, self.W_input))
        # => th.Size([B, PK, lstm_size])

        # concatination of input and hidden vector 
        X = th.cat([x_t, old_h], dim=2)
        # => th.Size([B, PK, 2*lstm_size])

        # Compute the input, output and candidate cell gates with one MM.
        gate_weights = th.matmul(X, self.W_lstm)

        # Split the combined gate weight matrix into its components.
        gates = gate_weights.chunk(4, dim=2)

        #
        # (normal) LSTM forward pass
        f_t = th.sigmoid(gates[0])
        i_t = th.sigmoid(gates[1])
        o_t = th.sigmoid(gates[2])
        Ctilde_t = th.tanh(gates[3])

        C_t = f_t * old_c + i_t * Ctilde_t
        # => th.Size([B, PK, lstm_size])

        h_t = th.tanh(C_t) * o_t
        # => th.Size([B, PK, lstm_size])

        # last fully connected layer
        y_hat_ = th.matmul(h_t, self.W_output)
        y_hat = th.tanh(y_hat_)
        # => th.Size([B, PK, DYN + LAT])


        return y_hat, h_t, C_t

        