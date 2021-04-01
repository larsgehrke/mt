'''
v1a [batch processing; stacked lateral output]
   
Based on old, but this implementation can process the samples of one batch 
(with arbitrary batch size) in parallel. 
The Prediction Kernel is implemented as a custom PyTorch class in Python 
with the usage of tensor operations realising the nn layers (fc, lstm, fc). 
In the last iteration per epoch the batch size will probably not fit the rest of the data samples. 
In this case the weight tensors are automatically adapted for this last iteration 
with a special batch size (amount of remaining samples).
'''

import torch as th

from model.abstract_evaluator import AbstractEvaluator

# use the scripts in this folder
from model.v1a.kernel_net import KernelNetwork
from model.v1a.kernel_tensors import KernelTensors


class Evaluator(AbstractEvaluator):

    def __init__(self, kernel_config):
        self.tensors = KernelTensors(kernel_config)

        net = KernelNetwork(kernel_config, self.tensors)
        batch_processing = True
        super().__init__(kernel_config, net, batch_processing)
       

    def _evaluate(self, net_input, batch_size):

        seq_len = self.config.seq_len
        amount_pks = self.config.amount_pks
        pk_dyn_size = self.config.pk_dyn_size


        # Set up an array of zeros to store the network outputs
        net_outputs = th.zeros(size=(batch_size,
                                     seq_len,                              
                                     amount_pks,
                                     pk_dyn_size),
                              device=self.config.device)

        
        # Reset the network to clear the previous sequence
        self.net.reset(batch_size)

        # Iterate over the whole sequence of the training example and perform a
        # forward pass
        for t in range(seq_len):

            # Prepare the network input for this sequence step
            if self.is_testing and t > self.teacher_forcing_steps:
                #
                # Closed loop - receiving the output of the last time step as
                # input
                dyn_net_in_step = net_outputs[:,t-1,:,:pk_dyn_size]
                
            else:
                #
                # Teacher forcing
                #
                # Set the dynamic input for this iteration
                dyn_net_in_step = net_input[:, t, :, :pk_dyn_size]

                # [B, PK, DYN]

            # Forward the input through the network
            self.net.forward(dyn_in=dyn_net_in_step)

            # Just saving the output of the current time step
            net_outputs[:,t,:,:] = self.tensors.pk_dyn_out

        return net_outputs

    








