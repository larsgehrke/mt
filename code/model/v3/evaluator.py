'''
v3 [batch processing; single lateral output; flexible lateral connections in CUDA]

As v2, but instead of using a CUDA kernel that is hard coding the lateral connections, 
the CUDA kernel processes adjacency lists that define the lateral connections. 
The adjacency lists are created in Python and passed as a static value to the CUDA code.
'''
import math
import numpy as np
import torch as th

from model.abstract_evaluator import AbstractEvaluator

# use the script in this folder
from model.v3.kernel_net import KernelNetwork

from model.kernel_tensors import KernelTensors


class Evaluator(AbstractEvaluator):

    def __init__(self, kernel_config):

        self.tensors = KernelTensors(kernel_config)

        net = KernelNetwork(kernel_config, self.tensors)
        super().__init__(kernel_config, net, batch_processing = True)
        

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
        
    



