import math
import os
import numpy as np
import torch as th

from model.abstract_evaluator import AbstractEvaluator

# Important: link to the scripts in this folder!
from model.th2.kernel_net import KernelNetwork
from model.th2.kernel_tensors import KernelTensors


class Evaluator(AbstractEvaluator):

    def __init__(self, kernel_config):

        self.tensors = KernelTensors(kernel_config)

        net = KernelNetwork(kernel_config, self.tensors)
        super().__init__(kernel_config, net, batch_processing = True)

        self.cuda_is_compiled = False
        

    def _evaluate(self, net_input, batch_size):

        seq_len = self.config.seq_len
        amount_pks = self.config.amount_pks
        pk_dyn_size = self.config.pk_dyn_size

        if self.config.use_gpu and not self.cuda_is_compiled:
            self._save_cpp_config()
            self.cuda_is_compiled = True

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

    def _save_cpp_config(self):
        cpp_config_file = os.path.join('model', 'th2', 'include','config.h')

        batch_size = 1
        if self.config.is_testing:
            batch_size = self.config.batch_size_test
        else:
            batch_size = self.config.batch_size_train

        with open(cpp_config_file, 'w') as conf_file:
            conf_file.write("#define BATCH_SIZE " + str(batch_size) + os.linesep)
            conf_file.write("#define PK_ROWS " + str(self.config.pk_rows) + os.linesep)
            conf_file.write("#define PK_COLS " + str(self.config.pk_cols) + os.linesep)
            conf_file.write("#define DIMS 3" + os.linesep)
            conf_file.write("#define NEIGHBORS 8" + os.linesep)
            conf_file.write("#define LAT_SIZE " + str(self.config.pk_lat_size) + os.linesep)
            conf_file.write("#define DYN_SIZE " + str(self.config.pk_dyn_size) + os.linesep)
    



