import math
import numpy as np
import torch as th

# This modle is based on the th model
from model.th_base import BaseEvaluator

from model.th2.kernel_net import KernelNetwork
from model.th2.kernel_tensors import KernelTensors

from tools.debug import sprint


class Evaluator(BaseEvaluator):

    def __init__(self, kernel_config):
        
        tensors = KernelTensors(kernel_config)
        net = KernelNetwork(kernel_config, tensors)

        super().__init__(kernel_config,tensors, net)


    def train(self, iter_idx):
        
        self.is_testing = False

        if self.train_filenames is None or self.optimizer is None \
            or self.train_criterion is None:
                raise ValueError("Missing the training configuration: Data File names, Optimizer and/or Criterion.")

        net_input, net_label, batch_size = self._set_up_batch(iter_idx = iter_idx)

        # Set the gradients back to zero
        self.optimizer.zero_grad()

        net_outputs = self._evaluate(self._np_to_th(net_input), batch_size)

        mse = self.train_criterion(net_outputs, self._np_to_th(net_label))
        # Alternatively, the mse can be calculated 'manually'
        # mse = th.mean(th.pow(net_outputs - th.from_numpy(net_label), 2))

        # backward pass
        mse.backward()
        self.optimizer.step()

        return mse.item() # return only the number, not the th object
        

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




