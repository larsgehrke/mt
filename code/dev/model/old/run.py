import math
import numpy as np
import torch as th

from model.old.kernel_net import KernelNetwork
from model.old.kernel_tensors import KernelTensors

from tools.debug import sprint


class Evaluator():

    def __init__(self, kernel_config):
        self.config = kernel_config
        self.tensors = KernelTensors(kernel_config)
        self.net = KernelNetwork(kernel_config,self.tensors)

        # Train or Test mode
        self.is_testing = False

        # Used for Training
        self.train_filenames = None 
        self.optimizer = None 
        self.train_criterion = None 

        # Used for Testing
        self.test_criterion = None 
        self.test_filenames = None 
        self.teacher_forcing_steps = None 
        

    def set_training(self, train_filenames, optimizer, criterion):
        self.train_filenames = train_filenames
        self.optimizer = optimizer
        self.train_criterion = criterion

        return self._get_iters(train_filenames)

    def set_testing(self, test_filenames, criterion, teacher_forcing_steps):
        self.test_filenames = test_filenames
        self.test_criterion = criterion
        self.teacher_forcing_steps = teacher_forcing_steps

        return self._get_iters(test_filenames)

    def train(self, iter_idx):
        self.is_testing = False

        if self.train_filenames is None or self.optimizer is None \
            or self.train_criterion is None:
                raise ValueError("Missing the training configuration: Data File names, Optimizer and/or Criterion.")

        net_input, net_label = self._set_up_batch(iter_idx = iter_idx)

        # Set the gradients back to zero
        self.optimizer.zero_grad()

        net_outputs = self._evaluate(self._np_to_th(net_input))

        mse = self.train_criterion(net_outputs, self._np_to_th(net_label))
        # Alternatively, the mse can be calculated 'manually'
        # mse = th.mean(th.pow(net_outputs - th.from_numpy(net_label), 2))

        # backward pass
        mse.backward()
        self.optimizer.step()

        return mse.item() # return only the number, not the th object


    def test(self, iter_idx, return_only_error=True):
        self.is_testing = True

        if self.test_filenames is None or self.test_criterion is None \
            or self.teacher_forcing_steps is None:
            raise ValueError("Missing the testing configuration: Data File names, Criterion and/or Amount teacher forcing steps.")

        net_input, net_label = self._set_up_batch(iter_idx = iter_idx)

        net_outputs = self._evaluate(self._np_to_th(net_input))

        mse = self.test_criterion(net_outputs, self._np_to_th(net_label))
        # Alternatively, the mse can be calculated 'manually'
        # mse = th.mean(th.pow(net_outputs - th.from_numpy(net_label), 2))

        if return_only_error:
            return mse.item()
        else:
            # Convert outgoing objects from PyTorch to NumPy
            net_outputs = self._th_to_np(net_outputs)

            # This model cannot handle batches, so we need to manually add a batch dimension
            # with batch size = 1 for the further processing
            return mse.item(), np.expand_dims(net_outputs,0), 
                np.expand_dims(net_label), np.expand_dims(net_input)

    def _evaluate(self, net_input):

        seq_len = self.config.seq_len
        amount_pks = self.config.amount_pks
        pk_dyn_size = self.config.pk_dyn_size


        # Set up an array of zeros to store the network outputs
        net_outputs = th.zeros(size=(seq_len,                              
                                     amount_pks,
                                     pk_dyn_size),
                              device=self.config.device)

        
        # Reset the network to clear the previous sequence
        self.net.reset()

        # Iterate over the whole sequence of the training example and perform a
        # forward pass
        for t in range(seq_len):

            # Prepare the network input for this sequence step
            if self.is_testing and t > self.teacher_forcing_steps:
                #
                # Closed loop - receiving the output of the last time step as
                # input
                dyn_net_in_step = net_outputs[t-1,:,:pk_dyn_size]
                
            else:
                #
                # Teacher forcing
                #
                # Set the dynamic input for this iteration
                dyn_net_in_step = net_input[t, :, :pk_dyn_size]

                # [B, PK, DYN]

            # Forward the input through the network
            self.net.forward(dyn_in=dyn_net_in_step)

            # Just saving the output of the current time step
            net_outputs[t,:,:] = self.tensors.pk_dyn_out

        return net_outputs

    def set_weights(self,loaded_weights, is_training):
        self.net.load_state_dict(loaded_weights)
        if is_training:
            self.net.train()
        else:
            self.net.eval()


    def _set_up_batch(self, iter_idx):
        """
            Create the batch data for the current training/testing iteration in the epoch.

            Returns NumPy objects

        """
        
        # Shuffle data at the beginning of the epoch
        if iter_idx == 0:
            if self.is_testing:
                np.random.shuffle(self.test_filenames) # in place operation!
            else:
                np.random.shuffle(self.train_filenames) # in place operation!

        data_all = None

        if self.is_testing:
            data_all = self.test_filenames
        else:
            data_all = self.train_filenames

        seq_len = self.config.seq_len
        pks = self.config.amount_pks

        first_sample = iter_idx
  
  
        data = np.load(data_all[first_sample])[:seq_len + 1]

        # Get first and second dimension of data
        dim0, dim1 = np.shape(data)[:2]

       

        # Reshape the data array to have the kernels on one dimension
        data = np.reshape(data, [dim0, dim1, self.config.pk_rows * self.config.pk_cols])
        # data.shape = (41, 2, 256)
        

        # Swap the second and third dimension of the data
        data = np.swapaxes(data, axis1=1, axis2=2)
        # shape = (41, 256, 2)

        # Split the data into inputs (where some noise is added) and labels
        # Add noise to all timesteps except very last one
        noise = 0 if self.is_testing else self.config.data_noise
        _net_input = np.array(
            data[:-1] + np.random.normal(0, noise, np.shape(data[:-1])),
            dtype=np.float32
        )
        # shape: (40, 256, 2)

        _net_label = np.array(data[1:, :, 0:1], dtype=np.float32)
        # shape: (40, 256, 1)
        

        if not self.is_testing:
            # Set the dynamic inputs with a certain probability to zero to force
            # the network to use lateral connections
            _net_input *= np.array(
                np.random.binomial(n=1, p=1 - self.config.p_zero_input,
                                   size=_net_input.shape),
                dtype=np.float32
            )

        return _net_input, _net_label



    def _get_iters(self, filenames):
        # Return amount of iterations per epoch
        return len(filenames)

    def _np_to_th(self, x_np):
        return th.from_numpy(x_np).to(self.config.device)

    def _th_to_np(self, x_th):
        return x_th.cpu().detach().numpy()




