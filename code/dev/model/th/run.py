import math
import numpy as np
import torch as th

from model.th.kernel_net import KernelNetwork
from model.th.kernel_tensors import KernelTensors


class Evaluator():

    def __init__(self, kernel_config):
        self.config = kernel_config
        self.tensors = KernelTensors(kernel_config)
        self.net = KernelNetwork(kernel_config,self.tensors)

        self.train_filenames = None
        self.optimizer = None
        self.criterion = None

    def set_training(self, train_filenames, optimizer, criterion):
        self.train_filenames = train_filenames
        self.optimizer = optimizer
        self.criterion = criterion

        amount_iterations = math.ceil(len(train_filenames)/self.config.batch_size)
        # Return amount of iterations per epoch
        return amount_iterations

    def train(self, iter_idx):

        if self.train_filenames is None or self.optimizer is None \
            or self.criterion is None:
                raise ValueError("Missing the training configuration: Data File names, Optimizer and/or Criterion.")

        net_input, net_label, batch_size = self._set_up_batch(iter_idx = iter_idx)

        
        # Set the gradients back to zero
        self.optimizer.zero_grad()

        mse, net_outputs = self._evaluate(net_input, net_label, batch_size)

        mse = self.criterion(net_outputs, th.from_numpy(net_label))
        # Alternatively, the mse can be calculated 'manually'
        # mse = th.mean(th.pow(net_outputs - th.from_numpy(net_label), 2))

        # backward pass
        mse.backward()
        self.optimizer.step()

        return mse.item() # return only the number, not the th object


    def test(self, filenames):
        th.autograd.set_detect_anomaly(True)

        net_input, net_label, batch_size = self._set_up_batch(data_all = filenames, all_files = True)

        mse, net_outputs = self._evaluate(net_input, net_label, batch_size)

        mse = self.criterion(net_outputs, th.from_numpy(net_label))
        # Alternatively, the mse can be calculated 'manually'
        # mse = th.mean(th.pow(net_outputs - th.from_numpy(net_label), 2))

        return mse.item()# return only the number, not the th object

    def _evaluate(self, net_input, net_label, batch_size):
        mse = None
        th.autograd.set_detect_anomaly(True)

        seq_len = self.config.seq_len
        amount_pks = self.config.amount_pks
        pk_dyn_out_size = self.config.pk_dyn_out_size



        # Set up an array of zeros to store the network outputs
        net_outputs = th.zeros(size=(batch_size,
                                     seq_len,                              
                                     amount_pks,
                                     pk_dyn_out_size))

        
        # Reset the network to clear the previous sequence
        self.net.reset(batch_size)

        # Iterate over the whole sequence of the training example and perform a
        # forward pass
        for t in range(seq_len):

            # Set the dynamic input for this iteration
            dyn_net_in_step = net_input[:, t, :, :pk_dyn_out_size]


            # Forward the input through the network
            self.net.forward(dyn_in=dyn_net_in_step)

            # (10, 256, 1, 1)
            # Swapping the second with third dimension
            # because time dimension is need
            # (10, 1, 256, 1)
            net_outputs[:,t] = th.transpose(self.tensors.pk_dyn_out, 1, 2)[:,0]

        return mse, net_outputs

    def set_weights(self,loaded_weights):
        print('Loading model (that is the network\'s weights) from file...')
        self.net.load_state_dict(load_state_dict)
        self.net.train()


    def _set_up_batch(self, data_all = None, iter_idx=0, all_files = False, is_train = True):
        """
            Training:
                Create the batch data for the current training iteration in the epoch.
            Testing:
                Create a batch of all test data samples.

        """
        # Shuffle training data at the beginning of the epoch

        if data_all is None:
            data_all = self.train_filenames

        if iter_idx == 0:
            np.random.shuffle(data_all)        

        batch_size = self.config.batch_size
        seq_len = self.config.seq_len
        pks = self.config.amount_pks

        first_sample = self.config.batch_size * iter_idx

        # Handling also last batch
        last_sample_excl = min(first_sample + batch_size, len(data_all))


        if(all_files):
            first_sample = 0
            last_sample_excl = len(data_all)
  

        data = np.load(data_all[first_sample])[:seq_len + 1]
        # Expand Dim for batch 
        data = data[np.newaxis, :]

        for file in data_all[first_sample+1:last_sample_excl]:
            data_file = np.load(file)[:seq_len + 1]
            # Expand Dim for batch 
            data_file = data_file[np.newaxis, :]
            data = np.append(data, data_file, axis=0)

        
        # shape: ( BATCH_SIZE , 41, 2, 16, 16)

        # Get first and second dimension of data
        dim0, dim1, dim2 = np.shape(data)[:3]

        # Reshape the data array to have the kernels on one dimension
        data = np.reshape(data, [dim0, dim1, dim2, pks])

        # Swap the third and fourth dimension of the data
        data = np.swapaxes(data, axis1=2, axis2=3)
        # (8, 41, 256, 2)

        # Split the data into inputs (where some noise is added) and labels
        # Add noise to all timesteps except very last one
        _net_input = np.array(
            data[:,:-1] + np.random.normal(0, self.config.data_noise, np.shape(data[:,:-1])),
            dtype=np.float32
        )

        _net_label = np.array(data[:,1:, :, 0:1], dtype=np.float32)

        if is_train:
            # Set the dynamic inputs with a certain probability to zero to force
            # the network to use lateral connections
            _net_input *= np.array(
                np.random.binomial(n=1, p=1 - self.config.p_zero_input,
                                   size=_net_input.shape),
                dtype=np.float32
            )

        batch_size = len(_net_input)

        # sequenz/time should be first dimension
        # and batch should be second dimension
        # _net_input = np.swapaxes(_net_input, axis1=0, axis2=1) 
        # _net_label = np.swapaxes(_net_label, axis1=0, axis2=1) 


        return _net_input, _net_label, batch_size

