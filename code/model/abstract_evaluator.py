import math
import numpy as np
import torch as th

from config.kernel_config import KernelConfig

class AbstractEvaluator():
    '''
    This class is an abstract super class for the evaluator of the different versions.
    This code was outsourced to reduce the amount of duplicated code, 
    because most of the model versions shared the same implementation framework.
    The most important difference of the model versions is, wheather the version
    is able to process batches. Depending on this, different code segments are executed.
    '''

    def __init__(self, 
        config: KernelConfig, 
        net: th.nn.Module, 
        batch_processing: bool):
        '''
        Initialisation of AbstractEvaluator.
        :param config: container object with all important configuration values
        :param net: the network class
        :param batch_processing: wheater the model is able to process batches
        '''

        self.config = config
        self.net = net
        self.batch_processing = batch_processing

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

        if self.batch_processing:
            return math.ceil(len(train_filenames)/self.config.batch_size_train)
        else:
            return len(train_filenames)

    def set_testing(self, test_filenames, criterion, teacher_forcing_steps):
        self.test_filenames = test_filenames
        self.test_criterion = criterion
        self.teacher_forcing_steps = teacher_forcing_steps

        if self.batch_processing:
            return math.ceil(len(test_filenames)/self.config.batch_size_test)
        else:
            return len(test_filenames)

    def train(self, iter_idx):
        self.is_testing = False

        if self.train_filenames is None or self.optimizer is None \
            or self.train_criterion is None:
                raise ValueError("Missing the training configuration: data file names, optimizer and/or criterion.")

        return self._train(iter_idx)


    def test(self, iter_idx, return_only_error=True):
        self.is_testing = True

        if self.test_filenames is None or self.test_criterion is None \
            or self.teacher_forcing_steps is None:
            raise ValueError("Missing the testing configuration: data file names, criterion and/or amount teacher forcing steps.")

        return self._test(iter_idx, return_only_error)


    def _train(self, iter_idx):
        # Set the gradients back to zero
        self.optimizer.zero_grad()

        net_output = None
        if self.batch_processing:
            net_input, net_label, batch_size = self._load_data(iter_idx = iter_idx)
            net_output = self._evaluate(self._np_to_th(net_input), batch_size)
        else:
            net_input, net_label = self._load_data(iter_idx = iter_idx)
            net_output = self._evaluate(self._np_to_th(net_input))

        mse = self.train_criterion(net_output, self._np_to_th(net_label))
        # Alternatively, the mse can be calculated 'manually'
        # mse = th.mean(th.pow(net_output - th.from_numpy(net_label), 2))

        # backward pass
        mse.backward()
        self.optimizer.step()

        return mse.item() # return only the number, not the th object


    def _test(self, iter_idx, return_only_error):

        net_output = None
        if self.batch_processing:
            net_input, net_label, batch_size = self._load_data(iter_idx = iter_idx)
            net_output = self._evaluate(self._np_to_th(net_input), batch_size)
        else:
            net_input, net_label = self._load_data(iter_idx = iter_idx)
            net_output = self._evaluate(self._np_to_th(net_input))

        mse = self.test_criterion(net_output, self._np_to_th(net_label))
        # Alternatively, the mse can be calculated 'manually'
        # mse = th.mean(th.pow(net_output - th.from_numpy(net_label), 2))

        if return_only_error:
            return mse.item()
        else:
            # Convert outgoing objects from PyTorch to NumPy
            net_output = self._th_to_np(net_output)

            if self.batch_processing:
                return mse.item(), net_output, net_label, net_input
            else:
                # The model cannot handle batches, so we need to manually add a batch dimension
                # with batch size = 1 for the further processing
                return mse.item(), np.expand_dims(net_output,0), np.expand_dims(net_label,0), np.expand_dims(net_input,0)


    def _evaluate(self, iter_idx):
        raise NotImplementedError("This is an abstract class with no concrete implementation of the evaluate function.")


    def _evaluate(self, iter_idx, batch_size):
        raise NotImplementedError("This is an abstract class with no concrete implementation of the evaluate function.")

    
    def set_weights(self,loaded_weights, is_training):
        self.net.load_state_dict(loaded_weights)
        if is_training:
            self.net.train()
        else:
            self.net.eval()


    def _np_to_th(self, x_np):
        return th.from_numpy(x_np).to(self.config.device)

    def _th_to_np(self, x_th):
        return x_th.cpu().detach().numpy()

    def _load_data(self, iter_idx):
        # Shuffle data at the beginning of the epoch
        if iter_idx == 0:
            if self.is_testing:
                np.random.shuffle(self.test_filenames) # in place operation!
            else:
                np.random.shuffle(self.train_filenames) # in place operation!

        data_files = None
        
        if self.is_testing:
            data_files = self.test_filenames
        else:
            data_files = self.train_filenames

        if self.batch_processing:
            data = self._load_data_batch(data_files, iter_idx)
            return self._prepare_data_batch(data)
        else:
            data = self._load_data_single(data_files, iter_idx)
            return self._prepare_data_single(data)

    def _load_data_batch(self, data_files, iter_idx):
        batch_size = None

        if self.is_testing:
            batch_size = self.config.batch_size_test
        else:
            batch_size = self.config.batch_size_train

        seq_len = self.config.seq_len

        first_sample = batch_size * iter_idx

        # Handling also last batch
        last_sample_excl = min(first_sample + batch_size, len(data_files))
  
  
        data = np.load(data_files[first_sample])[:seq_len + 1]
        # Expand Dim for batch 
        data = data[np.newaxis, :]

        for file in data_files[first_sample+1:last_sample_excl]:
            data_file = np.load(file)[:seq_len + 1]
            # Expand Dim for batch 
            data_file = data_file[np.newaxis, :]
            data = np.append(data, data_file, axis=0)

        return data

    def _load_data_single(self, data_files, iter_idx):
        seq_len = self.config.seq_len
        
        first_sample = iter_idx
  
        data = np.load(data_files[first_sample])[:seq_len + 1]

        return data

    def _prepare_data_batch(self, data):
        pks = self.config.amount_pks

        # Get first, second and third dimension of data
        dim0, dim1, dim2 = np.shape(data)[:3]

        # Reshape the data array to have the kernels on one dimension
        data = np.reshape(data, [dim0, dim1, dim2, pks])

        # Swap the third and fourth dimension of the data
        data = np.swapaxes(data, axis1=2, axis2=3)

        # Split the data into inputs (where some noise is added) and labels
        # Add noise to all timesteps except very last one
        noise = self.config.data_noise # it does not matter if it is train, val or test!

        _net_input = np.array(
            data[:,:-1] + np.random.normal(0, noise, np.shape(data[:,:-1])),
            dtype=np.float32
        )

        _net_label = np.array(data[:,1:, :, 0:1], dtype=np.float32)

        if not self.is_testing:
            # Set the dynamic inputs with a certain probability to zero to force
            # the network to use lateral connections
            _net_input *= np.array(
                np.random.binomial(n=1, p=1 - self.config.p_zero_input,
                                   size=_net_input.shape),
                dtype=np.float32
            )

        _batch_size = len(_net_input)

        return _net_input, _net_label, _batch_size


    def _prepare_data_single(self, data):
        # Get first and second dimension of data
        dim0, dim1 = np.shape(data)[:2]

        # Reshape the data array to have the kernels on one dimension
        data = np.reshape(data, [dim0, dim1, self.config.pk_rows * self.config.pk_cols])
        # data.shape = (T, 2, PK)
        

        # Swap the second and third dimension of the data
        data = np.swapaxes(data, axis1=1, axis2=2)
        # shape = (T, PK, 2)

        # Split the data into inputs (where some noise is added) and labels
        # Add noise to all timesteps except very last one
        noise = self.config.data_noise # it does not matter if it is train, val or test!

        _net_input = np.array(
            data[:-1] + np.random.normal(0, noise, np.shape(data[:-1])),
            dtype=np.float32
        )
        # shape: (T-1, PK, 2)

        _net_label = np.array(data[1:, :, 0:1], dtype=np.float32)
        # shape: (T-1, PK, 1)
        

        if not self.is_testing:
            # Set the dynamic inputs with a certain probability to zero to force
            # the network to use lateral connections
            _net_input *= np.array(
                np.random.binomial(n=1, p=1 - self.config.p_zero_input,
                                   size=_net_input.shape),
                dtype=np.float32
            )

        return _net_input, _net_label


    def get_trainable_params(self):
        
        # Count number of trainable parameters
        pytorch_total_params = sum(
            p.numel() for p in self.net.parameters() if p.requires_grad
        )
        return pytorch_total_params
        







