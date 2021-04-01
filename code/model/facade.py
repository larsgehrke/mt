import torch as th
import numpy as np
from config.kernel_config import KernelConfig

class Facade():
    '''
    Access point for usage of the model versions. 
    This method automatically selects the model version specified by the given parameters.
    '''

    def __init__(self, params: dict):
        '''
        Intitialisation of the Facade. Selection of the specified model version.
        :param params: parameter dict
        '''

        if params["model_name"] == "old":
            import model.old.evaluator as model
        elif params["model_name"] == "old2":
            import model.old2.evaluator as model
        elif params["model_name"] == "v1a":
            import model.v1a.evaluator as model
        elif params["model_name"] == "v1b":
            import model.v1b.evaluator as model
        elif params["model_name"] == "v2":
            import model.v2.evaluator as model
        elif params["model_name"] == "v3":
            import model.v3.evaluator as model
        else:
            raise ValueError("Model name is not valid.")

        config = KernelConfig(params)

        self.model = model.Evaluator(config)

    def set_training(self, 
        train_data: np.ndarray, 
        optimizer: th.optim.Optimizer, 
        criterion) -> int:
        '''
        Set the training configuration.
        :param train_data: array of training data file names
        :param optimizer: PyTorch Optimizer
        :param criterion: callable PyTorch Loss 
        :return: amount of iterations per epoch
        '''
        return self.model.set_training(train_data,optimizer, criterion)

    def set_testing(self, test_data: np.ndarray, 
        criterion, 
        teacher_forcing_steps: int) -> int:
        '''
        Set the testing configuration.
        :param test_data: array of testing data file names
        :param criterion: callable PyTorch Loss
        :param teacher_forcing_steps: amount of time steps for teacher forcing
        :return: amount of iterations per epoch
        '''
        return self.model.set_testing(test_data, criterion, teacher_forcing_steps)

    def net(self) -> th.nn.Module:
        '''
        Get the network class.
        :return: the network (PyTorch module)
        '''
        return self.model.net

    def config(self):
        '''
        Get the configuration object.
        :return: the configuration object
        '''
        return self.model.config

    def get_trainable_params(self) -> int:
        '''
        Get the amount of trainable params.
        :return: amount of trainable params
        '''
        return self.model.get_trainable_params()

    def set_weights(self,loaded_weights, is_training: bool):
        '''
        Set the network weights.
        :param loaded_weights: the network weights which should be loaded
        :param is_training: choose training or testing mode
        '''
        print("type(loaded_weights)", type(loaded_weights))
        self.model.set_weights(loaded_weights,is_training)

    def train(self, iter_idx:int) -> float:
        '''
        Do the training of the model with the given iteration index.
        :param iter_idx: index of iteration of the current epoch
        :return: training error
        '''
        return self.model.train(iter_idx)

    def test(self, iter_idx:int, return_only_error:bool=True) -> list:
        '''
        Do the testing of the model with the given iteration index.
        :param iter_idx: index of the test data iteration
        :param return_only_error: decide amount of returning values
        :return: 
            list[0]: testing error
            list[1]: network output
            list[2]: network label
            list[3]: network input
        '''
        return self.model.test(iter_idx, return_only_error)




















