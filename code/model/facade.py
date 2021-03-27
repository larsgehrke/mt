import torch as th
from config.kernel_config import KernelConfig

class Facade():

    def __init__(self, params):

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

    def set_training(self, train_data, optimizer, criterion):
        return self.model.set_training(train_data,optimizer, criterion)

    def set_testing(self, test_data, criterion, teacher_forcing_steps):
        return self.model.set_testing(test_data, criterion, teacher_forcing_steps)

    def net(self):
        return self.model.net

    def get_trainable_params(self):
        return self.model.get_trainable_params()

    def set_weights(self,loaded_weights, is_training):
        self.model.set_weights(loaded_weights,is_training)

    def train(self, iter_idx):
        return self.model.train(iter_idx)

    def test(self, iter_idx, return_only_error=True):
        return self.model.test(iter_idx, return_only_error)




















