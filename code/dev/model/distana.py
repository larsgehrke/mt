import torch as th
from config.kernel_config import KernelConfig


class DISTANA():


    def __init__(self, params):

        if params["model_name"] == "old":
            import model.old.run as model
            params = self._disable_batch(params)
        elif params["model_name"] == "old2":
            import model.old2.run as model
            params = self._disable_batch(params)
        elif params["model_name"] == "th":
            import model.th.run as model
        elif params["model_name"] == "th2":
            import model.th2.run as model
        else:
            raise ValueError("Model name is not valid.")

        self.model = model.Evaluator(KernelConfig(params))

    def set_training(self, train_data, optimizer, criterion):
        return self.model.set_training(train_data,optimizer, criterion)

    def set_testing(self, test_data, criterion, teacher_forcing_steps):
        return self.model.set_testing(test_data, criterion, teacher_forcing_steps)

    def net(self):
        return self.model.net

    def get_trainable_params(self):
        # Count number of trainable parameters
        pytorch_total_params = sum(
            p.numel() for p in self.net().parameters() if p.requires_grad
        )
        return pytorch_total_params

    def set_weights(self,loaded_weights, is_training):
        self.model.set_weights(loaded_weights,is_training)

    def train(self, iter_idx):
        return self.model.train(iter_idx)

    def test(self, iter_idx, return_only_error=True):
        return self.model.test(iter_idx, return_only_error)

    def _disable_batch(self, params):
        params["batch_size_train"] = 1
        params["batch_size_test"] = 1

        return params