import torch as th
from config.kernel_config import KernelConfig


class DISTANA():


    def __init__(self, params):

        if params["model_name"] == "old":
            import model.old.run as model
        elif params["model_name"] == "th":
            import model.th.run as model
        elif params["model_name"] == "cuda":
            import model.cuda.run as model

        self.model = model.Evaluator(KernelConfig(params))

    def set_training(self,train_data, optimizer, criterion):
        return self.model.set_training(train_data,optimizer, criterion)

    def net(self):
        return self.model.net

    def get_trainable_params(self):
        # Count number of trainable parameters
        pytorch_total_params = sum(
            p.numel() for p in net().parameters() if p.requires_grad
        )
        return pytorch_total_params

    def set_weights(self,loaded_weights):
        self.model.set_weights(loaded_weights)

    def train(self, iter_idx):
        return self.model.train(iter_idx)

    def test(self, data):
        return self.model.test(data)