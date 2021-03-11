import math
import numpy as np
import torch as th

from model.th_base import BaseEvaluator

from model.th.kernel_net import KernelNetwork
from model.th.kernel_tensors import KernelTensors


class Evaluator(BaseEvaluator):

    def __init__(self, kernel_config):

        tensors = KernelTensors(kernel_config)
        net = KernelNetwork(kernel_config, tensors)
        
        super().__init__(kernel_config,tensors, net)
        





