#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from typing import Tuple
import os
import torch
import center_surround_cuda as csc
from torch.utils.cpp_extension import load

# TODO: use the Just In Time compiler from pytorch to load the module that you
# exported from c++.
center_surround_convolution= load(name="center_surround_convolution",
sources=["center_surround_convolution.cu"], verbose=True)

# e) Load your the exported python module in center surround convolution.py and
# implement the torch.autograd.Function class center surround convolution.
class center_surround_convolution(torch.autograd.Function):
    @staticmethod
    def forward(ctx,
                I: torch.Tensor,
                w_c: torch.Tensor,
                w_s: torch.Tensor,
                w_b: torch.Tensor) -> torch.Tensor:
        # TODO: implement the forward pass using the imported module

        return None  # This leads to test errors

    @staticmethod
    def backward(ctx, dL_dO: torch.Tensor) -> Tuple[torch.Tensor]:
        # TODO: implement the backward pass using the imported module
        return None, None, None, None  # This leads to test errors!


# f) In the same file create a new torch.nn.Module called
# CenterSurroundConvolution which can be used as layer in a neural network.
# TODO: Create the CenterSurroundConvolution Module that represents a layer.
