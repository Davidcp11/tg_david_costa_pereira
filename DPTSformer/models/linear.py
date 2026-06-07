# This code is based on the original implementation by Facebook Research.
# Copyright (c) Facebook, Inc. and its affiliates.
# Licensed under the Creative Commons Attribution-NonCommercial 4.0 International (CC BY-NC 4.0)
# Original source: https://github.com/facebookresearch/TimeSformer/blob/main/timesformer/models/linear.py

""" 
Linear layer (alternate definition)
"""
import torch
import torch.nn.functional as F
from torch import nn as nn


class Linear(nn.Linear):
    def forward(self, input: torch.Tensor) -> torch.Tensor:
        if torch.jit.is_scripting():
            bias = self.bias.to(dtype=input.dtype) if self.bias is not None else None
            return F.linear(input, self.weight.to(dtype=input.dtype), bias=bias)
        else:
            return F.linear(input, self.weight, self.bias)
