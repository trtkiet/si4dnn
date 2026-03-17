from .operations import Linear, ReLU, LeakyReLU, BatchNorm1d
from . import util
import torch
import numpy as np
import time


class CUDAModel:
    def __init__(self, model):
        self.layers = util.parse_model(model)

    def forward(self, a, b, z):
        a = torch.tensor(a, dtype=torch.float32, device="cuda")  # GPU tensor
        b = torch.tensor(b, dtype=torch.float32, device="cuda")  # GPU tensor
        itv = torch.tensor(
            [-float("inf"), float("inf")], dtype=torch.float32, device="cuda"
        )  # GPU interval
        z_gpu = torch.tensor(z, dtype=torch.float32, device="cuda")  # GPU scalar
        for name, params in self.layers:
            if name == "Linear":
                a, b = Linear(a, b, params)
            elif name == "ReLU":
                a, b, itv = ReLU(a, b, z_gpu, itv)
            elif name == "LeakyReLU":
                # Extract alpha from params if available, otherwise use default
                alpha = params if params is not None else 0.01
                a, b, itv = LeakyReLU(a, b, z_gpu, itv, alpha)
            elif name == "BatchNorm1d":
                a, b = BatchNorm1d(a, b, params)
        a = a.cpu().numpy()
        b = b.cpu().numpy()
        itv = [itv[0].cpu().item(), itv[1].cpu().item()]
        return a, b, itv
