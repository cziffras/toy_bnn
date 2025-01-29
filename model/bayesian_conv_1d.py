import torch.nn.functional as F
import pyro.distributions as dist
from pyro.nn import PyroModule, PyroSample

class BayesianConv1d(PyroModule):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=0, prior_std=0.1):
        super().__init__()
        self.weight = PyroSample(
            dist.Normal(0., prior_std)
            .expand([out_channels, in_channels, kernel_size])
            .to_event(3)
        )
        self.bias = PyroSample(
            dist.Normal(0., prior_std)
            .expand([out_channels])
            .to_event(1)
        )
        self.stride = stride
        self.padding = padding

    def forward(self, x):
        return F.conv1d(x, self.weight, self.bias, stride=self.stride, padding=self.padding)
