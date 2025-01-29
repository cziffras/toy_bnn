import pyro
import pyro.distributions as dist
from pyro.nn import PyroModule, PyroSample
import torch.nn.functional as F

class BayesianLinear(PyroModule):
    def __init__(self, in_features, out_features, prior_std=0.1):
        super().__init__()
        self.weight = PyroSample(
            dist.Normal(0., prior_std)
            .expand([out_features, in_features])
            .to_event(2)
        )
        self.bias = PyroSample(
            dist.Normal(0., prior_std)
            .expand([out_features])
            .to_event(1)
        )

    def forward(self, x):
        return F.linear(x, self.weight, self.bias)
