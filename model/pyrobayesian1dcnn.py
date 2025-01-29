import torch.nn.functional as F
import pyro
import pyro.distributions as dist
from pyro.nn import PyroModule
import torch.nn as nn

from model.bayesian_conv_1d import BayesianConv1d
from model.bayesian_linear import BayesianLinear
from model.tabular_to_sequence import TabularToSequence

class PyroBayesian1DCNN(PyroModule):
    def __init__(self,
                 input_dim,
                 output_dim=1,
                 proj_channels=16,
                 proj_seq_len=4,
                 conv_out_channels=32,
                 kernel_size=3,
                 prior_std=0.1):
        super().__init__()
        
        # Projette du tabulaire vers séquence
        self.tab2seq = TabularToSequence(input_dim, proj_channels, proj_seq_len)
        self.bn1 = nn.BatchNorm1d(proj_channels)

        self.conv1 = BayesianConv1d(
            in_channels=proj_channels,
            out_channels=conv_out_channels,
            kernel_size=kernel_size,
            padding=kernel_size // 2,
            prior_std=prior_std
        )
        self.bn2 = nn.BatchNorm1d(conv_out_channels)

        flattened_size = conv_out_channels * proj_seq_len
        self.lin_out = BayesianLinear(
            in_features=flattened_size,
            out_features=output_dim,
            prior_std=prior_std
        )

    def forward(self, x):
        # x: (N, input_dim)
        x = self.tab2seq(x)    # => (N, proj_channels, seq_len)
        x = self.bn1(x)
        x = F.relu(x)

        x = self.conv1(x)      # => (N, conv_out_channels, seq_len)
        x = self.bn2(x)
        x = F.relu(x)

        x = x.flatten(start_dim=1)  # => (N, conv_out_channels * seq_len)
        x = self.lin_out(x)         # => (N, output_dim)
        return x

    def model(self, x, y=None):
        pyro.module("PyroBayesian1DCNN", self)
        logits = self.forward(x).squeeze(-1)
        with pyro.plate("data_plate", x.size(0)):
            pyro.sample("obs", dist.Bernoulli(logits=logits), obs=y)
        return logits

    def guide(self, x, y=None):
        pyro.module("PyroBayesian1DCNN", self)
        self.forward(x).squeeze(-1)
        with pyro.plate("data_plate", x.size(0)):
            pass
