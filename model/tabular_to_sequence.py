import torch
import torch.nn as nn

class TabularToSequence(nn.Module):
    def __init__(self, input_dim, out_channels, seq_len):
        super().__init__()
        self.out_channels = out_channels
        self.seq_len = seq_len
        self.linear = nn.Linear(input_dim, out_channels * seq_len)

    def forward(self, x):
        # x: (N, input_dim)
        out = self.linear(x)  # => (N, out_channels * seq_len)
        out = out.view(-1, self.out_channels, self.seq_len)
        return out
