import torch
import torch.nn as nn
import torch.nn.functional as F
from modules import Conv, ResBlock


class Wavenet_Student(nn.Module):
    def __init__(self, num_blocks_student=[1, 1, 1, 1, 1, 1], num_layers=10,
                 front_channels=32, residual_channels=64, gate_channels=128, skip_channels=64,
                 kernel_size=3, cin_channels=80, causal=True):
        super(Wavenet_Student, self).__init__()
        self.num_blocks = num_blocks_student
        self.num_flow = len(self.num_blocks)
        self.num_layers = num_layers

        self.iafs = nn.ModuleList()
        for i in range(self.num_flow):
            self.iafs.append(Wavenet_Flow(out_channels=2,
                                          num_blocks=self.num_blocks[i], num_layers=self.num_layers,
                                          front_channels=front_channels, residual_channels=residual_channels,
                                          gate_channels=gate_channels, skip_channels=skip_channels,
                                          kernel_size=kernel_size, cin_channels=cin_channels, causal=causal))

    def forward(self, z, c):
        return self.iaf(z, c)

    def iaf(self, z, c_up):
        mu_tot, logs_tot = 0., 0.
        for i, iaf in enumerate(self.iafs):
            mu_logs = iaf(z, c_up)
            mu = mu_logs[:, 0:1, :-1]
            logs = mu_logs[:, 1:, :-1]
            mu_tot = mu_tot * torch.exp(logs) + mu
            logs_tot = logs_tot + logs
            z = z[:, :, 1:] * torch.exp(logs) + mu
            z = F.pad(z, pad=(1, 0), mode='constant', value=0)
        return z, mu_tot, logs_tot

    def receptive_field(self):
        receptive_field = 1
        for iaf in self.iafs:
            receptive_field += iaf.receptive_field_size() - 1
        return receptive_field

    def generate(self, z, c_up):
        x, _, _ = self.iaf(z, c_up)
        return x

    def remove_weight_norm(self):
        for iaf in self.iafs:
            iaf.remove_weight_norm()


class Wavenet_Flow(nn.Module):
    def __init__(self, out_channels=1, num_blocks=1, num_layers=10,
                 front_channels=32, residual_channels=64, gate_channels=32, skip_channels=None,
                 kernel_size=3, cin_channels=80, causal=True):
        super(Wavenet_Flow, self). __init__()

        self.causal = causal
        self.num_blocks = num_blocks
        self.num_layers = num_layers
        self.front_channels = front_channels
        self.out_channels = out_channels
        self.gate_channels = gate_channels
        self.residual_channels = residual_channels
        self.skip_channels = skip_channels
        self.cin_channels = cin_channels
        self.kernel_size = kernel_size

        self.front_conv = nn.Sequential(
            Conv(1, self.residual_channels, self.front_channels, causal=self.causal),
            nn.ReLU()
        )
        self.res_blocks = nn.ModuleList()
        self.res_blocks_fast = nn.ModuleList()
        for b in range(self.num_blocks):
            for n in range(self.num_layers):
                self.res_blocks.append(ResBlock(self.residual_channels, self.gate_channels, self.skip_channels,
                                                self.kernel_size, dilation=2**n,
                                                cin_channels=self.cin_channels, local_conditioning=True,
                                                causal=self.causal, mode='SAME'))
        self.final_conv = nn.Sequential(
            nn.ReLU(),
            Conv(self.skip_channels, self.skip_channels, 1, causal=self.causal),
            nn.ReLU(),
            Conv(self.skip_channels, self.out_channels, 1, causal=self.causal)
        )

    def forward(self, x, c):
        return self.wavenet(x, c)

    def wavenet(self, tensor, c=None):
        h = self.front_conv(tensor)
        skip = 0
        for i, f in enumerate(self.res_blocks):
            h, s = f(h, c)
            skip += s
        out = self.final_conv(skip)
        return out

    def receptive_field_size(self):
        num_dir = 1 if self.causal else 2
        dilations = [2 ** (i % self.num_layers) for i in range(self.num_layers * self.num_blocks)]
        return num_dir * (self.kernel_size - 1) * sum(dilations) + 1 + (self.front_channels - 1)

    def remove_weight_norm(self):
        for f in self.res_blocks:
            f.remove_weight_norm()
