import torch
import torch.nn as nn
import torch.nn.functional as F
from loss import gaussian_loss, KL_gaussians
import numpy as np
import math


class Conv(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, dilation=1, causal=False, mode='SAME'):
        super(Conv, self).__init__()

        self.causal = causal
        self.mode = mode
        if self.causal and self.mode == 'SAME':
            self.padding = dilation * (kernel_size - 1)
        elif self.mode == 'SAME':
            self.padding = dilation * (kernel_size - 1) // 2
        else:
            self.padding = 0
        self.conv = nn.Conv1d(in_channels, out_channels, kernel_size, dilation=dilation, padding=self.padding)
        self.conv = nn.utils.weight_norm(self.conv)
        nn.init.kaiming_normal_(self.conv.weight)

    def forward(self, tensor):
        out = self.conv(tensor)
        if self.causal and self.padding is not 0:
            out = out[:, :, :-self.padding]
        return out

    def remove_weight_norm(self):
        nn.utils.remove_weight_norm(self.conv)


class ResBlock(nn.Module):
    def __init__(self, in_channels, out_channels, skip_channels, kernel_size, dilation,
                 cin_channels=None, local_conditioning=True, causal=False, mode='SAME'):
        super(ResBlock, self).__init__()
        self.causal = causal
        self.local_conditioning = local_conditioning
        self.cin_channels = cin_channels
        self.mode = mode

        self.filter_conv = Conv(in_channels, out_channels, kernel_size, dilation, causal, mode)
        self.gate_conv = Conv(in_channels, out_channels, kernel_size, dilation, causal, mode)
        self.res_conv = nn.Conv1d(out_channels, in_channels, kernel_size=1)
        self.skip_conv = nn.Conv1d(out_channels, skip_channels, kernel_size=1)
        self.res_conv = nn.utils.weight_norm(self.res_conv)
        self.skip_conv = nn.utils.weight_norm(self.skip_conv)
        nn.init.kaiming_normal_(self.res_conv.weight)
        nn.init.kaiming_normal_(self.skip_conv.weight)

        if self.local_conditioning:
            self.filter_conv_c = nn.Conv1d(cin_channels, out_channels, kernel_size=1)
            self.gate_conv_c = nn.Conv1d(cin_channels, out_channels, kernel_size=1)
            self.filter_conv_c = nn.utils.weight_norm(self.filter_conv_c)
            self.gate_conv_c = nn.utils.weight_norm(self.gate_conv_c)
            nn.init.kaiming_normal_(self.filter_conv_c.weight)
            nn.init.kaiming_normal_(self.gate_conv_c.weight)

    def forward(self, tensor, c=None):
        h_filter = self.filter_conv(tensor)
        h_gate = self.gate_conv(tensor)

        if self.local_conditioning:
            h_filter += self.filter_conv_c(c)
            h_gate += self.gate_conv_c(c)

        out = F.tanh(h_filter) * F.sigmoid(h_gate)

        res = self.res_conv(out)
        skip = self.skip_conv(out)
        if self.mode == 'SAME':
            return (tensor + res) * math.sqrt(0.5), skip
        else:
            return (tensor[:, :, 1:] + res) * math.sqrt(0.5), skip

    def remove_weight_norm(self):
        self.filter_conv.remove_weight_norm()
        self.gate_conv.remove_weight_norm()
        nn.utils.remove_weight_norm(self.res_conv)
        nn.utils.remove_weight_norm(self.skip_conv)
        nn.utils.remove_weight_norm(self.filter_conv_c)
        nn.utils.remove_weight_norm(self.gate_conv_c)


class GaussianLoss(nn.Module):
    def __init__(self):
        super(GaussianLoss, self).__init__()

    def forward(self, input, target, size_average=True):
        losses = gaussian_loss(input, target)
        if size_average:
            return losses.mean()
        else:
            return losses.mean(1).sum(0)


class KL_Loss(nn.Module):
    def __init__(self):
        super(KL_Loss, self).__init__()

    def forward(self, mu_q, logs_q, mu_p, logs_p, regularization=True, size_average=True):
        KL_loss, reg_loss = KL_gaussians(mu_q, logs_q, mu_p, logs_p, regularization=regularization)
        loss_tot = KL_loss + reg_loss * 4.

        if size_average:
            return loss_tot.mean(), KL_loss.mean(), reg_loss.mean()
        else:
            return loss_tot.sum(), KL_loss.sum(), reg_loss.sum()


class ExponentialMovingAverage(object):
    def __init__(self, decay):
        self.decay = decay
        self.shadow = {}

    def register(self, name, val):
        self.shadow[name] = val.clone()

    def update(self, name, x):
        assert name in self.shadow
        new_average = self.decay * x + (1.0 - self.decay) * self.shadow[name]
        self.shadow[name] = new_average.clone()


def stft(y, scale='linear'):
    D = torch.stft(y, n_fft=1024, hop_length=256, win_length=1024)#, window=torch.hann_window(1024).cuda())
    D = torch.sqrt(D.pow(2).sum(-1) + 1e-10)
    # D = torch.sqrt(torch.clamp(D.pow(2).sum(-1), min=1e-10))
    if scale == 'linear':
        return D
    elif scale == 'log':
        S = 2 * torch.log(torch.clamp(D, 1e-10, float("inf")))
        return S
    else:
        pass

