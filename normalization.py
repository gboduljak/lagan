
import torch
import torch.nn as nn
from torch.nn.parameter import Parameter


class AdaILN(nn.Module):
  def __init__(self, num_features, eps=1e-5):
    super(AdaILN, self).__init__()
    self.eps = eps
    self.rho = Parameter(torch.Tensor(1, num_features, 1, 1))
    self.rho.data.fill_(0.9)

  def forward(self, x, gamma, beta):
    in_mean, in_var = (
        torch.mean(x, dim=[2, 3], keepdim=True),
        torch.var(x, dim=[2, 3], keepdim=True)
    )
    out_in = (x - in_mean) / torch.sqrt(in_var + self.eps)
    ln_mean, ln_var = (
        torch.mean(x, dim=[1, 2, 3], keepdim=True),
        torch.var(x, dim=[1, 2, 3], keepdim=True)
    )
    out_ln = (x - ln_mean) / torch.sqrt(ln_var + self.eps)
    out = (
        self.rho.expand(x.shape[0], -1, -1, -1) * out_in +
        (1-self.rho.expand(x.shape[0], -1, -1, -1)) * out_ln
    )
    out = out * gamma.unsqueeze(2).unsqueeze(3) + beta.unsqueeze(2).unsqueeze(3)
    return out


class ILN(nn.Module):
  def __init__(self, num_features, eps=1e-5):
    super(ILN, self).__init__()
    self.eps = eps
    self.rho = Parameter(torch.Tensor(1, num_features, 1, 1))
    self.gamma = Parameter(torch.Tensor(1, num_features, 1, 1))
    self.beta = Parameter(torch.Tensor(1, num_features, 1, 1))
    self.rho.data.fill_(0.0)
    self.gamma.data.fill_(1.0)
    self.beta.data.fill_(0.0)

  def forward(self, x):
    in_mean, in_var = (
        torch.mean(x, dim=[2, 3], keepdim=True),
        torch.var(x, dim=[2, 3], keepdim=True)
    )
    out_in = (x - in_mean) / torch.sqrt(in_var + self.eps)
    ln_mean, ln_var = (
        torch.mean(x, dim=[1, 2, 3], keepdim=True),
        torch.var(x, dim=[1, 2, 3], keepdim=True)
    )
    out_ln = (x - ln_mean) / torch.sqrt(ln_var + self.eps)
    out = (
        self.rho.expand(x.shape[0], -1, -1, -1) * out_in +
        (1-self.rho.expand(x.shape[0], -1, -1, -1)) * out_ln
    )
    out = (
        out * self.gamma.expand(x.shape[0], -1, -1, -1) +
        self.beta.expand(x.shape[0], -1, -1, -1)
    )
    return out


class RhoClipper(object):
  def __init__(self, min, max):
    self.clip_min = min
    self.clip_max = max
    assert min < max

  def __call__(self, module):
    if hasattr(module, 'rho'):
      w = module.rho.data
      w = w.clamp(self.clip_min, self.clip_max)
      module.rho.data = w
