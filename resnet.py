import torch.nn as nn

from normalization import AdaILN


class ResnetBlock(nn.Module):
  def __init__(self, channels: int, bias: bool, adaptive_norm: bool):
    super(ResnetBlock, self).__init__()
    self.adaptive_norm = adaptive_norm
    self.pad1 = nn.ReflectionPad2d(1)
    self.conv1 = nn.Conv2d(
        in_channels=channels,
        out_channels=channels,
        kernel_size=3,
        stride=1,
        padding=0,
        bias=bias
    )
    self.norm1 = (
        AdaILN(channels) if adaptive_norm else
        nn.InstanceNorm2d(channels)
    )
    self.relu1 = nn.ReLU(True)
    self.pad2 = nn.ReflectionPad2d(1)
    self.conv2 = nn.Conv2d(
        in_channels=channels,
        out_channels=channels,
        kernel_size=3,
        stride=1,
        padding=0,
        bias=bias
    )
    self.norm2 = (
        AdaILN(channels) if adaptive_norm else
        nn.InstanceNorm2d(channels)
    )

  def forward(self, x, gamma=None, beta=None):
    out = self.pad1(x)
    out = self.conv1(out)
    if self.adaptive_norm:
      out = self.norm1(out, gamma, beta)
    else:
      out = self.norm1(out)
    out = self.relu1(out)
    out = self.pad2(out)
    out = self.conv2(out)
    if self.adaptive_norm:
      out = self.norm2(out, gamma, beta)
    else:
      out = self.norm2(out)
    return out + x
