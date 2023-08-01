

import torch.nn as nn

from cam import CAMAttention


class ConvBlock(nn.Sequential):
  def __init__(self,
               in_channels: int,
               out_channels: int,
               reflection_padding: int,
               kernel_size: int,
               stride: int,
               padding: int,
               bias: bool,
               act: nn.Module = nn.LeakyReLU(0.2, True)
               ):
    super().__init__(
        nn.ReflectionPad2d(reflection_padding),
        nn.utils.spectral_norm(
            nn.Conv2d(
                in_channels=in_channels,
                out_channels=out_channels,
                kernel_size=kernel_size,
                stride=stride,
                padding=padding,
                bias=bias
            )
        ),
        act
    )


class Discriminator(nn.Module):
  def __init__(self, in_channels, ndf=64, num_layers=5):
    super(Discriminator, self).__init__()
    # encoder
    enc = [
        ConvBlock(
            in_channels=in_channels,
            out_channels=ndf,
            reflection_padding=1,
            kernel_size=4,
            stride=2,
            padding=0,
            bias=True
        )
    ]
    for i in range(1, num_layers - 2):
      mult = 2 ** (i - 1)
      enc += [
          ConvBlock(
              in_channels=ndf * mult,
              out_channels=ndf * mult * 2,
              reflection_padding=1,
              kernel_size=4,
              stride=2,
              padding=0,
              bias=True
          )
      ]
    mult = 2 ** (num_layers - 2 - 1)
    enc += [
        ConvBlock(
            in_channels=ndf * mult,
            out_channels=ndf * mult * 2,
            reflection_padding=1,
            kernel_size=4,
            stride=1,
            padding=0,
            bias=True
        )
    ]
    self.enc = nn.Sequential(*enc)
    # attention
    mult = 2 ** (num_layers - 2)
    self.cam = CAMAttention(
        channels=ndf * mult,
        act=nn.LeakyReLU(0.2, True),
        spectral_norm=True
    )
    # head
    self.out = ConvBlock(
        in_channels=ndf * mult,
        reflection_padding=1,
        out_channels=1,
        kernel_size=4,
        stride=1,
        padding=0,
        bias=False,
        act=nn.Identity()
    )

  def forward(self, x):
    x = self.enc(x)
    x, cam_logit, heatmap = self.cam(x)
    x = self.out(x)
    return x, cam_logit, heatmap
