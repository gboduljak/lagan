from typing import List, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F

from cam import CAMAttention
from normalization import ILN
from resnet import ResnetBlock


class FromRGB(nn.Sequential):
  def __init__(self, ngf: int, in_channels: int = 3):
    super().__init__(
        nn.ReflectionPad2d(3),
        nn.Conv2d(
            in_channels=in_channels,
            out_channels=ngf,
            kernel_size=7,
            stride=1,
            padding=0,
            bias=False
        ),
        nn.InstanceNorm2d(ngf),
        nn.ReLU(True)
    )


class ToRGB(nn.Sequential):
  def __init__(self, in_channels: int, out_channels: int = 3):
    super().__init__(
        nn.ReflectionPad2d(3),
        nn.Conv2d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=7,
            stride=1,
            padding=0,
            bias=False
        ),
        nn.Tanh()
    )


class DownsampleBlock(nn.Sequential):
  def __init__(self,
               in_channels: int,
               out_channels: int,
               kernel_size: int,
               stride: int,
               reflection_padding: int,
               padding: int,
               bias: bool
               ):
    super().__init__(
        nn.ReflectionPad2d(reflection_padding),
        nn.Conv2d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            bias=bias
        ),
        nn.InstanceNorm2d(out_channels),
        nn.ReLU(True)
    )


class UpsampleBlock(nn.Sequential):
  def __init__(self,
               in_channels: int,
               out_channels: int,
               scale_factor: int,
               kernel_size: int,
               stride: int,
               reflection_padding: int,
               padding: int,
               bias: bool,
               activation: nn.Module
               ):
    super().__init__(
        nn.Upsample(scale_factor=scale_factor, mode='nearest'),
        nn.ReflectionPad2d(reflection_padding),
        nn.Conv2d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            bias=bias
        ),
        ILN(out_channels),
        activation
    )


class AdaNormParamsInferenceNet(nn.Module):
  def __init__(self, ngf: int, mult: int):
    super(AdaNormParamsInferenceNet, self).__init__()

    self.fc = nn.Sequential(
        nn.Linear(ngf * mult, ngf * mult, bias=False),
        nn.ReLU(True),
        nn.Linear(ngf * mult, ngf * mult, bias=False),
        nn.ReLU(True)
    )
    self.gamma = nn.Linear(ngf * mult, ngf * mult, bias=False)
    self.beta = nn.Linear(ngf * mult, ngf * mult, bias=False)

  def forward(self, x: torch.Tensor):
    batch_size, _, _, _ = x.shape
    x_ = F.adaptive_avg_pool2d(x, 1)
    x_ = x_.view(batch_size, -1)
    x_ = self.fc(x_)
    return self.gamma(x_), self.beta(x_)


class ResnetGenerator(nn.Module):
  def __init__(self,
               in_channels: int = 3,
               out_channels: int = 3,
               ngf=64,
               num_bottleneck_blocks=9,
               num_downsampling_blocks=2,
               nce_layers_indices: List[int] = [],
               ):
    super(ResnetGenerator, self).__init__()

    self.in_channels = in_channels
    self.output_nc = out_channels
    self.ngf = ngf
    self.num_resnet_blocks = num_bottleneck_blocks
    self.num_resnet_enc_blocks = num_bottleneck_blocks // 2
    self.num_resnet_dec_blocks = num_bottleneck_blocks // 2

    if num_bottleneck_blocks % 2:
      self.num_resnet_enc_blocks += 1

    self.nce_layers_indices = nce_layers_indices

    # encoder downsampling
    self.enc_down = nn.ModuleList([
        nn.Identity(),  # For easier NCE indexing.
        FromRGB(in_channels=in_channels, ngf=ngf),
    ])
    for i in range(num_downsampling_blocks):
      mult = 2**i
      self.enc_down += [
          DownsampleBlock(
              reflection_padding=1,
              in_channels=ngf * mult,
              out_channels=ngf * mult * 2,
              kernel_size=3,
              stride=2,
              padding=0,
              bias=False
          )
      ]
    # encoder bottleneck
    mult = 2**num_downsampling_blocks
    self.enc_bottleneck = nn.ModuleList([
        ResnetBlock(
            channels=ngf * mult,
            bias=False,
            adaptive_norm=False
        )
        for _ in range(self.num_resnet_enc_blocks)
    ])
    # CAM
    self.cam = CAMAttention(channels=ngf * mult, act=nn.ReLU(True))
    # AdaLIN params
    self.ada_norm_params_infer = AdaNormParamsInferenceNet(ngf, mult)
    # decoder bottleneck
    self.dec_bottleneck = nn.ModuleList([
        ResnetBlock(
            channels=ngf * mult,
            bias=False,
            adaptive_norm=True
        )
        for _ in range(self.num_resnet_dec_blocks)
    ])
    # decoder upsampling
    self.dec_up = nn.ModuleList([])
    for i in range(num_downsampling_blocks):
      mult = 2**(num_downsampling_blocks - i)
      self.dec_up += [
          UpsampleBlock(
              reflection_padding=1,
              in_channels=ngf * mult,
              out_channels=int(ngf * mult / 2),
              scale_factor=2,
              kernel_size=3,
              stride=1,
              padding=0,
              bias=False,
              activation=nn.ReLU(True)
          )
      ]
    self.dec_up += [ToRGB(in_channels=ngf, out_channels=out_channels)]
    # layer index
    self.layers = dict(
        enumerate(
            self.enc_down +
            self.enc_bottleneck +
            [self.cam] +
            [self.ada_norm_params_infer] +
            self.dec_bottleneck +
            self.dec_up
        )
    )

  def encode(self, x: torch.Tensor):
    assert self.nce_layers_indices
    nce_layers = [
        self.layers[layer_idx]
        for layer_idx in self.nce_layers_indices
    ]
    final_nce_layer = nce_layers[-1] if nce_layers else None
    nce_layers_outs = []

    for layer in (self.enc_down + self.enc_bottleneck):
      x = layer(x)
      if layer in nce_layers:
        nce_layers_outs.append(x)
      if layer == final_nce_layer:
        return nce_layers_outs
    for out in nce_layers_outs:
      print(out.shape)
    raise ValueError(
        'final nce layer must be within the encoder of the generator!'
    )

  def forward(self, x: torch.Tensor, cam: bool = True) -> Union[Tuple[torch.Tensor, torch.Tensor, torch.Tensor], torch.Tensor]:
    for layer in self.enc_down:
      x = layer(x)
    for layer in self.enc_bottleneck:
      x = layer(x)
    x, cam_logits, heatmap = self.cam(x)
    gamma, beta = self.ada_norm_params_infer(x)
    for layer in self.dec_bottleneck:
      x = layer(x, gamma, beta)
    for layer in self.dec_up:
      x = layer(x)
    if cam:
      return x, cam_logits, heatmap
    else:
      return x
