
from typing import Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


class CAMAttention(nn.Module):
  def __init__(self, channels: int, act: nn.Module = nn.ReLU(True), spectral_norm: bool = False):
    super(CAMAttention, self).__init__()

    if spectral_norm:
      self.gap_fc = nn.utils.spectral_norm(
          nn.Linear(in_features=channels, out_features=1, bias=False)
      )
      self.gmp_fc = nn.utils.spectral_norm(
          nn.Linear(in_features=channels, out_features=1, bias=False)
      )
    else:
      self.gap_fc = nn.Linear(in_features=channels, out_features=1, bias=False)
      self.gmp_fc = nn.Linear(in_features=channels, out_features=1, bias=False)

    # In the original paper's implementation, this layer is not spectrally normalized.
    self.conv1x1 = nn.Conv2d(
        in_channels=2 * channels,
        out_channels=channels,
        kernel_size=1,
        stride=1,
        bias=True
    )
    self.act = act

  def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    gap = F.adaptive_avg_pool2d(x, 1)
    gap_logit = self.gap_fc(gap.view(x.shape[0], -1))
    gap_weight = list(self.gap_fc.parameters())[0]
    gap = x * gap_weight.unsqueeze(2).unsqueeze(3)

    gmp = F.adaptive_max_pool2d(x, 1)
    gmp_logit = self.gmp_fc(gmp.view(x.shape[0], -1))
    gmp_weight = list(self.gmp_fc.parameters())[0]
    gmp = x * gmp_weight.unsqueeze(2).unsqueeze(3)

    cam_logit = torch.cat([gap_logit, gmp_logit], 1)
    x = torch.cat([gap, gmp], 1)
    x = self.act(self.conv1x1(x))

    heatmap = torch.sum(x, dim=1, keepdim=True)
    return x, cam_logit, heatmap
