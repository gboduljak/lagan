from typing import List, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange

from batch_index_select import *

# TODO: We are using different initialization of MLPs in comparison to the paper.
# It might be necessary to verify consequences of this.


class PatchSampler(nn.Module):
  def __init__(self,
               patch_embedding_dim: int,
               num_patches_per_layer: int,
               device: torch.device) -> None:
    super().__init__()
    self.mlps_init = False
    self.patch_embedding_dim = patch_embedding_dim
    self.num_patches_per_layer = num_patches_per_layer
    self.device = device

  def create_mlps_if_necessary(self, layer_outs: List[torch.Tensor]):
    if self.mlps_init:
      return

    for (mlp_id, layer_out) in enumerate(layer_outs):
      B, C, H, W = layer_out.shape
      setattr(
          self,
          f'mlp_{mlp_id}',
          nn.Sequential(
              nn.Linear(
                  in_features=C,
                  out_features=self.patch_embedding_dim
              ),
              nn.ReLU(),
              nn.Linear(
                  in_features=self.patch_embedding_dim,
                  out_features=self.patch_embedding_dim
              )
          ).to(self.device)
      )

    self.mlps_init = True

  def forward(self,
              layer_outs: List[torch.Tensor],
              patch_idx_per_layer: List[torch.Tensor] = [],
              apply_mlp: bool = True
              ) -> Tuple[List[torch.Tensor], List[torch.Tensor]]:

    self.create_mlps_if_necessary(layer_outs)

    sampled_patches = []
    sampled_patches_idx = []

    for layer_idx, layer_out in enumerate(layer_outs):
      B, C, H, W = layer_out.shape
      layer_spatial_size = H * W
      layer_patches = rearrange(layer_out, 'b c h w -> b (h w) c')

      if not patch_idx_per_layer:
        layer_patch_idx = torch.vstack(
            [
                torch.multinomial(
                    input=torch.ones(layer_spatial_size).to(self.device),
                    num_samples=min(layer_spatial_size,
                                    self.num_patches_per_layer)
                )
                for _ in range(B)
            ]
        )
      else:
        layer_patch_idx = patch_idx_per_layer[layer_idx]

      layer_sampled_patches = batch_index_select(
          x=layer_patches.to(self.device),
          idx=layer_patch_idx.to(self.device)
      )
      layer_mlp = getattr(self, f'mlp_{layer_idx}')
      layer_patch_embeddings = F.normalize(
          input=(
              layer_mlp(layer_sampled_patches)
              if apply_mlp
              else layer_sampled_patches
          ),
          dim=-1,
          p=2
      )
      sampled_patches.append(
          layer_patch_embeddings
      )
      sampled_patches_idx.append(layer_patch_idx)

    return sampled_patches, sampled_patches_idx
