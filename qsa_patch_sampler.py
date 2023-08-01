from typing import List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import einsum, rearrange
from strenum import StrEnum

from batch_index_select import *

# TODO: We are using different initialization of MLPs in comparison to the paper.
# It might be necessary to verify consequences of this.


class QSAType(StrEnum):
  GLOBAL = 'global'
  LOCAL = 'local'
  GLOBAL_AND_LOCAL = 'global+local'


class QSAPatchSampler(nn.Module):
  def __init__(self,
               patch_embedding_dim: int,
               num_patches_per_layer: int,
               qsa_type: QSAType,
               max_spatial_size: int,
               device: torch.device) -> None:
    super().__init__()
    self.mlps_init = False
    self.patch_embedding_dim = patch_embedding_dim
    self.num_patches_per_layer = num_patches_per_layer
    self.qsa_type = qsa_type
    self.max_spatial_size = max_spatial_size
    self.attn_layers = []
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
      if (H * W <= self.max_spatial_size):
        self.attn_layers.append(mlp_id)

    self.mlps_init = True

  def sample_local(self, layer_out: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    B, C, H, W = layer_out.shape
    K_S = 7  # as in the paper
    layer_patches_local = F.unfold(
        layer_out,
        kernel_size=K_S,
        stride=1,
        padding=3
    )
    k = rearrange(
        layer_patches_local,
        'b (i j c) l -> (b l) (i j) c',
        i=K_S,
        j=K_S,
        c=C
    )
    W_S = layer_patches_local.shape[-1]  # window size
    q = rearrange(
        layer_out,
        'b c h w -> b (h w) c'
    ).reshape((B*W_S, C, 1))
    dots = einsum(k, q, 'b k c, b c l -> b k l')
    attn = F.softmax(dots, dim=1).reshape((B, W_S, -1))
    prob = -torch.log(attn)
    prob = torch.where(
        torch.isinf(prob),
        torch.full_like(prob, 0),
        prob
    )
    ent = torch.sum(torch.mul(attn, prob), dim=2)
    _, ent_idx = torch.sort(ent)
    layer_attn_map_idx = ent_idx[:, :self.num_patches_per_layer]
    layer_attn_map = batch_index_select(
        x=attn,
        idx=layer_attn_map_idx
    )
    v = layer_patches_local[
        torch.arange(B)[:, None],
        :,
        layer_attn_map_idx
    ]
    v = rearrange(v, 'b n l -> (b l) n').reshape(
        (B, self.num_patches_per_layer, K_S * K_S, C)
    )
    layer_sampled_patches = einsum(
        layer_attn_map,
        v,
        'b n l, b n l c -> b n c'
    )
    return layer_sampled_patches, layer_attn_map, layer_attn_map_idx

  def sample_global(self, layer_out: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    # Returns global attention map
    q = rearrange(layer_out, 'b c h w -> b (h w) c')
    k = rearrange(layer_out, 'b c h w -> b (h w) c')
    v = rearrange(layer_out, 'b c h w -> b (h w) c')

    dots = einsum(q, k, 'b i c, b j c -> b i j')
    attn = torch.softmax(dots, dim=2)
    prob = -torch.log(attn)
    prob = torch.where(
        torch.isinf(prob),
        torch.full_like(prob, 0).to(self.device),
        prob
    )
    ent = torch.sum(torch.mul(attn, prob), dim=2)
    _, ent_idx = torch.sort(ent)
    layer_attn_map_idx = ent_idx[:, :self.num_patches_per_layer]
    layer_attn_map = batch_index_select(
        x=attn,
        idx=layer_attn_map_idx
    )
    layer_sampled_patches = einsum(
        layer_attn_map,
        v,
        'b i j, b j c -> b i c'
    )
    return layer_sampled_patches, layer_attn_map, layer_attn_map_idx

  def sample_global_and_local(self, layer_out: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    B, C, H, W = layer_out.shape
    K_S = 7  # as in the paper
    layer_patches_local = F.unfold(
        layer_out,
        kernel_size=7,
        stride=1,
        padding=3
    )  # (B, ks*ks*C, L)
    k_local = rearrange(
        layer_patches_local,
        'b (i j c) l -> (b l) (i j) c',
        i=K_S,
        j=K_S,
        c=C
    )
    W_S = layer_patches_local.shape[-1]  # window size
    q_local = rearrange(
        layer_out,
        'b c h w -> b (h w) c'
    ).reshape((B*W_S, C, 1))
    dots_local = einsum(k_local, q_local, 'b k c, b c l -> b k l')
    attn_local = F.softmax(dots_local, dim=1).reshape((B, W_S, -1))
    prob = -torch.log(attn_local)
    prob = torch.where(
        torch.isinf(prob),
        torch.full_like(prob, 0).to(self.device),
        prob
    )
    ent = torch.sum(torch.mul(attn_local, prob), dim=2)
    _, ent_idx = torch.sort(ent)
    local_attn_idx = ent_idx[:, :self.num_patches_per_layer]

    q_global = rearrange(layer_out, 'b c h w -> b (h w) c')
    k_global = rearrange(layer_out, 'b c h w -> b (h w) c')
    v_global = rearrange(layer_out, 'b c h w -> b (h w) c')

    dots_global = einsum(q_global, k_global, 'b i c, b j c -> b i j')
    attn_global = F.softmax(dots_global, dim=2)  # softmax along rows
    layer_attn_map = batch_index_select(
        x=attn_global,
        idx=local_attn_idx
    )
    layer_sampled_patches = einsum(
        layer_attn_map,
        v_global,
        'b i j, b j c -> b i c'
    )
    return layer_sampled_patches, layer_attn_map, local_attn_idx

  def forward(self,
              layer_outs: List[torch.Tensor],
              patch_idx_per_layer: List[Optional[torch.Tensor]] = [],
              attn_map_per_layer: List[Optional[torch.Tensor]] = [],
              apply_mlp: bool = True,
              return_only_full_attn_maps: bool = False,
              ) -> Tuple[List[torch.Tensor], List[torch.Tensor]]:
    # rules: Use patch_idx_per_layer if we have it. Use layer_attn_map if we have it. Otherwise, sample.
    self.create_mlps_if_necessary(layer_outs)

    sampled_patches = []
    sampled_patches_idx = []
    sampled_patches_layer_attn_maps = []

    if return_only_full_attn_maps:
      full_attn_maps = []

    for layer_idx, layer_out in enumerate(layer_outs):
      B, C, H, W = layer_out.shape
      layer_spatial_size = H * W

      if layer_spatial_size <= self.max_spatial_size:
        if not attn_map_per_layer:
          samplers = {
              QSAType.LOCAL: self.sample_local,
              QSAType.GLOBAL: self.sample_global,
              QSAType.GLOBAL_AND_LOCAL: self.sample_global_and_local
          }
          (layer_sampled_patches,
           layer_attn_map,
           layer_attn_idx) = samplers[self.qsa_type](layer_out)
          if return_only_full_attn_maps:
            full_attn_maps.append(layer_attn_map)
        else:
          layer_attn_map = attn_map_per_layer[layer_idx]

          if self.qsa_type == QSAType.LOCAL:
            K_S = 7
            layer_attn_idx = patch_idx_per_layer[layer_idx]
            layer_patches_local = F.unfold(
                layer_out,
                kernel_size=7,
                stride=1,
                padding=3
            )
            v = layer_patches_local[
                torch.arange(B)[:, None],
                :,
                layer_attn_idx
            ]
            v = rearrange(v, 'b n l -> (b l) n').reshape(
                (B, self.num_patches_per_layer, K_S * K_S, C)
            )
            layer_sampled_patches = einsum(
                layer_attn_map,
                v,
                'b n l, b n l c -> b n c'
            )
          else:
            v = rearrange(layer_out, 'b c h w -> b (h w) c')
            layer_sampled_patches = einsum(
                layer_attn_map,
                v,
                'b i j, b j c -> b i c'
            )
        assert (layer_sampled_patches != None)
        assert (layer_attn_map != None)

        sampled_patches_layer_attn_maps.append(layer_attn_map)
        sampled_patches_idx.append(
            layer_attn_idx if self.qsa_type == QSAType.LOCAL else None
        )

      else:
        if not patch_idx_per_layer:
          # sample random
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
          # no need to sample, patch idx known
          layer_patch_idx = patch_idx_per_layer[layer_idx]

        layer_patches = rearrange(layer_out, 'b c h w -> b (h w) c')
        assert (
            layer_patch_idx != None and
            type(layer_patch_idx) == torch.Tensor
        )
        layer_sampled_patches = batch_index_select(
            x=layer_patches.to(self.device),
            idx=layer_patch_idx.to(self.device)
        )
        sampled_patches_idx.append(layer_patch_idx)
        sampled_patches_layer_attn_maps.append(None)

      assert (
          layer_sampled_patches != None and
          type(layer_sampled_patches) == torch.Tensor
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

    if return_only_full_attn_maps:
      return full_attn_maps

    return sampled_patches, sampled_patches_idx, sampled_patches_layer_attn_maps
