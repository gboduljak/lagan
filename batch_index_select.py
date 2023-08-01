import torch
from einops import repeat


def batch_index_select(x: torch.Tensor, idx: torch.Tensor):
  """
  Applies (batched) index_select on x according to indices idx.
  Arguments:
      x:   Tensor[N, S, D]
      idx: Tensor[N, K]
  Returns:
      Tensor[N, K, D] 
  """
  N_x, _, D = x.shape
  N_idx, _ = idx.shape
  assert N_x == N_idx
  return torch.gather(
      input=x,
      dim=1,
      index=repeat(
          idx,
          'N K -> N K D',
          D=D
      )
  )
