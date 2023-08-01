import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import einsum, rearrange, repeat


class PatchNCELoss(nn.Module):
  def __init__(self,
               temperature: float,
               use_external_patches: bool = False,
               detach_keys: bool = True):
    super().__init__()
    self.temperature = temperature
    self.use_external_patches = use_external_patches
    self.detach_keys = detach_keys

  def forward(self, q: torch.Tensor, k: torch.Tensor):
    # q : [batch_size, num_patches_per_batch, patch_dim]
    # k : [batch_size, num_patches_per_batch, patch_dim]
    batch_size, num_patches_per_batch, _ = q.shape
    if self.detach_keys:
      """
      The gradient of PNCE applies on the anchor q to train the parameters in the generator, while it
      is detached on k+ and k-, so that the generator is guided for the single
      direction of domain translation.
      """
      k = k.detach()
    # logits: [batch_size, num_patches_per_batch, num_patches_per_batch]
    #       - diagonal entries are positives
    #       - non-diagonal entries are negatives
    logits = (1 / self.temperature) * einsum(
        q,
        k,
        'b i d, b j d -> b i j'
    )
    labels = torch.arange(num_patches_per_batch).to(logits.device).long()

    if self.use_external_patches:
      external_logits = (1 / self.temperature) * einsum(
          q,
          k,
          'k i d, l j d -> k l i j'
      )
      mask_internal_logits = repeat(
          torch.eye(batch_size),
          'k l -> k l i j',
          i=num_patches_per_batch,
          j=num_patches_per_batch
      )
      masked_external_negatives_logits = torch.masked_fill(
          input=external_logits,
          mask=mask_internal_logits.bool(),
          value=float('-inf')
      )
      external_negatives_logits = rearrange(
          masked_external_negatives_logits,
          'b l i j -> b i (l j)'
      )
      # logits : [batch_size, num_patches_per_batch, num_patches_per_batch]
      # external_negatives_logits:  [batch_size, num_patches_per_batch, (batch_size * num_patches_per_batch)]
      logits = torch.cat((logits, external_negatives_logits), dim=2)
      # logits : [batch_size, num_patches_per_batch, num_patches_per_batch + (batch_size * num_patches_per_batch)]
    # print(logits) for debugging.
    # out : [batch_size * num_patches_per_batch, ]
    return F.cross_entropy(
        input=rearrange(
            logits,
            'b i j -> (b i) j'
        ),
        target=repeat(
            labels,
            'i -> (b i)',
            b=batch_size
        ),
        reduction='none'
    )
