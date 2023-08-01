import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import init


def init_weights(net, init_type='normal', init_gain=0.02, debug=False):
  """Initialize network weights.

  Parameters:
      net (network)   -- network to be initialized
      init_type (str) -- the name of an initialization method: normal | xavier | kaiming | orthogonal
      init_gain (float)    -- scaling factor for normal, xavier and orthogonal.

  We use 'normal' in the original pix2pix and CycleGAN paper. But xavier and kaiming might
  work better for some applications. Feel free to try yourself.
  """
  def init_func(m):  # define the initialization function
    classname = m.__class__.__name__
    if hasattr(m, 'weight') and (classname.find('Conv') != -1 or classname.find('Linear') != -1):
      if debug:
        print(classname)
      if init_type == 'normal':
        init.normal_(m.weight.data, 0.0, init_gain)
      elif init_type == 'xavier':
        init.xavier_normal_(m.weight.data, gain=init_gain)
      elif init_type == 'kaiming':
        init.kaiming_normal_(m.weight.data, a=0, mode='fan_in')
      elif init_type == 'orthogonal':
        init.orthogonal_(m.weight.data, gain=init_gain)
      else:
        raise NotImplementedError(
            'initialization method [%s] is not implemented' % init_type)
      if hasattr(m, 'bias') and m.bias is not None:
        init.constant_(m.bias.data, 0.0)
    # BatchNorm Layer's weight is not a matrix; only normal distribution applies.
    elif classname.find('BatchNorm2d') != -1:
      init.normal_(m.weight.data, 1.0, init_gain)
      init.constant_(m.bias.data, 0.0)

  net.apply(init_func)  # apply the initialization function <init_func>


def init_net(net, init_type='normal', init_gain=0.02, device='cpu', debug=False, initialize_weights=True):
  """Initialize a network: 1. register CPU/GPU device (with multi-GPU support); 2. initialize the network weights
  Parameters:
      net (network)      -- the network to be initialized
      init_type (str)    -- the name of an initialization method: normal | xavier | kaiming | orthogonal
      gain (float)       -- scaling factor for normal, xavier and orthogonal.
      gpu_ids (int list) -- which GPUs the network runs on: e.g., 0,1,2

  Return an initialized network.
  """
  net = net.to(device)
  if initialize_weights:
    init_weights(net, init_type, init_gain=init_gain, debug=debug)
  return net


class Normalize(nn.Module):

  def __init__(self, power=2):
    super(Normalize, self).__init__()
    self.power = power

  def forward(self, x):
    norm = x.pow(self.power).sum(1, keepdim=True).pow(1. / self.power)
    out = x.div(norm + 1e-7)
    return out


class PatchSampleF(nn.Module):
  def __init__(self, use_mlp=False, init_type='normal', init_gain=0.02, nc=256, device: torch.device = 'cpu'):
    # potential issues: currently, we use the same patch_ids for multiple images in the batch
    super(PatchSampleF, self).__init__()
    self.l2norm = Normalize(2)
    self.use_mlp = use_mlp
    self.nc = nc
    self.mlp_init = False
    self.init_type = init_type
    self.init_gain = init_gain
    self.device = device

  def create_mlp(self, feats):
    for mlp_id, feat in enumerate(feats):
      input_nc = feat.shape[1]
      mlp = nn.Sequential(*[nn.Linear(input_nc, self.nc),
                          nn.ReLU(), nn.Linear(self.nc, self.nc)]).to(self.device)

      setattr(self, 'mlp_%d' % mlp_id, mlp)
    init_net(self, self.init_type, self.init_gain, self.device)
    self.mlp_init = True

  def forward(self, feats, num_patches=64, patch_ids=None, attn_mats=None):
    return_ids = []
    return_feats = []
    return_mats = []
    k_s = 7  # kernel size in unfold
    if self.use_mlp and not self.mlp_init:
      self.create_mlp(feats)
    for feat_id, feat in enumerate(feats):
      B, C, H, W = feat.shape[0], feat.shape[1], feat.shape[2], feat.shape[3]
      feat_reshape = feat.permute(0, 2, 3, 1).flatten(1, 2)  # B*HW*C
      if num_patches > 0:
        if feat_id < 3:
          if patch_ids is not None:
            patch_id = patch_ids[feat_id]
          else:
            patch_id = torch.randperm(
                feat_reshape.shape[1], device=feats[0].device)  # random id in [0, HW]
            # .to(patch_ids.device)
            patch_id = patch_id[:int(min(num_patches, patch_id.shape[0]))]
          x_sample = feat_reshape[:, patch_id, :].flatten(
              0, 1)  # reshape(-1, x.shape[1])
          attn_qs = torch.zeros(1).to(feat.device)
        else:
          feat_local = F.unfold(feat, kernel_size=k_s,
                                stride=1, padding=3)  # (B, ks*ks*C, L)
          L = feat_local.shape[2]
          if attn_mats is not None:
            patch_id = patch_ids[feat_id]
            attn_qs = attn_mats[feat_id]
          else:
            feat_k = feat_local.permute(0, 2, 1).reshape(
                B, L, k_s * k_s, C).flatten(0, 1)  # (B*L, ks*ks, C)
            feat_q = feat_reshape.reshape(B*L, C, 1)
            dots_local = torch.bmm(feat_k, feat_q)  # (B*L, ks*ks, 1)
            attn_local = dots_local.softmax(dim=1)
            attn_local = attn_local.reshape(B, L, -1)  # (B, L, ks*ks)
            prob = -torch.log(attn_local)
            prob = torch.where(torch.isinf(
                prob), torch.full_like(prob, 0), prob)
            entropy = torch.sum(torch.mul(attn_local, prob), dim=2)
            _, index = torch.sort(entropy)
            patch_id = index[:, :num_patches]
            attn_qs = attn_local[torch.arange(
                B)[:, None], patch_id, :]  # (B, n_p, ks*ks)
            attn_qs = attn_qs.flatten(0, 1).unsqueeze(1)  # (B*n_p, 1, ks*ks)
          feat_v = feat_local[torch.arange(B)[:, None], :, patch_id].permute(
              0, 2, 1)  # (B, n_p, ks*ks*C)
          feat_v = feat_v.flatten(0, 1).view(B*num_patches, k_s*k_s, C)
          feat_reshape = torch.bmm(attn_qs, feat_v)  # (B*n_p, 1, C)
          x_sample = feat_reshape.flatten(0, 1)
      else:
        x_sample = feat_reshape
        patch_id = []
      if self.use_mlp:
        mlp = getattr(self, 'mlp_%d' % feat_id)
        x_sample = mlp(x_sample)
      return_ids.append(patch_id)
      return_mats.append(attn_qs)
      x_sample = self.l2norm(x_sample)

      if num_patches == 0:
        x_sample = x_sample.permute(0, 2, 1).reshape(
            [B, x_sample.shape[-1], H, W])
      return_feats.append(x_sample)
    return return_feats, return_ids, return_mats
