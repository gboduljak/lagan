import itertools
import os
import time
from glob import glob
from pathlib import Path
from typing import Tuple

import numpy as np
import torch
import torch_fidelity
from torch.utils.data import DataLoader
from torchvision import transforms

from dataset import ImageFolder
from discriminator import Discriminator
from generator import ResnetGenerator
from normalization import RhoClipper
from patch_nce import PatchNCELoss
from patch_sampler import PatchSampler
from qsa_patch_sampler import QSAPatchSampler
from seed import get_seeded_generator, seed_everything, seeded_worker_init_fn
from utils import *
from visualizations import (generate_translation_example,
                            plot_translation_examples, translate_dataset)


class LaGAN:
  def __init__(self, args):
    self.model_name = 'LaGAN'
    self.result_dir = args.result_dir
    self.dataset = args.dataset
    self.ckpt = args.ckpt

    self.iters = args.iters
    self.decay_lr = args.decay_lr
    self.batch_size = args.batch_size

    self.display_freq = args.display_freq
    self.eval_freq = args.eval_freq
    self.save_freq = args.save_freq

    """Optimization"""
    self.lr = args.lr
    self.weight_decay = args.weight_decay

    """ Weights """
    self.gan_weight = args.gan_weight
    self.cam_weight = args.cam_weight
    self.nce_weight = args.nce_weight

    """ Architecture """
    self.base_channels = args.base_channels

    """ Generator """
    self.num_bottleneck_blocks = args.num_bottleneck_blocks
    self.num_downsampling_blocks = args.num_downsampling_blocks

    self.use_global_discriminator = args.use_global_discriminator
    self.use_local_discriminator = args.use_local_discriminator

    self.img_size = args.img_size
    self.img_channels = args.img_channels

    self.device = args.device
    self.benchmark = args.benchmark
    self.resume = args.resume

    self.seed = args.seed

    """ CUT """
    self.cut_type = args.cut_type
    self.nce_temperature = args.nce_temperature
    self.nce_patch_embedding_dim = args.nce_patch_embedding_dim
    self.nce_num_patches = args.nce_num_patches
    self.nce_layers = [int(x) for x in args.nce_layers.split(',')]
    self.nce_detach_keys = args.nce_detach_keys

    if torch.backends.cudnn.enabled and self.benchmark:
      print('set benchmark!')
      torch.backends.cudnn.benchmark = True

    """QSA """
    self.qsa_max_spatial_size = args.qsa_max_spatial_size

    print()
    print("##### Information #####")
    print("# CUT sampling type : ", self.cut_type)
    print("# dataset : ", self.dataset)
    if self.ckpt:
      print("# ckpt : ", self.ckpt)
    print("# batch_size : ", self.batch_size)
    print("# training iterations : ", self.iters)
    print("# seed : ", self.seed)
    print()
    print("##### Generator #####")
    print("# bottleneck blocks : ", self.num_bottleneck_blocks)
    print("##### Discriminator #####")
    print("# use global discriminator : ", self.use_global_discriminator)
    print("# use local discriminator : ", self.use_local_discriminator)

    print("##### Weight #####")
    print("# adv_weight : ", self.gan_weight)
    print("# cam_weight : ", self.cam_weight)
    print("# nce_weight : ", self.nce_weight)

    print("##### CUT #####")
    print("# nce temperature : ", self.nce_temperature)
    print("# nce layers : ", self.nce_layers)
    print("# nce patches : ", self.nce_num_patches)
    print("# nce patch embedding dim : ", self.nce_patch_embedding_dim)
    print("# nce detach keys", self.nce_detach_keys)

    assert (self.use_global_discriminator or self.use_local_discriminator)

    self.build_model()

  ##################################################################################
  # Model
  ##################################################################################

  def build_model(self):
    """ Seed everything """
    seed_everything(self.seed)
    trainA_dataloader_generator = get_seeded_generator(self.seed)
    trainB_dataloader_generator = get_seeded_generator(self.seed)

    """ DataLoader """
    train_transform = transforms.Compose([
        transforms.RandomHorizontalFlip(),
        transforms.Resize((self.img_size + 30, self.img_size+30)),
        transforms.RandomCrop(self.img_size),
        transforms.ToTensor(),
        transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5))
    ])
    test_transform = transforms.Compose([
        transforms.Resize((self.img_size, self.img_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5))
    ])

    self.trainA = ImageFolder(os.path.join('dataset', self.dataset, 'trainA'), train_transform)
    self.trainA_without_aug = ImageFolder(
        os.path.join('dataset', self.dataset, 'trainA'),
        test_transform
    )
    self.trainB = ImageFolder(os.path.join('dataset', self.dataset, 'trainB'), train_transform)
    self.valA = ImageFolder(os.path.join('dataset', self.dataset, 'valA'), test_transform)
    self.valB = ImageFolder(os.path.join('dataset', self.dataset, 'valB'), test_transform)
    self.testA = ImageFolder(os.path.join('dataset', self.dataset, 'testA'), test_transform)
    self.testB = ImageFolder(os.path.join('dataset', self.dataset, 'testB'), test_transform)
    self.trainA_loader = DataLoader(
        self.trainA,
        batch_size=self.batch_size,
        worker_init_fn=seeded_worker_init_fn,
        generator=trainA_dataloader_generator,
        shuffle=True
    )
    self.trainA_without_aug_loader = DataLoader(
        self.trainA_without_aug,
        batch_size=1,
        shuffle=False
    )
    self.trainB_loader = DataLoader(
        self.trainB,
        batch_size=self.batch_size,
        worker_init_fn=seeded_worker_init_fn,
        generator=trainB_dataloader_generator,
        shuffle=True
    )
    self.valA_loader = DataLoader(self.valA, batch_size=1, shuffle=False)
    self.valB_loader = DataLoader(self.valB, batch_size=1, shuffle=False)
    self.testA_loader = DataLoader(self.testA, batch_size=1, shuffle=False)
    self.testB_loader = DataLoader(self.testB, batch_size=1, shuffle=False)

    """ Define Generator, Discriminator """
    self.generator = ResnetGenerator(
        in_channels=3,
        out_channels=3,
        ngf=self.base_channels,
        num_bottleneck_blocks=self.num_bottleneck_blocks,
        num_downsampling_blocks=self.num_downsampling_blocks,
        nce_layers_indices=self.nce_layers
    ).to(self.device)

    if self.use_global_discriminator:
      self.global_discriminator = Discriminator(
          in_channels=3,
          ndf=self.base_channels,
          num_layers=7
      ).to(self.device)
    if self.use_local_discriminator:
      self.local_discriminator = Discriminator(
          in_channels=3,
          ndf=self.base_channels,
          num_layers=5
      ).to(self.device)

    """Define Patch Sampler"""
    if self.cut_type == 'vanilla':
      self.patch_sampler = PatchSampler(
          patch_embedding_dim=self.nce_patch_embedding_dim,
          num_patches_per_layer=self.nce_num_patches,
          device=self.device
      )
    else:
      self.patch_sampler = QSAPatchSampler(
          patch_embedding_dim=self.nce_patch_embedding_dim,
          num_patches_per_layer=self.nce_num_patches,
          qsa_type=self.cut_type,
          max_spatial_size=self.qsa_max_spatial_size,
          device=self.device
      )
    self.initialize_patch_sampler()

    print('Generator:')
    print(self.generator)
    print(f'total params: {get_total_model_params(self.generator)}')
    print(
        f'total trainable params: {get_total_trainable_model_params(self.generator)}'
    )
    if self.use_global_discriminator:
      print('Global Discriminator:')
      print(self.global_discriminator)
      print(f'total params: {get_total_model_params(self.global_discriminator)}')
      print(
          f'total trainable params: {get_total_trainable_model_params(self.global_discriminator)}'
      )
    if self.use_local_discriminator:
      print('Local Discriminator:')
      print(self.local_discriminator)
      print(f'total params: {get_total_model_params(self.local_discriminator)}')
      print(
          f'total trainable params: {get_total_trainable_model_params(self.local_discriminator)}'
      )

    """ Define Loss """
    self.mse = nn.MSELoss().to(self.device)
    self.bce = nn.BCEWithLogitsLoss().to(self.device)
    self.nce_losses = []
    for _ in self.nce_layers:
      self.nce_losses.append(
          PatchNCELoss(
              temperature=self.nce_temperature,
              use_external_patches=False,
              detach_keys=self.nce_detach_keys
          ).to(self.device)
      )
    """ Trainer """
    self.generator_optim = torch.optim.Adam(
        self.generator.parameters(),
        lr=self.lr,
        betas=(0.5, 0.999),
        weight_decay=self.weight_decay
    )

    self.discriminator_optim = torch.optim.Adam(
        itertools.chain(
            self.global_discriminator.parameters() if self.use_global_discriminator else [],
            self.local_discriminator.parameters() if self.use_local_discriminator else []
        ),
        lr=self.lr,
        betas=(0.5, 0.999),
        weight_decay=self.weight_decay
    )
    self.patch_sampler_optim = torch.optim.Adam(
        self.patch_sampler.parameters(),
        lr=self.lr,
        betas=(0.5, 0.999),
        weight_decay=self.weight_decay
    )

    """ Define Rho clipper to constraint the value of rho in AdaILN and ILN"""
    self.Rho_clipper = RhoClipper(0, 1)
    """ For model selection. """
    self.smallest_val_fid = float('inf')

  def initialize_patch_sampler(self):
    with torch.no_grad():
      # initialize patch sampler
      x = torch.zeros(
          (self.batch_size, self.img_channels, self.img_size, self.img_size),
          device=self.device
      )
      # initialize patch sampler MLPs which depend on layer outs shapes
      self.patch_sampler(self.generator.encode(x))

  def sample_patches(self, feat_q: torch.Tensor, feat_k: torch.Tensor):
    if self.cut_type == 'vanilla':
      feat_k_pool, feat_k_pool_idx = self.patch_sampler(feat_k)
      feat_q_pool, _ = self.patch_sampler(feat_q, feat_k_pool_idx)
      return (feat_q_pool, feat_k_pool)
    else:
      (feat_k_pool,
       feat_k_patch_idx,
       feat_k_attn_map) = self.patch_sampler(feat_k)
      feat_q_pool, _, _ = self.patch_sampler(
          layer_outs=feat_q,
          patch_idx_per_layer=feat_k_patch_idx,
          attn_map_per_layer=feat_k_attn_map
      )
      return (feat_q_pool, feat_k_pool)

  def calculate_nce_loss(self, src: torch.Tensor, tgt: torch.Tensor):
    n_layers = len(self.nce_layers)

    feat_q = self.generator.encode(tgt)
    feat_k = self.generator.encode(src)

    feat_q_pool, feat_k_pool = self.sample_patches(feat_q, feat_k)

    total_nce_loss = 0.0
    for f_q, f_k, pnce, _ in zip(feat_q_pool, feat_k_pool, self.nce_losses, self.nce_layers):
      total_nce_loss += pnce(f_q, f_k).mean()

    return total_nce_loss / n_layers

  def lr_scheduler_step(self):
    if self.decay_lr and self.iter > (self.iters // 2):
      decay_term = (self.lr / (self.iters // 2))
      self.generator_optim.param_groups[0]['lr'] -= decay_term
      self.discriminator_optim.param_groups[0]['lr'] -= decay_term
      self.patch_sampler_optim.param_groups[0]['lr'] -= decay_term

  def resume_scheduler_step(self, start_iter):
    if self.decay_lr and start_iter > (self.iters // 2):
      decay_term = (self.lr / (self.iters // 2)) * (start_iter - self.iters // 2)
      self.generator_optim.param_groups[0]['lr'] -= decay_term
      self.discriminator_optim.param_groups[0]['lr'] -= decay_term
      self.patch_sampler_optim.param_groups[0]['lr'] -= decay_term

  def train_discriminator(self, real_A: torch.Tensor, real_B: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    # suffix LB -> local
    # suffix GB -> global

    self.discriminator_optim.zero_grad()

    fake_A2B, _, _ = self.generator(real_A)

    if self.use_global_discriminator:
      real_GB_logit, real_GB_cam_logit, _ = self.global_discriminator(real_B)
      fake_GB_logit, fake_GB_cam_logit, _ = self.global_discriminator(fake_A2B)

      discriminator_ad_loss_GB = self.mse(
          real_GB_logit,
          torch.ones_like(real_GB_logit).to(self.device)
      ) + self.mse(
          fake_GB_logit,
          torch.zeros_like(fake_GB_logit).to(self.device)
      )
      discriminator_ad_cam_loss_GB = self.mse(
          real_GB_cam_logit,
          torch.ones_like(real_GB_cam_logit).to(self.device)
      ) + self.mse(
          fake_GB_cam_logit,
          torch.zeros_like(fake_GB_cam_logit).to(self.device)
      )
    else:
      discriminator_ad_loss_GB = 0.0
      discriminator_ad_cam_loss_GB = 0.0

    if self.use_local_discriminator:
      real_LB_logit, real_LB_cam_logit, _ = self.local_discriminator(real_B)
      fake_LB_logit, fake_LB_cam_logit, _ = self.local_discriminator(fake_A2B)

      discriminator_ad_loss_LB = self.mse(
          real_LB_logit,
          torch.ones_like(real_LB_logit).to(self.device)
      ) + self.mse(
          fake_LB_logit,
          torch.zeros_like(fake_LB_logit).to(self.device)
      )
      discriminator_ad_cam_loss_LB = self.mse(
          real_LB_cam_logit,
          torch.ones_like(real_LB_cam_logit).to(self.device)
      ) + self.mse(
          fake_LB_cam_logit,
          torch.zeros_like(fake_LB_cam_logit).to(self.device)
      )
    else:
      discriminator_ad_loss_LB = 0.0
      discriminator_ad_cam_loss_LB = 0.0

    discriminator_domain_gan_loss = discriminator_ad_loss_GB + discriminator_ad_loss_LB
    discriminator_cam_gan_loss = discriminator_ad_cam_loss_GB + discriminator_ad_cam_loss_LB

    discriminator_loss = self.gan_weight * (
        discriminator_domain_gan_loss +
        discriminator_cam_gan_loss
    )
    discriminator_loss.backward()

    self.discriminator_optim.step()

    return (discriminator_loss,
            discriminator_domain_gan_loss,
            discriminator_cam_gan_loss)

  def train_generator(self,  real_A: torch.Tensor, real_B: torch.Tensor) -> Tuple[
      Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor],
      Tuple[torch.Tensor, torch.Tensor, torch.Tensor]
  ]:
    # suffix LB -> local
    # suffix GB -> global
    self.generator_optim.zero_grad()
    # Update patch sampler. If we are in the first iteration, its optimizer is not initialized.
    # This is because patch sampler MLPs's dimension depends on the feature map shapes which are known in the first forward pass of NCE loss.
    self.patch_sampler_optim.zero_grad()

    fake_A2B, fake_A2B_cam_logit, _ = self.generator(real_A)
    fake_B2B, fake_B2B_cam_logit, _ = self.generator(real_B)

    if self.use_global_discriminator:
      fake_GB_logit, fake_GB_cam_logit, _ = self.global_discriminator(fake_A2B)
    if self.use_local_discriminator:
      fake_LB_logit, fake_LB_cam_logit, _ = self.local_discriminator(fake_A2B)

    if self.use_global_discriminator:
      generator_ad_loss_GB = self.mse(
          fake_GB_logit,
          torch.ones_like(fake_GB_logit).to(self.device)
      )
      generator_ad_cam_loss_GB = self.mse(
          fake_GB_cam_logit,
          torch.ones_like(fake_GB_cam_logit).to(self.device)
      )
    else:
      generator_ad_loss_GB = 0.0
      generator_ad_cam_loss_GB = 0.0

    if self.use_local_discriminator:
      generator_ad_loss_LB = self.mse(
          fake_LB_logit,
          torch.ones_like(fake_LB_logit).to(self.device)
      )
      generator_ad_cam_loss_LB = self.mse(
          fake_LB_cam_logit,
          torch.ones_like(fake_LB_cam_logit).to(self.device)
      )
    else:
      generator_ad_loss_LB = 0.0
      generator_ad_cam_loss_LB = 0.0

    generator_cam_loss = self.bce(
        fake_A2B_cam_logit,
        torch.ones_like(fake_A2B_cam_logit).to(self.device)
    ) + self.bce(
        fake_B2B_cam_logit,
        torch.zeros_like(fake_B2B_cam_logit).to(self.device)
    )

    nce_loss_x = self.calculate_nce_loss(real_A, fake_A2B)
    nce_loss_y = self.calculate_nce_loss(real_B, fake_B2B)
    nce_loss_both = (nce_loss_x + nce_loss_y) * 0.5

    generator_domain_gan_loss = generator_ad_loss_GB + generator_ad_loss_LB
    generator_cam_gan_loss = generator_ad_cam_loss_GB + generator_ad_cam_loss_LB
    generator_loss = (
        self.gan_weight * (
            generator_domain_gan_loss +
            generator_cam_gan_loss
        ) +
        self.nce_weight * nce_loss_both +
        self.cam_weight * generator_cam_loss
    )
    generator_loss.backward()

    self.generator_optim.step()
    self.patch_sampler_optim.step()

    return (
        generator_loss,
        generator_domain_gan_loss,
        generator_cam_gan_loss,
        generator_cam_loss,
    ), (nce_loss_both,
        nce_loss_x,
        nce_loss_y)

  def train(self):
    self.start_iter = 1
    if self.resume:
      model_list = glob(
          os.path.join(self.result_dir, self.dataset, 'model', '*.pt')
      )
      if not len(model_list) == 0:
        self.load(ckpt=self.ckpt)
        self.resume_scheduler_step()
        print("[*] load successful!")

    loss_log_file = os.path.join(self.result_dir, self.dataset, 'loss_log.txt')
    lr_log_file = os.path.join(self.result_dir, self.dataset, 'lr_log.txt')

    if os.path.exists(loss_log_file):
      os.remove(loss_log_file)
    if os.path.exists(lr_log_file):
      os.remove(lr_log_file)

    print('[*] training...')
    start_time = time.time()

    for it in range(self.start_iter, self.iters + 1):
      self.iter = it
      self.lr_scheduler_step()

      self.generator.train()
      if self.use_global_discriminator:
        self.global_discriminator.train()
      if self.use_local_discriminator:
        self.local_discriminator.train()
      self.patch_sampler.train()

      try:
        real_A, _ = next(trainA_iter)
      except:
        trainA_iter = iter(self.trainA_loader)
        real_A, _ = next(trainA_iter)
      try:
        real_B, _ = next(trainB_iter)
      except:
        trainB_iter = iter(self.trainB_loader)
        real_B, _ = next(trainB_iter)

      real_A, real_B = real_A.to(self.device), real_B.to(self.device)

      # train discriminator
      (discriminator_loss,
       discriminator_domain_gan_loss,
       discriminator_cam_gan_loss) = self.train_discriminator(real_A, real_B)
      # train generator
      (
          (generator_loss,
           generator_domain_gan_loss,
           generator_cam_gan_loss,
           generator_cam_loss),
          (nce_loss_total,
           nce_loss_x,
           nce_loss_y)
      ) = self.train_generator(real_A, real_B)
      # clip parameter of AdaILN and ILN, applied after all optimizer steps
      self.generator.apply(self.Rho_clipper)
      lr_line_parts = [
          'g_lr: %.8f, d_lr: %.8f, ps_lr: %.8f' % (
              self.generator_optim.param_groups[0]['lr'],
              self.discriminator_optim.param_groups[0]['lr'],
              self.patch_sampler_optim.param_groups[0]['lr'],
          )
      ]
      loss_line_parts = [
          'd_loss: %.8f, d_dom_gan_loss: %.8f, d_cam_gan_loss: %.8f' % (
              discriminator_loss,
              discriminator_domain_gan_loss,
              discriminator_cam_gan_loss,),
          'g_loss: %.8f, g_dom_gan_loss: %.8f, g_cam_gan_loss: %.8f, g_cam_loss: %.8f' % (
              generator_loss,
              generator_domain_gan_loss,
              generator_cam_gan_loss,
              generator_cam_loss
          ),
          'nce_loss: %.8f, nce_x: %.8f, nce_y: %.8f' % (
              nce_loss_total,
              nce_loss_x,
              nce_loss_y
          ),
      ]
      time_line_part = '[%5d/%5d] time: %4.4f, ' % (it, self.iters, time.time() - start_time)
      lr_line = time_line_part + ', '.join(lr_line_parts)
      loss_line = time_line_part + ', '.join(loss_line_parts)

      print(loss_line)
      with open(loss_log_file, 'a') as ll:
        ll.write(f'{loss_line}\n')
      with open(lr_log_file, 'a') as ll:
        ll.write(f'{lr_line}\n')

      if it % self.display_freq == 0:
        self.generator.eval(),
        if self.use_global_discriminator:
          self.global_discriminator.eval()
        if self.use_local_discriminator:
          self.local_discriminator.eval()
        self.patch_sampler.eval()

        plot_translation_examples(
            generator=self.generator,
            patch_sampler=self.patch_sampler,
            trainA_iter=iter(self.trainA_loader),
            trainA_loader=self.trainA_loader,
            trainB_iter=iter(self.trainB_loader),
            trainB_loader=self.trainB_loader,
            valA_iter=iter(self.valA_loader),
            valA_loader=self.valA_loader,
            valB_iter=iter(self.valB_loader),
            valB_loader=self.valB_loader,
            testA_iter=iter(self.testA_loader),
            testA_loader=self.testA_loader,
            testB_iter=iter(self.testB_loader),
            testB_loader=self.testB_loader,
            device=self.device,
            A2B_results_filename=os.path.join(
                self.result_dir,
                self.dataset,
                'translations',
                'evolution',
                'A2B_%07d.png' % it
            ),
            B2B_results_filename=os.path.join(
                self.result_dir,
                self.dataset,
                'translations',
                'evolution',
                'B2B_%07d.png' % it
            ),
            img_size=self.img_size,
            cut_type=self.cut_type
        )

        self.generator.train()
        if self.use_global_discriminator:
          self.global_discriminator.train()
        if self.use_local_discriminator:
          self.local_discriminator.train()
        self.patch_sampler.train()

      if it % self.save_freq == 0:
        self.save(ckpt_file_name='iter_%07d.pt' % it)

      if it % 1000 == 0:
        self.save(ckpt_file_name='latest.pt')

      if it % self.eval_freq == 0:
        self.eval()

  def save(self, ckpt_file_name: str):
    params = {}
    params['generator'] = self.generator.state_dict()
    if self.use_global_discriminator:
      params['global_discriminator'] = self.global_discriminator.state_dict()
    if self.use_local_discriminator:
      params['local_discriminator'] = self.local_discriminator.state_dict()
    params['patch_sampler'] = self.patch_sampler.state_dict()
    params['generator_optim'] = self.generator_optim.state_dict()
    params['discriminator_optim'] = self.discriminator_optim.state_dict()
    params['patch_sampler_optim'] = self.patch_sampler_optim.state_dict()
    params['iter'] = self.iter
    params['smallest_val_fid'] = self.smallest_val_fid
    torch.save(
        params,
        os.path.join(
            self.result_dir,
            self.dataset,
            'model',
            ckpt_file_name
        )
    )

  def load(self, cktpt_file_name: str):
    params = torch.load(
        os.path.join(
            self.result_dir,
            self.dataset,
            'model',
            cktpt_file_name
        )
    )
    self.generator.load_state_dict(params['generator'])
    if self.use_global_discriminator:
      self.global_discriminator.load_state_dict(params['global_discriminator'])
    if self.use_local_discriminator:
      self.local_discriminator.load_state_dict(params['local_discriminator'])
    self.start_iter = params['iter']
    self.smallest_val_fid = params['smallest_val_fid']
    # setup patch sampler
    self.patch_sampler.load_state_dict(params['patch_sampler'])
    # restore optimizers
    self.generator_optim.load_state_dict(params['generator_optim'])
    self.discriminator_optim.load_state_dict(params['discriminator_optim'])
    self.patch_sampler_optim.load_state_dict(params['patch_sampler_optim'])

  def eval(self):
    results_dir = Path(self.result_dir, self.dataset)

    model_translations_dir = Path(self.result_dir, self.dataset, 'translations')
    if not os.path.exists(model_translations_dir):
      os.mkdir(model_translations_dir)
    model_iter_translations_dir = Path(
        model_translations_dir,
        'iter_%07d' % self.iter
    )

    if not os.path.exists(model_iter_translations_dir):
      os.mkdir(model_iter_translations_dir)

    model_train_translations_dir = Path(model_iter_translations_dir, 'train')
    model_val_translations_dir = Path(model_iter_translations_dir, 'val')

    if not os.path.exists(model_train_translations_dir):
      os.mkdir(model_train_translations_dir)
    if not os.path.exists(model_val_translations_dir):
      os.mkdir(model_val_translations_dir)

    if not os.path.exists(model_train_translations_dir):
      os.mkdir(model_train_translations_dir)
    if not os.path.exists(model_val_translations_dir):
      os.mkdir(model_val_translations_dir)

    self.generator.eval()
    # translate train and val
    print('translating train...')
    translate_dataset(
        dataset_loader=self.trainA_without_aug_loader,
        translations_dirs=[model_train_translations_dir],
        generator=self.generator,
        device=self.device
    )
    print('translating val...')
    translate_dataset(
        dataset_loader=self.valA_loader,
        translations_dirs=[model_val_translations_dir],
        generator=self.generator,
        device=self.device
    )
    # compute metrics
    target_real_train_dir = os.path.join(
        'dataset',
        self.dataset,
        'trainB'
    )
    target_real_val_dir = os.path.join(
        'dataset',
        self.dataset,
        'valB'
    )
    train_metrics = torch_fidelity.calculate_metrics(
        input1=str(target_real_train_dir),
        input2=str(Path(model_train_translations_dir)),  # fake dir
        fid=True,
        verbose=False,
        cuda=torch.cuda.is_available(),
    )
    val_metrics = torch_fidelity.calculate_metrics(
        input1=str(target_real_val_dir),
        input2=str(Path(model_val_translations_dir)),  # fake dir,
        fid=True,
        verbose=False,
        rng_seed=self.seed,
        cuda=torch.cuda.is_available(),
    )
    # update logs
    train_log_file = os.path.join(results_dir, 'train_log.txt')
    val_log_file = os.path.join(results_dir, 'val_log.txt')
    smallest_val_fid_file = os.path.join(results_dir, 'smallest_val_fid.txt')
    with open(train_log_file, 'a') as tl:
      tl.write(f'iter: {self.iter}\n')
      tl.write(
          f'frechet_inception_distance: {train_metrics["frechet_inception_distance"]}\n'
      )
    with open(val_log_file, 'a') as vl:
      vl.write(f'iter: {self.iter}\n')
      vl.write(
          f'frechet_inception_distance: {val_metrics["frechet_inception_distance"]}\n'
      )
    # track frechet_inception_distance
    if val_metrics['frechet_inception_distance'] < self.smallest_val_fid:
      self.smallest_val_fid = val_metrics['frechet_inception_distance']
      print(
          f'{self.smallest_val_fid} is the smallest val fid so far, saving this model...'
      )
      self.save(ckpt_file_name='smallest_val_fid.pt')
      if os.path.exists(smallest_val_fid_file):
        os.remove(smallest_val_fid_file)

      with open(smallest_val_fid_file, 'a') as tl:
        tl.write(
            f'iter: {self.iter}\n'
        )
        tl.write(
            f'frechet_inception_distance: {val_metrics["frechet_inception_distance"]}\n'
        )
    self.generator.train()

  def test(self):
    model_list = glob(
        os.path.join(
            self.result_dir,
            self.dataset,
            'model',
            '*.pt'
        )
    )

    if not len(model_list) == 0:
      self.load(cktpt_file_name=self.ckpt)
      print("[*] load successful")
    else:
      print("[*] load failed")
      return

    self.generator.eval()
    for n, (real_A, _) in enumerate(self.testA_loader):
      img_path, _ = self.testA_loader.dataset.samples[n]
      img_name = Path(img_path).name.split('.')[0]
      translated = generate_translation_example(
          real_A,
          generator=self.generator,
          device=self.device,
          include_cam_heatmap=True
      )
      translated_out_file = os.path.join(
          self.result_dir, self.dataset, 'test', f'{img_name}_fake_B.jpg'
      )
      cv2.imwrite(translated_out_file, translated)

    for n, (real_B, _) in enumerate(self.testB_loader):
      img_path, _ = self.testB_loader.dataset.samples[n]
      img_name = Path(img_path).name.split('.')[0]
      translated = generate_translation_example(
          real_B,
          generator=self.generator,
          device=self.device,
          include_cam_heatmap=True
      )
      translated_out_file = os.path.join(
          self.result_dir, self.dataset, 'test', f'{img_name}_fake_B.jpg'
      )
      cv2.imwrite(translated_out_file, translated)

  def translate(self):
    model_list = glob(
        os.path.join(
            self.result_dir,
            self.dataset,
            'model',
            '*.pt'
        )
    )
    if not len(model_list) == 0:
      self.load(cktpt_file_name=self.ckpt)
      print("[*] load successful")
    else:
      print("[*] load failed")
      return

    if not os.path.exists('translations'):
      os.mkdir('translations')

    model_translations_dir = Path('translations', self.dataset)
    if not os.path.exists(model_translations_dir):
      os.mkdir(model_translations_dir)

    model_with_ckpt_translations_dir = Path(
        model_translations_dir,
        Path(self.ckpt).stem
    )
    if not os.path.exists(model_with_ckpt_translations_dir):
      os.mkdir(model_with_ckpt_translations_dir)

    train_translated_imgs_dir = Path(model_with_ckpt_translations_dir, 'train')
    val_translated_imgs_dir = Path(model_with_ckpt_translations_dir, 'val')
    test_translated_imgs_dir = Path(model_with_ckpt_translations_dir, 'test')
    full_translated_imgs_dir = Path(model_with_ckpt_translations_dir, 'full')

    if not os.path.exists(train_translated_imgs_dir):
      os.mkdir(train_translated_imgs_dir)
    if not os.path.exists(test_translated_imgs_dir):
      os.mkdir(test_translated_imgs_dir)
    if not os.path.exists(val_translated_imgs_dir):
      os.mkdir(val_translated_imgs_dir)
    if not os.path.exists(full_translated_imgs_dir):
      os.mkdir(full_translated_imgs_dir)

    self.generator.eval()

    print('translating train...')
    translate_dataset(
        dataset_loader=self.trainA_without_aug_loader,
        translations_dirs=[train_translated_imgs_dir, full_translated_imgs_dir],
        generator=self.generator,
        device=self.device
    )
    print('translating val...')
    translate_dataset(
        dataset_loader=self.valA_loader,
        translations_dirs=[val_translated_imgs_dir, full_translated_imgs_dir],
        generator=self.generator,
        device=self.device
    )
    print('translating test...')
    translate_dataset(
        dataset_loader=self.testA_loader,
        translations_dirs=[test_translated_imgs_dir, full_translated_imgs_dir],
        generator=self.generator,
        device=self.device
    )
