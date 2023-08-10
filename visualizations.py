from itertools import chain
from pathlib import Path
from typing import Iterable, List

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from utils import *


def translate_dataset(dataset_loader: DataLoader,
                      translations_dirs: List[str],
                      generator: nn.Module,
                      device: torch.device,
                      include_attention: bool = False,
                      attention_position: str = 'horizontal'
                      ):
  generator = generator.to(device)
  with torch.inference_mode():
    for n, (real_A, _) in enumerate(dataset_loader):
      real_A = real_A.to(device)
      img_path, _ = dataset_loader.dataset.samples[n]
      img_name = Path(img_path).name.split('.')[0]
      if not include_attention:
        fake_B = generator(real_A, cam=False)
        fake_B_RGB = RGB2BGR(tensor2numpy(denorm(fake_B[0]))) * 255.0
      else:
        _, _, H, W = real_A.shape
        assert (H == W)
        fake_B, _, fake_B_heatmap = generator(real_A, cam=True)
        if attention_position == 'horizontal':
          fake_B_RGB = np.hstack([
              cam(tensor2numpy(fake_B_heatmap[0]), H) * 255.0,
              RGB2BGR(tensor2numpy(denorm(fake_B[0]))) * 255.0,
          ])
        else:
          fake_B_RGB = np.vstack([
              cam(tensor2numpy(fake_B_heatmap[0]), H) * 255.0,
              RGB2BGR(tensor2numpy(denorm(fake_B[0]))) * 255.0,
          ])

      for translations_dir in translations_dirs:
        cv2.imwrite(
            os.path.join(
                translations_dir,
                f'{img_name}_fake_B.jpg'
            ),
            fake_B_RGB
        )


def generate_translation_example(
    real: torch.Tensor,
    generator: nn.Module,
    device: torch.device,
    include_cam_heatmap: bool = False
):
  generator = generator.to(device)
  real = real.to(device)

  if include_cam_heatmap:
    _, _, H, W = real.shape
    assert (H == W)
    with torch.inference_mode():
      fake, _, heatmap = generator(real, True)
      return np.vstack([
          RGB2BGR(tensor2numpy(denorm(real[0]))) * 255.0,
          cam(tensor2numpy(heatmap[0]), H) * 255.0,
          RGB2BGR(tensor2numpy(denorm(fake[0]))) * 255.0,
      ])
  else:
    with torch.inference_mode():
      fake = generator(real, False)
      return np.vstack([
          RGB2BGR(tensor2numpy(denorm(real[0]))) * 255.0,
          RGB2BGR(tensor2numpy(denorm(fake[0]))) * 255.0,
      ])


def plot_translation_examples(
        generator: nn.Module,
        patch_sampler: nn.Module,
        trainA_iter: Iterable,
        trainA_loader: DataLoader,
        trainB_iter: Iterable,
        trainB_loader: DataLoader,
        valA_iter: Iterable,
        valA_loader: DataLoader,
        valB_iter: Iterable,
        valB_loader: DataLoader,
        testA_iter: Iterable,
        testA_loader: DataLoader,
        testB_iter: Iterable,
        testB_loader: DataLoader,
        device: torch.device,
        A2B_results_filename: str,
        B2B_results_filename: str,
        train_examples_num: int = 4,
        val_examples_num: int = 4,
        test_examples_num: int = 4,
        img_size: int = 256,
        cut_type: str = 'vanilla'
):

  def generate_translation_examples_matrix(
      A_examples_iter: Iterable,
      A_examples_loader: DataLoader,
      B_examples_iter: Iterable,
      B_examples_loader: DataLoader,
      examples_num: int
  ):
    with torch.no_grad():
      A2B, B2B = [], []
      for _ in range(examples_num):
        try:
          real_A, _ = next(A_examples_iter)
        except:
          A_examples_iter = iter(A_examples_loader)
          real_A, _ = next(A_examples_iter)
        try:
          real_B, _ = next(B_examples_iter)
        except:
          B_examples_iter = iter(B_examples_loader)
          real_B, _ = next(B_examples_iter)

        real_A = real_A.to(device)
        real_B = real_B.to(device)

        if cut_type == 'vanilla':
          A2B.append(generate_translation_example(real_A, generator, include_cam_heatmap=True, device=device))
          B2B.append(generate_translation_example(real_B, generator, include_cam_heatmap=True, device=device))
        else:
          loss_attn_A2B = patch_sampler(
              generator.encode(real_A),
              return_only_full_attn_maps=True,
          )
          loss_attn_B2B = patch_sampler(
              generator.encode(real_B),
              return_only_full_attn_maps=True,
          )
          A2B.append(
              np.vstack(
                  [generate_translation_example(
                      real_A,
                      generator,
                      include_cam_heatmap=True,
                      device=device
                  )] + [
                      cam(tensor2numpy(attn), img_size) * 255
                      for attn in loss_attn_A2B
                  ])
          )
          B2B.append(
              np.vstack(
                  [generate_translation_example(
                      real_B,
                      generator,
                      include_cam_heatmap=True,
                      device=device
                  )] + [
                      cam(tensor2numpy(attn), img_size) * 255
                      for attn in loss_attn_B2B
                  ])
          )
      return A2B, B2B

  train_A2B, train_B2B = generate_translation_examples_matrix(
      A_examples_iter=trainA_iter,
      A_examples_loader=trainA_loader,
      B_examples_iter=trainB_iter,
      B_examples_loader=trainB_loader,
      examples_num=train_examples_num
  )
  val_A2B, val_B2B = generate_translation_examples_matrix(
      A_examples_iter=valA_iter,
      A_examples_loader=valA_loader,
      B_examples_iter=valB_iter,
      B_examples_loader=valB_loader,
      examples_num=val_examples_num
  )
  test_A2B, test_B2B = generate_translation_examples_matrix(
      A_examples_iter=testA_iter,
      A_examples_loader=testA_loader,
      B_examples_iter=testB_iter,
      B_examples_loader=testB_loader,
      examples_num=test_examples_num
  )

  A2B = np.hstack(list(chain(train_A2B, val_A2B, test_A2B)))
  B2B = np.hstack(list(chain(train_B2B, val_B2B, test_B2B)))

  cv2.imwrite(A2B_results_filename, A2B)
  cv2.imwrite(B2B_results_filename, B2B)
