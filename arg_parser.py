
import argparse

from qsa_patch_sampler import QSAType
from utils import *


def parse_args():
  desc = "Pytorch implementation of LaGAN"
  parser = argparse.ArgumentParser(description=desc)
  parser.add_argument('--phase', type=str, default='train', help='[train / test / translate]')
  parser.add_argument('--dataset', type=str, default='YOUR_DATASET_NAME', help='dataset name')
  parser.add_argument('--ckpt', type=str, default=None, help='checkpoint to load.')
  # Training config
  parser.add_argument('--iters', type=int, default=500000, help='number of training iterations')
  parser.add_argument('--batch_size', type=int, default=1, help='batch size')
  parser.add_argument('--display_freq', type=int, default=1000, help='translations display freq')
  parser.add_argument('--eval_freq', type=int, default=25000, help='eval freq')
  parser.add_argument('--save_freq', type=int, default=100000, help='model save freq')
  parser.add_argument('--decay_lr', type=str2bool, default=True, help='should decay lr')
  parser.add_argument('--benchmark', type=str2bool, default=False)
  parser.add_argument('--resume', type=str2bool, default=False)
  # Translate config
  parser.add_argument('--translate_include_attention', type=str2bool, default=False)
  # U-GAT-IT Defaults
  parser.add_argument('--lr', type=float, default=0.0001, help='learning rate')
  parser.add_argument('--weight_decay', type=float, default=0.0001, help='weight decay')
  parser.add_argument('--gan_weight', type=int, default=1, help='weight for GAN loss')
  parser.add_argument('--cam_weight', type=int, default=1000, help='weight for CAM loss')
  # Generator
  parser.add_argument('--base_channels', type=int, default=64, help='base channel number per layer')
  parser.add_argument('--num_bottleneck_blocks', type=int, default=9, help='number of generator bottleneck blocks')
  parser.add_argument('--num_downsampling_blocks', type=int, default=2, help='number of generator downsampling blocks')
  # Discriminator
  parser.add_argument('--use_global_discriminator',
                      type=str2bool,
                      default=True,
                      help='should use global discrimininator')
  parser.add_argument('--use_local_discriminator',
                      type=str2bool,
                      default=True,
                      help='should use local discrimininator')
  # Input
  parser.add_argument('--img_size', type=int, default=256, help='size of image')
  parser.add_argument('--img_channels', type=int, default=3, help='image channels')
  # Results
  parser.add_argument('--result_dir', type=str, default='results', help='directory to save the results')
  # Device
  parser.add_argument('--device', type=str, default='cuda',
                      choices=['cpu', 'cuda', 'mps'],
                      help='Set gpu mode; [cpu, cuda, mps]')
  parser.add_argument('--seed', type=int, default=269902365, help='seed')
  # CUT Defaults
  parser.add_argument('--nce_weight', type=float, default=10.0, help='weight for NCE loss: NCE(G(X), X)')
  parser.add_argument('--cut_type', type=str, default='vanilla',
                      choices=['vanilla',
                               QSAType.GLOBAL,
                               QSAType.LOCAL,
                               QSAType.GLOBAL_AND_LOCAL],
                      help='set cut sampling type.')
  parser.add_argument(
      '--nce_idt',
      type=str2bool,
      nargs='?',
      const=True,
      default=False,
      help='use NCE loss for identity mapping: NCE(G(Y), Y))'
  )
  parser.add_argument(
      '--nce_temperature',
      type=float,
      default=0.07,
      help='temperature for NCE loss'
  )
  parser.add_argument(
      '--nce_patch_embedding_dim',
      type=int,
      default=256
  )
  parser.add_argument(
      '--nce_detach_keys',
      type=str2bool,
      nargs='?',
      const=True,
      default=True,
      help='detach keys in nce loss forward pass or not'
  )
  parser.add_argument(
      '--nce_num_patches',
      type=int,
      default=256,
      help='number of patches per layer'
  )
  parser.add_argument(
      '--nce_layers',
      type=str,
      default='0,2,3,4,8',
      help='layers contributing to nce loss'
  )
  parser.add_argument(
      '--qsa_max_spatial_size',
      type=int,
      default=64 * 64,
      help='max spatial size of layer for QSA sampling'
  )
  return check_args(parser.parse_args())


"""checking arguments"""


def check_args(args):
  # --result_dir
  ensure_folder_exists(os.path.join(args.result_dir, args.dataset, 'model'))
  ensure_folder_exists(os.path.join(args.result_dir, args.dataset, 'translations'))
  ensure_folder_exists(os.path.join(args.result_dir, args.dataset, 'translations', 'evolution'))
  ensure_folder_exists(os.path.join(args.result_dir, args.dataset, 'test'))
  # --iters
  try:
    assert args.iters >= 1
  except:
    print('number of iters must be larger than or equal to one')
  # --batch_size
  try:
    assert args.batch_size >= 1
  except:
    print('batch size must be larger than or equal to one')
  # --nce_layers
  try:
    assert len(''.join(args.nce_layers.split(','))) >= 1
  except:
    print('there must be at least one layer for NCE loss')
  return args
