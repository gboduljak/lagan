import os
import random

import numpy as np
import torch


# Taken from https://pytorch.org/docs/stable/notes/randomness.html
def seeded_worker_init_fn(worker_id: int):
  worker_seed = torch.initial_seed() % 2**32
  np.random.seed(worker_seed)
  random.seed(worker_seed)


def get_seeded_generator(seed: int):
  g = torch.Generator()
  g.manual_seed(seed)
  return g


def seed_everything(seed: int):
  random.seed(seed)
  os.environ['PYTHONHASHSEED'] = str(seed)
  np.random.seed(seed)
  torch.manual_seed(seed)
  torch.cuda.manual_seed(seed)
  torch.mps.manual_seed(seed)
