#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import random, numpy as np, torch

def set_all_seeds(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

