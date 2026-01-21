#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
General utility functions for reproducibility and common operations.
Provides random seed setting for deterministic model training.

通用工具函数模块，用于可复现性和常用操作。
提供随机种子设置以实现确定性模型训练。

Author: Wenhao Wang
"""
import torch
import numpy as np
import random
import logging

def set_seed(seed: int):
    """Sets the random seed for reproducibility.

    Args:
        seed (int): The seed to set.
    """
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        # Disabling benchmark and enabling deterministic mode can impact performance,
        # but is necessary for reproducibility.
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    logging.info(f"Global random seed set to: {seed}")