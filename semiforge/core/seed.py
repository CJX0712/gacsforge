"""全局确定性 seed 入口。作者: 晨星"""

from __future__ import annotations

import random

import numpy as np


def set_all(seed: int) -> None:
    """唯一 seed 入口：python random + numpy。库级 seed 在各方法内使用同一 seed 传入。"""
    random.seed(seed)
    np.random.seed(seed)
