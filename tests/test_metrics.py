"""metrics 单测（手写实现 vs 手算值）。作者: 晨星"""

import numpy as np
import pytest

from semiforge.eval.metrics import accuracy, macro_f1, significant


def test_accuracy():
    y = np.array([0, 0, 1, 1])
    p = np.array([0, 1, 1, 1])
    assert accuracy(y, p) == 0.75


def test_macro_f1_hand():
    y = np.array([0, 0, 1, 1])
    p = np.array([0, 1, 1, 1])
    # class0: P=1, R=0.5, F1=2/3 ; class1: P=2/3, R=1, F1=0.8
    assert macro_f1(y, p) == pytest.approx((2 / 3 + 0.8) / 2)


def test_macro_f1_perfect():
    y = np.array([0, 1, 2, 2])
    assert macro_f1(y, y) == pytest.approx(1.0)


def test_length_mismatch_raises():
    with pytest.raises(ValueError):
        accuracy(np.array([0, 1]), np.array([0]))


def test_significant_rule():
    assert significant(0.80, 0.01, 0.70, 0.01)  # diff 0.10 > 0.5*0.02
    assert not significant(0.71, 0.05, 0.70, 0.05)  # diff 0.01 < 0.5*0.10
