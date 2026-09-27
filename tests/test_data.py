"""data 单测：切分无泄漏 + 可复现。作者: 晨星"""

import numpy as np

from semiforge.data.synthetic import make_ssl_split


def test_split_no_leakage():
    sp = make_ssl_split("moons", 400, 6, 60, 100, seed=0)
    # 三池互斥（索引级不重叠 → 样本级也必然不同集合，但直接验证样本行不完全相同集合）
    lab = {tuple(row) for row in sp.X_l}
    val = {tuple(row) for row in sp.X_val}
    tst = {tuple(row) for row in sp.X_test}
    unl = {tuple(row) for row in sp.X_u}
    assert not (lab & val)
    assert not (lab & tst)
    assert not (val & tst)
    assert not (lab & unl)
    assert len(sp.X_l) == 12
    assert len(sp.X_val) == 60
    assert len(sp.X_test) == 100


def test_split_stratified():
    sp = make_ssl_split("moons", 400, 6, 60, 100, seed=0)
    assert set(np.unique(sp.y_l)) == {0, 1}
    assert int(np.sum(sp.y_l == 0)) == 6


def test_split_reproducible():
    a = make_ssl_split("circles", 400, 6, 60, 100, seed=3)
    b = make_ssl_split("circles", 400, 6, 60, 100, seed=3)
    assert np.array_equal(a.X_l, b.X_l)
    assert np.array_equal(a.y_test, b.y_test)


def test_split_multiclass_digits():
    sp = make_ssl_split("digits_sub", 600, 5, 60, 90, seed=1)
    assert sp.n_classes == 6
    assert len(sp.X_l) == 30
