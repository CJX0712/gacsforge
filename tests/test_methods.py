"""方法级单测：契约 + 概率归一 + 确定性 + 离线兜底。作者: 晨星"""

import numpy as np
import pytest

from semiforge.data.synthetic import make_ssl_split
from semiforge.ssl.baselines import MajorityClassifier, SupervisedRF
from semiforge.ssl.classical import SKLEARN_OK, SelfTrainingRF
from semiforge.ssl.factory import build_methods
from semiforge.ssl.flagship import GACS
from semiforge.ssl.numpy_fallback import NumpyKNNClassifier, NumpyLabelPropagate, NumpySelfTrain


def _tiny(seed=0):
    return make_ssl_split("moons", 300, 6, 50, 80, seed=seed)


METHODS = []
if SKLEARN_OK:
    METHODS += [
        (lambda s: SupervisedRF(rf_trees=40, seed=s), "supervised_rf"),
        (lambda s: SelfTrainingRF(rf_trees=40, seed=s, max_iter=8), "selftrain_rf"),
    ]
METHODS += [
    (lambda s: NumpyKNNClassifier(k=5), "numpy_knn"),
    (lambda s: NumpyLabelPropagate(k=5, iters=30), "numpy_labelprop"),
    (lambda s: NumpySelfTrain(k=5, max_iter=5), "numpy_selftrain"),
    (lambda s: GACS(rf_trees=40, seed=s, rounds=4), "gacs"),
    (lambda s: MajorityClassifier(), "majority"),
]


@pytest.mark.parametrize(
    "factory,name", [(f, n) for f, n in METHODS], ids=[n for _, n in METHODS]
)
def test_method_contract(factory, name):
    sp = _tiny()
    m = factory(0).fit(sp.X_l, sp.y_l, sp.X_u)
    P = m.predict_proba(sp.X_test)
    assert P.shape == (len(sp.X_test), sp.n_classes)
    assert np.allclose(P.sum(axis=1), 1.0, atol=1e-5)
    pred = m.predict(sp.X_test)
    assert set(np.unique(pred)).issubset(set(np.unique(sp.y_l)))
    acc = float(np.mean(pred == sp.y_test))
    if name != "majority":
        assert acc > 0.5, f"{name} acc={acc}"


def test_determinism_gacs():
    sp = _tiny()
    a = GACS(rf_trees=30, seed=1, rounds=3).fit(sp.X_l, sp.y_l, sp.X_u).predict(sp.X_test)
    b = GACS(rf_trees=30, seed=1, rounds=3).fit(sp.X_l, sp.y_l, sp.X_u).predict(sp.X_test)
    assert np.array_equal(a, b)


def test_gacs_pseudo_labels_added():
    sp = _tiny()
    m = GACS(rf_trees=30, seed=0, rounds=5).fit(sp.X_l, sp.y_l, sp.X_u)
    assert m.n_pseudo_ > 0


def test_gacs_ablation_flags():
    sp = _tiny()
    for params in ({"use_graph": False}, {"balance": False}, {"use_anneal": False}):
        m = GACS(rf_trees=30, seed=0, rounds=3, **params).fit(sp.X_l, sp.y_l, sp.X_u)
        acc = float(np.mean(m.predict(sp.X_test) == sp.y_test))
        assert acc > 0.5


def test_numpy_labelprop_graph_invariant():
    """不变量：传播算子谱半径 <= alpha < 1（行和=1 的 S），F 收敛有界。"""
    sp = _tiny()
    m = NumpyLabelPropagate(k=5, alpha=0.75, iters=80).fit(sp.X_l, sp.y_l, sp.X_u)
    assert np.isfinite(m.F_).all()
    assert m.F_.max() <= 1.0 + 1e-6 and m.F_.min() >= -1e-6


def test_factory_sklearn_missing_fallback(monkeypatch):
    import semiforge.ssl.factory as factory_mod

    monkeypatch.setattr(factory_mod, "available_sklearn", lambda: False)
    from semiforge.core.config import Config

    cfg = Config()
    methods = build_methods(cfg, seed=0)
    names = [m.name for m in methods]
    assert "numpy_selftrain" in names and "selftrain_rf" not in names
    assert "gacs" in names
    # 兜底路径可跑通
    sp = _tiny()
    g = [m for m in methods if m.name == "gacs"][0]
    g.fit(sp.X_l, sp.y_l, sp.X_u)
    assert float(np.mean(g.predict(sp.X_test) == sp.y_test)) > 0.5


def test_gacs_invalid_quantile():
    with pytest.raises(Exception):
        GACS(quantile=1.0)
