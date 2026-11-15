"""The hand-written backward pass must agree with numerical differentiation."""
import numpy as np
from terrafovea import model


def test_gradients_match_finite_differences():
    r = np.random.default_rng(0)
    n = 400
    x, y = r.uniform(0, 6, n), r.uniform(0, 6, n)
    z = r.uniform(0, 2, n).astype(np.float32); h = z.copy()
    f0, stats, groups = model.build_features(x, y, z, h, r.random(n).astype(np.float32), np.hypot(x, y).astype(np.float32))
    labels = r.integers(0, 8, n)
    p = {k: v.astype(np.float64) for k, v in model.init_params(1).items()}
    f0 = f0.astype(np.float64); stats = stats.astype(np.float64)
    cw = np.ones(8)
    _F = model.F
    model.F = np.float64
    try:
        loss, g, _ = model.loss_and_grads(p, f0, stats, groups, labels, cw)
        worst = 0.0
        for k in p:
            flat = p[k].reshape(-1)
            for i in r.choice(flat.size, size=min(6, flat.size), replace=False):
                old = flat[i]
                flat[i] = old + 1e-5; lp = model.loss_and_grads(p, f0, stats, groups, labels, cw)[0]
                flat[i] = old - 1e-5; lm = model.loss_and_grads(p, f0, stats, groups, labels, cw)[0]
                flat[i] = old
                num = (lp - lm) / 2e-5
                ana = g[k].reshape(-1)[i]
                worst = max(worst, abs(num - ana) / max(1e-7, abs(num) + abs(ana)))
    finally:
        model.F = _F
    assert worst < 1e-4, worst


if __name__ == "__main__":
    test_gradients_match_finite_differences(); print("ok")
