import numpy as np

from plogic.utils import sigmoid, soft_logic


def test_sigmoid_bounds_monotone():
    x = np.linspace(-5, 5, 201)
    y = sigmoid(x, beta=10.0)
    # Bounds
    assert np.all(y > 0.0)
    assert np.all(y < 1.0)
    # Monotone non-decreasing (allow tiny numerical slack)
    dy = np.diff(y)
    assert np.all(dy >= -1e-12)


def test_soft_logic_limits():
    thr = 0.5
    vals = np.array([0.0, 0.25, 0.5, 0.75, 1.0])
    beta = 1e3  # large slope -> approach hard threshold
    soft = soft_logic(vals, thr, beta)

    # Far from threshold, soft approximates hard
    assert np.isclose(float(soft[0]), 0.0, atol=1e-2)
    assert np.isclose(float(soft[-1]), 1.0, atol=1e-2)
    # Exactly at threshold -> 0.5 by symmetry
    assert np.isclose(float(soft[2]), 0.5, atol=1e-2)
