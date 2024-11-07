import numpy as np

def to_unit_cube(x, lb, ub):
    assert np.all(lb <= ub) and lb.ndim == 1 and ub.ndim == 1 \
        and x.ndim == 2, (lb.ndim, ub.ndim, x.ndim)
    xx = (x - lb) / (ub - lb)
    return xx


def from_unit_cube(x, lb, ub):
    assert np.all(lb <= ub) and lb.ndim == 1 and ub.ndim == 1 \
        and x.ndim == 2, (lb.ndim, ub.ndim, x.ndim)
    xx = x * (ub - lb) + lb
    return xx


