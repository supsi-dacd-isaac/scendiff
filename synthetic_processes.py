import numpy as np
import pandas as pd


def sin_process(steps=24, n_scens=20, double=False):
    n_days = 30
    t = 24 * n_days
    t_index = pd.date_range('01-01-2020', '01-30-2020', t)
    signal = pd.DataFrame(np.sin(np.arange(t) * 2 * n_days * np.pi / t).reshape(-1, 1), index=t_index)
    target = pd.concat([signal.shift(l) for l in -np.arange(steps)], axis=1)
    target = target.loc[~target.isna().any(axis=1)]
    target.columns = ['target_{}'.format(i) for i in range(steps)]
    random_walk = np.cumsum(np.linspace(0.01, 0.2, steps) * np.random.randn(n_scens, steps), axis=1)
    scenarios = np.expand_dims(target.values, 2) + np.expand_dims(random_walk.T, 0)
    if double:
        test_scens = np.hstack([scenarios[0][:, :int(n_scens/2)], scenarios[12][:, :int(n_scens/2)]])
    else:
        test_scens = scenarios[0]
    return test_scens


def random_walk(steps=24, n_scens=20):
    random_walk = np.cumsum(np.linspace(0.01, 0.2, steps) * np.random.randn(n_scens, steps), axis=1).T
    return random_walk