import multiprocessing

import pandas as pd
import numpy as np
from scendiff.trees import NeuralGas, DiffTree, ScenredTree, QuantileTree
from scendiff.plot_utils import plot_from_graph
from synthetic_processes import sin_process, random_walk
from time import time
import seaborn as sb
import matplotlib.pyplot as plt
from glob import glob
from os.path import join
from os import mkdir
from multiprocessing import Pool, cpu_count
from itertools import product
from functools import partial

semaphore = multiprocessing.Semaphore(int(cpu_count()))
savepath = 'results'

def mapper(f, pars, *argv, **kwarg):
    poolsize = cpu_count() - 1
    pool = Pool(poolsize)
    #for p in [(p, *argv, *list(kwarg.values())) for p in pars]:
    #    res = pool.apply_async(f,  p)
    # parallel process over shared object
    #pool = multiprocessing.Semaphore(cpu_count() - 1)
    res = pool.starmap_async(f, [(p, *argv, *list(kwarg.values())) for p in pars])
    a = res.get()
    pool.close()
    pool.join()
    return a


def plot_results(df, x, y, effect_1, effect_2=None, subplot_effect=None, figsize=(4.5, 3), basepath='results'):

    x_plots = 1
    if subplot_effect is not None:
        col_effect_names = df[subplot_effect].value_counts().index
        figsize = (figsize[0], figsize[1] * len(col_effect_names))
        x_plots = len(col_effect_names)
        fig, ax = plt.subplots(x_plots, 1, figsize=figsize)
        for i, se in enumerate(col_effect_names):
            sb.lineplot(x=x, y=y, hue=effect_1, style=effect_2, data=df[df[subplot_effect] == se], ax=ax[i])
            ax[i].set_title('{}={}'.format(subplot_effect, se))
        plt.subplots_adjust(bottom=0.15, left=0.15, hspace=0.4)
    else:
        fig, ax = plt.subplots(x_plots, 1, figsize=figsize)
        sb.lineplot(x=x, y=y, hue=effect_1, style=effect_2, data=df, ax=ax)
        plt.subplots_adjust(bottom=0.15, left=0.15, wspace=0.3, hspace=0.3)

    if glob(basepath) == []:
        mkdir(basepath)

    [a.legend(fontsize='x-small', ncols=2) for a in ax]

    plt.savefig(join(basepath, '{}_{}_{}_{}.pdf'.format(x, y, effect_1, effect_2, subplot_effect)))


max_iterations = 300
scens_min, scens_max = 10, 100
steps_min, steps_max = 10, 100
par_steps = 10


models = {'dt scenred': DiffTree(init='scenred', base_tree='scenred'),
          'dt quant': DiffTree(init='quantiles', base_tree='quantiles'),
          'ng scenred': NeuralGas(init='scenred', base_tree='scenred'),
          'ng quant': NeuralGas(init='quantiles', base_tree='quantiles'),
          'scenred': ScenredTree(),
          'qt': QuantileTree()}


processes = {'sin': sin_process,
             'double sin': partial(sin_process, double=True),
             'random walk': random_walk}

parameters = {'n_scens': np.linspace(scens_min, scens_max, par_steps, dtype=int),
              'steps': np.linspace(steps_min, steps_max, par_steps, dtype=int)}


pars = list(product(parameters['steps'], parameters['n_scens']))


def parfun(pars, processes, models, max_iterations=200):
    with semaphore:
        results = []
        s, n = pars
        t_00 = time()
        for p_name, p in processes.items():
            test_scens = p(steps=s, n_scens=n)
            for m_name, m in models.items():
                if m_name in ['dt scenred', 'dt quant'] and p_name in ['double sin']:
                    do_plot = True
                else:
                    do_plot = False
                t_0 = time()
                tree, _, _, _ = m.gen_tree(test_scens, k_max=max_iterations, do_plot=do_plot,tol=1e-4)
                t_1 = time()
                results.append(pd.DataFrame({'model': str(m_name), 'process': str(p_name),
                                             'n_scens': np.copy(n), 'steps': np.copy(s), 'time': t_1 - t_0,
                                             'loss': float(m.losses[-1])}, index=[0]))
        plt.close('all')
        del test_scens
        print('{},{}: {:0.1f} min'.format(s, n, (time() - t_00)/60))
        print(pd.concat(results))
        return pd.concat(results)


#results = mapper(parfun, pars, processes, models, max_iterations=max_iterations)

results = []
for p in pars:
    results.append(parfun(p, processes, models, max_iterations=max_iterations))

results = pd.concat(results, axis=0)

if glob(savepath) == []:
    mkdir(savepath)
results.to_pickle(join(savepath, 'results.pk'))

plot_results(results, 'n_scens', 'loss', 'model', subplot_effect='process')
plot_results(results, 'steps', 'loss', 'model', subplot_effect='process')

plot_results(results, 'n_scens', 'time', 'process','steps', subplot_effect='model')
