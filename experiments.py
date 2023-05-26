import multiprocessing

import pandas as pd
import numpy as np
from scendiff.trees import NeuralGas, DiffTree, ScenredTree, QuantileTree
from scendiff.plot_utils import plot_from_graph
from synthetic_processes import sin_process, random_walk
from time import time, strftime
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
use_parallel = False


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


def plot_results(df, x, y, effect_1, effect_2=None, subplot_effect=None, figsize=(4.5, 3), basepath='results',
                 textpos=(0.05, 0.6), semilog=False):

    x_plots = 1
    if subplot_effect is not None:
        col_effect_names = df[subplot_effect].value_counts().index
        figsize = (figsize[0], figsize[1] * len(col_effect_names))
        x_plots = len(col_effect_names)
        sb.set_style('darkgrid')
        fig, ax = plt.subplots(x_plots, 1, figsize=figsize)
        for i, se in enumerate(col_effect_names):
            sb.lineplot(x=x, y=y, hue=effect_1, style=effect_2, data=df[df[subplot_effect] == se], ax=ax[i])
            ax[i].text(*textpos, '{}={}'.format(subplot_effect, se), transform=ax[i].transAxes, fontsize=9)
        plt.subplots_adjust(bottom=0.08, left=0.15, right=0.99, hspace=0.02, top=0.98)
    else:
        fig, ax = plt.subplots(x_plots, 1, figsize=figsize)
        sb.lineplot(x=x, y=y, hue=effect_1, style=effect_2, data=df, ax=ax)
        plt.subplots_adjust(bottom=0.2, left=0.15, wspace=0.3, hspace=0.3, top=0.98)

    if glob(basepath) == []:
        mkdir(basepath)

    [a.legend(fontsize='x-small', ncols=2) for a in np.atleast_1d(ax)]
    if semilog:
       [a.semilogy() for a in plt.gcf().axes]
    plt.savefig(join(basepath, '{}_{}_{}_{}_{}.pdf'.format(strftime("%Y-%m-%d_%H"), x, y, effect_1, effect_2, subplot_effect)))


max_iterations = 300
scens_min, scens_max = 10, 100
steps_min, steps_max = 10, 100
par_steps = 10


models = {'dt scenred': DiffTree(init='scenred', base_tree='scenred'),
          'dt q-gen': DiffTree(init='quantiles', base_tree='quantiles'),
          'ng scenred': NeuralGas(init='scenred', base_tree='scenred'),
          'ng q-gen': NeuralGas(init='quantiles', base_tree='quantiles'),
          'scenred': ScenredTree(),
          'q-gen': QuantileTree()}


processes = {'sin': sin_process,
             'double sin': partial(sin_process, double=True),
             'random walk': random_walk}

parameters = {'n_scens': np.linspace(scens_min, scens_max, par_steps, dtype=int),
              'steps': np.linspace(steps_min, steps_max, par_steps, dtype=int)}


pars = list(product(parameters['steps'], parameters['n_scens']))


def parfun(pars, processes, models, max_iterations=200, do_plot=False, keep_solutions=False):
    with semaphore:
        results = []
        s, n = pars
        t_00 = time()
        solutions = {}
        for p_name, p in processes.items():
            test_scens = p(steps=s, n_scens=n)
            sol = {}
            for m_name, m in models.items():
                t_0 = time()
                if do_plot:
                    print('{},{}: {}'.format(s, n, m_name))
                tree, _, _, _ = m.gen_tree(test_scens, k_max=max_iterations, do_plot=do_plot, tol=1e-4)
                t_1 = time()
                loss, reliability = m.evaluate_tree(test_scens)
                results.append(pd.DataFrame({'model': str(m_name), 'process': str(p_name),
                                             'n_scens': np.copy(n), 'steps': np.copy(s), 'time': t_1 - t_0,
                                             'loss': float(loss),
                                             'reliability':float(reliability)}, index=[0]))
                if keep_solutions:
                    sol[m_name] = (tree, test_scens)
            if keep_solutions:
                solutions[p_name] = sol
        plt.close('all')
        del test_scens
        print('{},{}: {:0.1f} min'.format(s, n, (time() - t_00)/60))
        print(pd.concat(results))
        if keep_solutions:
            return pd.concat(results), solutions
        return pd.concat(results)


# -------------------------------------------------------------------------------------------------------------------- #
# ----------------------------------------  obtain results ----------------------------------------------------------- #
# -------------------------------------------------------------------------------------------------------------------- #

if use_parallel:
    results = mapper(parfun, pars, processes, models, max_iterations=max_iterations)
else:
    results = []
    for p in pars:
        results.append(parfun(p, processes, models, max_iterations=max_iterations))

    results = pd.concat(results, axis=0)

if glob(savepath) == []:
    mkdir(savepath)

results.to_pickle(join(savepath, 'results_{}.pk'.format(strftime("%Y-%m-%d_%H"))))

# -------------------------------------------------------------------------------------------------------------------- #
# ----------------------------------------  plot   results ----------------------------------------------------------- #
# -------------------------------------------------------------------------------------------------------------------- #

results.rename({'n_scens': r'$N$', 'loss': r'$d(\xi^{sc}, \xi^{tr})$', 'steps':r'$T$', 'time': 't [s]'}, axis=1, inplace=True)

plot_results(results, '$N$', r'$d(\xi^{sc}, \xi^{tr})$', 'model', subplot_effect='process', figsize=(4.5, 2.5))
plot_results(results, '$T$',  r'$d(\xi^{sc}, \xi^{tr})$', 'model', subplot_effect='process', figsize=(4.5, 2.5))
plot_results(results, '$T$',  't [s]', 'model', subplot_effect='process', figsize=(4.5, 2.5), textpos=(0.05, 0.9), semilog=True)
plot_results(results, '$T$',  't [s]', 'model', figsize=(4.5, 2.5), semilog=True)
plot_results(results, '$N$', r'$d(\xi^{sc}, \xi^{tr})$', 'model', figsize=(4.5, 2.5))

def rankplot(df):
    rankmatrix = np.nan *np.zeros((len(df.model.unique()), len(df.model.unique())))
    for i, m in enumerate(df.model.unique()):
        for j, n in enumerate(df.model.unique()):
            if j>i:
                rankmatrix[i, j] = np.sum(df[df.model == m][r'$d(\xi^{sc}, \xi^{tr})$'].values < df[df.model == n][r'$d(\xi^{sc}, \xi^{tr})$'].values)

    sb.set_style('whitegrid')
    fig, ax = plt.subplots(1, 1, figsize=(4.5, 3))
    sb.heatmap(rankmatrix, annot=True, ax=ax, cmap='Blues', cbar=False, fmt='g')
    ax.set_xticklabels(df.model.unique(), rotation=45)
    ax.set_yticklabels(df.model.unique(), rotation=0)
    plt.subplots_adjust(bottom=0.25, left=0.25, wspace=0.3, hspace=0.3)
    plt.savefig(join(savepath, '{}_rankplot.pdf'.format(strftime("%Y-%m-%d_%H"))))

rankplot(results)

# -------------------------------------------------------------------------------------------------------------------- #
# ---------------------------  obtain animations and final solutions ------------------------------------------------- #
# -------------------------------------------------------------------------------------------------------------------- #

models = {'q-gen': QuantileTree(),
          'scenred': ScenredTree(),
          'ng scenred': NeuralGas(init='scenred', base_tree='scenred',savepath='results/figs/neuralgas/'),
          'dt scenred': DiffTree(init='scenred', base_tree='scenred',savepath='results/figs/difftree/'),
          }

_, solutions = parfun((25, 100), processes, models, max_iterations=max_iterations, keep_solutions=True, do_plot=False)

fig, ax = plt.subplots(3, 4, figsize=(4.5, 4.5))
plt.subplots_adjust(wspace=0,hspace=0)
for pm, a in zip(product(processes.keys(), models.keys()), ax.ravel()):
    p, m = pm
    models[m].plot_res(*solutions[p][m], ax=a, alpha=0.3, linewidth=0.5)
[a.set_yticks([]) for a in ax[:, 1:].ravel()]
[a.set_xticks([]) for a in ax[:-1, :].ravel()]
plt.savefig(join(savepath, '{}_examples.pdf'.format(strftime("%Y-%m-%d_%H"))))