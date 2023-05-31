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

np.random.seed(10)
semaphore = multiprocessing.Semaphore(int(cpu_count()))
savepath = 'results/combined'
use_parallel = True


def mapper(f, pars, *argv, **kwarg):
    poolsize = cpu_count() - 1
    pool = Pool(poolsize)
    res = pool.starmap_async(f, [(p, *argv, *list(kwarg.values())) for p in pars])
    a = res.get()
    pool.close()
    pool.join()
    return a


def plot_results(df, x, y, effect_1, effect_2=None, subplot_effect=None, figsize=(4.5, 3), basepath='results',
                 textpos=(0.05, 0.6), semilog=False, ax=None):

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
        if ax is None:
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
par_steps = 11


models = {'difft-c scenred': DiffTree(init='scenred', base_tree='scenred', loss='combined'),
          'difft scenred': DiffTree(init='scenred', base_tree='scenred'),
          'difft q-gen': DiffTree(init='quantiles', base_tree='quantiles'),
          'ng scenred': NeuralGas(init='scenred', base_tree='scenred'),
          'ng q-gen': NeuralGas(init='quantiles', base_tree='quantiles'),
          'scenred': ScenredTree(),
          'q-gen': QuantileTree()}


processes = {'sin': sin_process,
             'double sin': partial(sin_process, double=True),
             'random walk': random_walk}


def parfun(pars, processes, models, max_iterations=200, do_plot=False, keep_solutions=False, tol=1e-4):
    with semaphore:
        results = []
        s, n = pars
        t_00 = time()
        solutions = {}
        nodes_at_step = np.linspace(2, s, s).astype(int)
        for p_name, p in processes.items():
            test_scens = p(steps=s, n_scens=n)
            test_scens_ex_post = p(steps=s, n_scens=500)
            sol = {}
            for m_name, m in models.items():
                t_0 = time()
                if do_plot:
                    print('{},{}: {}'.format(s, n, m_name))
                #do_plot = True if m_name == 'dt scenred' and p_name == 'double sin' else False
                tree, _, _, _ = m.gen_tree(test_scens, k_max=max_iterations, do_plot=do_plot, tol=tol,
                                           nodes_at_step=nodes_at_step)
                t_1 = time()
                scen_dist, t_dist, reliability = m.evaluate_tree(test_scens)
                scen_dist_ex_post, t_dist_ex_post, reliability_ex_post = m.evaluate_tree(test_scens_ex_post)
                results.append(pd.DataFrame({'model': str(m_name), 'process': str(p_name),
                                             'n_scens': np.copy(n), 'steps': np.copy(s), 'time': t_1 - t_0,
                                             'scen dist': float(scen_dist),
                                             't dist': float(t_dist),
                                             'reliability': float(reliability),
                                             'scen dist test': float(scen_dist_ex_post),
                                             't dist test': float(t_dist_ex_post),
                                             'reliability test': float(reliability_ex_post)
                                             }, index=[0]))
                if keep_solutions:
                    sol[m_name] = (tree, test_scens)
            if keep_solutions:
                solutions[p_name] = sol
        plt.close('all')
        del test_scens
        print('{},{}: {:0.1f} min'.format(s, n, (time() - t_00)/60))
        print(pd.concat(results)[['model', 'process', 'n_scens', 'steps', 'time', 'scen dist', 't dist', 'reliability']])
        if keep_solutions:
            return pd.concat(results), solutions
        return pd.concat(results)


# -------------------------------------------------------------------------------------------------------------------- #
# ----------------------------------------  obtain results ----------------------------------------------------------- #
# -------------------------------------------------------------------------------------------------------------------- #

if glob(savepath) == []:
    mkdir(savepath)

scens_min, scens_max = 10, 210
steps_min, steps_max = 10, 110
par_steps = 11
scensspace = np.linspace(scens_min, scens_max, par_steps, dtype=int)
stepsspace = np.linspace(steps_min, steps_max, par_steps, dtype=int)
print('###### number of scenarios tested: ######')
print(scensspace)
print('###### number of steps tested: ######')
print(stepsspace)
parameters = {'n_scens': scensspace,
              'steps': stepsspace}
pars = list(product(parameters['steps'], parameters['n_scens']))
# find a separation of tested pars such that total numbers of nodes (times*scenarios) in each partition is ~same
n_tot = np.cumsum([t*n for t, n in pars])
splits = np.quantile(n_tot, np.hstack([0, 2**(-np.linspace(1, 0, 6))]), method='nearest')
cut_points = np.where(np.isin(n_tot, splits))[0]
bins = [(a, cut_points[i+1]) for i, a in enumerate(cut_points[:-1])]
print([np.sum([a[0]*a[1] for a in pars[b[0]:b[1]]]) for i, b in enumerate(bins)])

all_res = []
for b in bins:
    bin_pars = pars[b[0]:b[1]]
    print('obtaining these combinations of times and scenarios:')
    print(bin_pars)
    if use_parallel:
        results = mapper(parfun, bin_pars, processes, models, max_iterations=max_iterations)
    else:
        results = []
        for p in bin_pars:
            results.append(parfun(p, processes, models, max_iterations=max_iterations))
    all_res.append(pd.concat(results, axis=0))

results = pd.concat(all_res, axis=0)


if glob(savepath) == []:
    mkdir(savepath)

results.to_pickle(join(savepath, 'results_{}.pk'.format(strftime("%Y-%m-%d_%H"))))

# -------------------------------------------------------------------------------------------------------------------- #
# ----------------------------------------  plot   results ----------------------------------------------------------- #
# -------------------------------------------------------------------------------------------------------------------- #
#results = pd.read_pickle(join(savepath, 'results_{}.pk'.format("2023-05-30_19")))
get_nodes = lambda t: np.sum(np.linspace(2, t, t).astype(int)-1)
results.rename({'n_scens': r'$S$', 'scen dist': r'$d(\xi^{sc}, \xi^{tr})$', 't dist': r'$d_d(\xi^{sc}, \xi^{tr})$', 'steps':r'$T$', 'time': 't [s]'}, axis=1, inplace=True)
results.rename({'scen dist test': r'$d(\xi^{sc, te}, \xi^{tr})$', 't dist test': r'$d_d(\xi^{sc, te}, \xi^{tr})$'}, axis=1, inplace=True)
results.rename({'reliability': r'$\mathcal{R}$', 'reliability test': r'$\mathcal{R}_{te}$'}, axis=1, inplace=True)

results['$N$'] = results['$T$'].apply(get_nodes)
results[r'$\frac{V}{N}$'] = results['$N$'] / results['$S$']
results[r'$\frac{V}{NT}$'] = results['$N$'] / results['$S$']/results['$T$']


sb.set_style('darkgrid')
fig, ax = plt.subplots(3, 1, figsize=(4.5, 6))
plot_results(results, '$S$', r'$d(\xi^{sc}, \xi^{tr})$', 'model', figsize=(4.5, 2.5), ax=ax[0])
plot_results(results, '$S$', r'$d_d(\xi^{sc}, \xi^{tr})$', 'model', figsize=(4.5, 2.5), ax=ax[1])
ax[1].semilogy()
plot_results(results, '$N$', r'$\mathcal{R}$', 'model', figsize=(4.5, 2.5), ax=ax[2])
plt.subplots_adjust(hspace=0.3, left=0.2, right=0.95, top=0.95, bottom=0.1)
ax[2].semilogx()
[a.get_legend().remove() for a in ax.ravel()[1:]]
plt.savefig(join(savepath, 'results_{}.pdf'.format(strftime("%Y-%m-%d_%H"))), bbox_inches='tight')


fig, ax = plt.subplots(3, 1, figsize=(4.5, 6))
plot_results(results, '$S$', r'$d(\xi^{sc, te}, \xi^{tr})$', 'model', figsize=(4.5, 2.5), ax=ax[0])
plot_results(results, '$S$', r'$d_d(\xi^{sc, te}, \xi^{tr})$', 'model', figsize=(4.5, 2.5), ax=ax[1])
ax[1].semilogy()
plot_results(results, '$N$', r'$\mathcal{R}_{te}$', 'model', figsize=(4.5, 2.5), ax=ax[2])
plt.subplots_adjust(hspace=0.3, left=0.2, right=0.95, top=0.95, bottom=0.1)
ax[2].semilogx()
[a.get_legend().remove() for a in ax.ravel()[1:]]
plt.savefig(join(savepath, 'results_test{}.pdf'.format(strftime("%Y-%m-%d_%H"))), bbox_inches='tight')


plot_results(results, '$T$', 't [s]', 'model', figsize=(4.5, 2.5))

def rankplot(df, key=r'$d(\xi^{sc}, \xi^{tr})$', ax=None):
    rankmatrix = np.nan *np.zeros((len(df.model.unique()), len(df.model.unique())))
    for i, m in enumerate(df.model.unique()):
        for j, n in enumerate(df.model.unique()):
            rankmatrix[i, j] = np.round(np.sum(df[df.model == m][key].values < df[df.model == n][key].values) / df[df.model == m][key].shape[0], 2)
    print(df[df.model == m][key].shape[0])
    sb.set_style('whitegrid')
    if ax is None:
        fig, ax = plt.subplots(1, 1, figsize=(4.5, 3))
    sb.heatmap(rankmatrix, annot=True, ax=ax, cmap='Blues', cbar=False, fmt='g',xticklabels=
               df.model.unique(), yticklabels=df.model.unique())
    ax.set_xticklabels(df.model.unique(), rotation=45)
    ax.set_yticklabels(df.model.unique(), rotation=0)
    plt.subplots_adjust(bottom=0.25, left=0.25, wspace=0.3, hspace=0.3)
    plt.savefig(join(savepath, '{}_{}_rankplot.pdf'.format(key, strftime("%Y-%m-%d_%H"))))


fig, ax = plt.subplots(3, 1, figsize=(4.5, 4.5))
rankplot(results, ax=ax[0])
rankplot(results, key=r'$d_d(\xi^{sc}, \xi^{tr})$', ax=ax[1])
rankplot(results, key=r'$\mathcal{R}$', ax=ax[2])
plt.subplots_adjust(hspace=0.1, right=0.95, top=0.95)
[a.set_xticklabels([]) for a in ax.ravel()[:-1]]
plt.savefig(join(savepath, 'rankplot_{}.pdf'.format(strftime("%Y-%m-%d_%H"))), bbox_inches='tight')


fig, ax = plt.subplots(3, 1, figsize=(4.5, 4.5))
rankplot(results, key=r'$d(\xi^{sc, te}, \xi^{tr})$',ax=ax[0])
rankplot(results, key=r'$d_d(\xi^{sc, te}, \xi^{tr})$', ax=ax[1])
rankplot(results, key=r'$\mathcal{R}_{te}$', ax=ax[2])
plt.subplots_adjust(hspace=0.1, right=0.95, top=0.95)
[a.set_xticklabels([]) for a in ax.ravel()[:-1]]
plt.savefig(join(savepath, 'rankplot_test_{}.pdf'.format(strftime("%Y-%m-%d_%H"))), bbox_inches='tight')



# -------------------------------------------------------------------------------------------------------------------- #
# ---------------------------  obtain animations and final solutions ------------------------------------------------- #
# -------------------------------------------------------------------------------------------------------------------- #

models = {'q-gen': QuantileTree(),
          'scenred': ScenredTree(),
          'ng scenred': NeuralGas(init='scenred', base_tree='scenred',savepath='results/figs/neuralgas/'),
          'difft-c scenred': DiffTree(init='scenred', base_tree='scenred',savepath='results/figs/difftree/'),
          }

_, solutions = parfun((25, 100), processes, models, max_iterations=max_iterations, keep_solutions=True, do_plot=False)

sb.set_style('white')
fig, ax = plt.subplots(3, 4, figsize=(4.5, 4.5))

colors=sb.color_palette('viridis', n_colors=20)
c1 = colors[16]
c2 = colors[1]


plt.subplots_adjust(wspace=0,hspace=0)
for pm, a in zip(product(processes.keys(), models.keys()), ax.ravel()):
    p, m = pm
    models[m].plot_res(*solutions[p][m], ax=a, alpha=0.3, lwscens=0.7, linewidth=0.8, c1=c1, c2=c2)
[a.set_yticks([]) for a in ax[:, 1:].ravel()]
[a.set_xticks([]) for a in ax[:-1, :].ravel()]
plt.savefig(join(savepath, '{}_examples.pdf'.format(strftime("%Y-%m-%d_%H"))))


models = {'dt scenred': DiffTree(init='scenred', base_tree='scenred',savepath='results/figs/difftree/')}
processes = {'double sin': partial(sin_process, double=True),}

_, solutions = parfun((50, 100), processes, models, max_iterations=max_iterations, keep_solutions=True, do_plot=True, tol=1e-5)

