import multiprocessing
import pandas as pd
import numpy as np
from scendiff.trees import NeuralGas, DiffTree, ScenredTree, QuantileTree
from scendiff.plot_utils import plot_results, rankplot
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

# ---------------------------------------------------------------------------------------------------------------------
# ------------------------------------ Set parameters -----------------------------------------------------------------
# ---------------------------------------------------------------------------------------------------------------------

np.random.seed(10)
semaphore = multiprocessing.Semaphore(int(cpu_count()))
savepath = 'results'
use_parallel = True
max_iterations = 300
par_steps = 11


# parallelization function
def mapper(f, pars, *argv, **kwarg):
    poolsize = cpu_count() - 1
    pool = Pool(poolsize)
    res = pool.starmap_async(f, [(p, *argv, *list(kwarg.values())) for p in pars])
    a = res.get()
    pool.close()
    pool.join()
    return a

# ---------------------------------------------------------------------------------------------------------------------
# ------------------------------------ Set tree models ----------------------------------------------------------------
# ---------------------------------------------------------------------------------------------------------------------


models = {'difft-c scenred': DiffTree(init_vals_method='scenred', init_tree_method='scenred', loss='combined'),
          'difft scenred': DiffTree(init_vals_method='scenred', init_tree_method='scenred'),
          'difft q-gen': DiffTree(init_vals_method='quantiles', init_tree_method='quantiles'),
          'ng scenred': NeuralGas(init_vals_method='scenred', init_tree_method='scenred'),
          'ng q-gen': NeuralGas(init_vals_method='quantiles', init_tree_method='quantiles'),
          'scenred': ScenredTree(),
          'q-gen': QuantileTree()}


processes = {'sin': sin_process,
             'double sin': partial(sin_process, double=True),
             'random walk': random_walk}


# ---------------------------------------------------------------------------------------------------------------------
# ------------------------- Define fitiing and evaluating function ----------------------------------------------------
# ---------------------------------------------------------------------------------------------------------------------
def parfun(pars, processes, models, max_iterations=200, do_plot=False, keep_solutions=False, tol=1e-4):
    with semaphore:
        # get number of steps and number of scenarios
        s, n = pars
        results = []
        t_00 = time()
        solutions = {}

        # number of nodes at each step
        nodes_at_step = np.linspace(2, s, s).astype(int)

        # for all the tested processes
        for p_name, p in processes.items():
            # define the process based on the number of steps and number of scenarios
            test_scens = p(steps=s, n_scens=n)
            test_scens_ex_post = p(steps=s, n_scens=500)
            sol = {}
            # for all the tested models
            for m_name, m in models.items():
                if do_plot:
                    print('{},{}: {}'.format(s, n, m_name))

                # fit the tree
                t_0 = time()
                tree, _, _, _ = m.gen_tree(test_scens, k_max=max_iterations, do_plot=do_plot, tol=tol,
                                           nodes_at_step=nodes_at_step)
                t_1 = time()

                # evaluate the tree on scenarios
                scen_dist, t_dist, reliability = m.evaluate_tree(test_scens)

                # evaluate the tree on new scenarios not seen during training
                scen_dist_ex_post, t_dist_ex_post, reliability_ex_post = m.evaluate_tree(test_scens_ex_post)

                # put results together
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

# define the number of steps and number of scenarios to test
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

# find a separation of tested pars such that total numbers of nodes (times*scenarios) in each partition is ~same. This
# is done to avoid flooding RAM with too many scenarios.

n_tot = np.cumsum([t*n for t, n in pars])
splits = np.quantile(n_tot, np.hstack([0, 2**(-np.linspace(1, 0, 6))]), method='nearest')
cut_points = np.where(np.isin(n_tot, splits))[0]
bins = [(a, cut_points[i+1]) for i, a in enumerate(cut_points[:-1])]

# run the fitting and evaluation function in parallel or sequentially
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

get_nodes = lambda t: np.sum(np.linspace(2, t, t).astype(int)-1)
results.rename({'n_scens': r'$S$', 'scen dist': r'$\tilde{d}(\xi^{sc}, \xi^{tr})$', 't dist': r'$d_d(\xi^{sc}, \xi^{tr})$', 'steps':r'$T$', 'time': 't [s]'}, axis=1, inplace=True)
results.rename({'scen dist test': r'$\tilde{d}(\xi^{sc, te}, \xi^{tr})$', 't dist test': r'$d_d(\xi^{sc, te}, \xi^{tr})$'}, axis=1, inplace=True)
results.rename({'reliability': r'$\mathcal{R}$', 'reliability test': r'$\mathcal{R}_{te}$'}, axis=1, inplace=True)

results['$N$'] = results['$T$'].apply(get_nodes)
results[r'$\frac{V}{N}$'] = results['$N$'] / results['$S$']
results[r'$\frac{V}{NT}$'] = results['$N$'] / results['$S$']/results['$T$']

# generate normalized results for some keys
n_keys = [r'$\tilde{d}(\xi^{sc}, \xi^{tr})$', r'$d_d(\xi^{sc}, \xi^{tr})$', r'$\tilde{d}(\xi^{sc, te}, \xi^{tr})$', r'$d_d(\xi^{sc, te}, \xi^{tr})$', r'$\mathcal{R}$', r'$\mathcal{R}_{te}$']
n_keys_new = [r'$\tilde{d}_{norm}(\xi^{sc}, \xi^{tr})$', r'$d_{d, norm}(\xi^{sc}, \xi^{tr})$',
              r'$\tilde{d}_{norm}(\xi^{sc, te}, \xi^{tr})$', r'$d_{d, norm}(\xi^{sc, te}, \xi^{tr})$', r'$\mathcal{R}_{norm}$',
              r'$\mathcal{R}_{norm, te}$']
model_keys = models.keys()
for k, k_new in zip(n_keys, n_keys_new):
    for m in model_keys:
        results.loc[results['model'] == m, k_new] = results.loc[results['model'] == m, k].values / results.loc[results['model'] == 'scenred', k].values

sb.set_style('darkgrid')
fig, ax = plt.subplots(3, 2, figsize=(6, 6), layout='compressed')
plot_results(results, '$S$', r'$\tilde{d}(\xi^{sc}, \xi^{tr})$', 'model', figsize=(4.5, 2.5), ax=ax[0, 0])
plot_results(results, '$S$', r'$d_d(\xi^{sc}, \xi^{tr})$', 'model', figsize=(4.5, 2.5), ax=ax[1, 0], legend=False)
ax[1, 0].semilogy()
plot_results(results, '$N$', r'$\mathcal{R}$', 'model', figsize=(4.5, 2.5), ax=ax[2, 0], legend=False)
ax[2, 0].semilogx()

results.loc[results['model'] == 'scenred', 'model'] = 'scenred (ref.)'
plot_results(results, '$S$', r'$\tilde{d}_{norm}(\xi^{sc}, \xi^{tr})$', 'model', figsize=(4.5, 2.5), ax=ax[0, 1])
plot_results(results, '$S$', r'$d_{d, norm}(\xi^{sc}, \xi^{tr})$', 'model', figsize=(4.5, 2.5), ax=ax[1, 1], legend=False)
plot_results(results, '$N$', r'$\mathcal{R}_{norm}$', 'model', figsize=(4.5, 2.5), ax=ax[2, 1], legend=False)
[a.semilogy() for a in ax[:, 1].ravel()]
ax[2, 1].semilogx()
ax[0, 0].set_title('scores')
ax[0, 1].set_title('normalized scores')

plt.savefig(join(savepath, 'results_combo_{}.pdf'.format(strftime("%Y-%m-%d_%H"))), bbox_inches='tight')


fig, ax = plt.subplots(3, 2, figsize=(6, 6), layout='compressed')
results.loc[results['model'] == 'scenred (ref.)', 'model'] = 'scenred'
plot_results(results, '$S$', r'$\tilde{d}(\xi^{sc, te}, \xi^{tr})$', 'model', figsize=(4.5, 2.5), ax=ax[0, 0])
plot_results(results, '$S$', r'$d_d(\xi^{sc, te}, \xi^{tr})$', 'model', figsize=(4.5, 2.5), ax=ax[1, 0], legend=False)
ax[1, 0].semilogy()
plot_results(results, '$N$', r'$\mathcal{R}_{te}$', 'model', figsize=(4.5, 2.5), ax=ax[2, 0], legend=False)
ax[2, 0].semilogx()

results.loc[results['model'] == 'scenred', 'model'] = 'scenred (ref.)'
plot_results(results, '$S$', r'$\tilde{d}_{norm}(\xi^{sc, te}, \xi^{tr})$', 'model', figsize=(4.5, 2.5), ax=ax[0, 1])
plot_results(results, '$S$', r'$d_{d, norm}(\xi^{sc, te}, \xi^{tr})$', 'model', figsize=(4.5, 2.5), ax=ax[1, 1], legend=False)
plot_results(results, '$N$', r'$\mathcal{R}_{norm, te}$', 'model', figsize=(4.5, 2.5), ax=ax[2, 1], legend=False)
[a.semilogy() for a in ax[:, 1].ravel()]
ax[2, 1].semilogx()
ax[0, 0].set_title('scores')
ax[0, 1].set_title('normalized scores')

plt.savefig(join(savepath, 'results_combo_test_{}.pdf'.format(strftime("%Y-%m-%d_%H"))), bbox_inches='tight')


plot_results(results, '$T$', 't [s]', 'model', figsize=(4.5, 2.5))


fig, ax = plt.subplots(3, 1, figsize=(4.5, 4.5))
rankplot(results, ax=ax[0], savepath=savepath)
rankplot(results, key=r'$d_d(\xi^{sc}, \xi^{tr})$', ax=ax[1], savepath=savepath)
rankplot(results, key=r'$\mathcal{R}$', ax=ax[2], savepath=savepath)
plt.subplots_adjust(hspace=0.1, right=0.95, top=0.95)
[a.set_xticklabels([]) for a in ax.ravel()[:-1]]
plt.savefig(join(savepath, 'rankplot_{}.pdf'.format(strftime("%Y-%m-%d_%H"))), bbox_inches='tight')


fig, ax = plt.subplots(3, 1, figsize=(4.5, 4.5))
rankplot(results, key=r'$\tilde{d}(\xi^{sc, te}, \xi^{tr})$',ax=ax[0], savepath=savepath)
rankplot(results, key=r'$d_d(\xi^{sc, te}, \xi^{tr})$', ax=ax[1], savepath=savepath)
rankplot(results, key=r'$\mathcal{R}_{te}$', ax=ax[2], savepath=savepath)
plt.subplots_adjust(hspace=0.1, right=0.95, top=0.95)
[a.set_xticklabels([]) for a in ax.ravel()[:-1]]
plt.savefig(join(savepath, 'rankplot_test_{}.pdf'.format(strftime("%Y-%m-%d_%H"))), bbox_inches='tight')


# -------------------------------------------------------------------------------------------------------------------- #
# ---------------------------  obtain animations and final solutions ------------------------------------------------- #
# -------------------------------------------------------------------------------------------------------------------- #

models = {'q-gen': QuantileTree(),
          'scenred': ScenredTree(),
          'ng scenred': NeuralGas(init_vals_method='scenred', init_tree_method='scenred',savepath='results/figs/neuralgas/'),
          'difft-c scenred': DiffTree(init_vals_method='scenred', init_tree_method='scenred',savepath='results/figs/difftree/'),
          }

_, solutions = parfun((25, 100), processes, models, max_iterations=max_iterations, keep_solutions=True, do_plot=False)


sb.set_style('white')
fig, ax = plt.subplots(3, 4, figsize=(4.5, 4.5))
colors = sb.color_palette('viridis', n_colors=100)
c1 = colors[84]
c2 = colors[4]

plt.subplots_adjust(wspace=0,hspace=0)
for pm, a in zip(product(processes.keys(), models.keys()), ax.ravel()):
    p, m = pm
    a = models[m].plot_res(*solutions[p][m], ax=a, alpha=0.3, lwscens=0.7, linewidth=0.8, c1=c1, c2=c2)
[a.set_xlabel('time step', size=8) for a in ax[-1, :].ravel()]

for a in ax.ravel():
    for spine in a.spines.values():
        spine.set_visible(True)

ax[0, 0].set_ylabel(r'$\xi^{sin}$')
ax[1, 0].set_ylabel(r'$\xi^{dsin}$')
ax[2, 0].set_ylabel(r'$\xi^{rw}$')
ax[0, 0].set_title('q-gen', size=8)
ax[0, 1].set_title('scenred', size=8)
ax[0, 2].set_title('ng scenred', size=8)
ax[0, 3].set_title('difft scenred', size=8)

plt.savefig(join(savepath, '{}_examples.pdf'.format(strftime("%Y-%m-%d_%H"))))


models = {'dt scenred': DiffTree(init_vals_method='scenred', init_tree_method='scenred',savepath='results/figs/difftree/')}
processes = {'double sin': partial(sin_process, double=True),}

_, solutions = parfun((50, 200), processes, models, max_iterations=max_iterations, keep_solutions=True, do_plot=True, tol=1e-5)


