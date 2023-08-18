import matplotlib.pyplot as plt
import numpy as np
import networkx as nx
from itertools import count
from networkx.drawing.nx_agraph import graphviz_layout
from scendiff.tree_utils import retrieve_scenarios_indexes, replace_var
from time import time, strftime
import seaborn as sb
import matplotlib.pyplot as plt
from glob import glob
from os.path import join
from os import mkdir


def plot_graph(g, ax=None):
    '''
    Plot the networkx graph which encodes the scenario tree
    :param g: the networkx graph which encodes the scenario tree
    :return:
    '''

    # get unique groups
    if ax is None:
        fig = plt.figure()
        ax = plt.gca()
    groups = set(np.array(list(nx.get_node_attributes(g, 'v').values())))
    mapping = dict(zip(sorted(groups), count()))
    nodes = g.nodes()
    colors = [g.nodes[n]['v'] for n in nodes]
    p = np.array(list(nx.get_node_attributes(g, 'p').values()))

    # drawing nodes and edges separately so we can capture collection for colobar
    pos = graphviz_layout(g, prog='dot')
    # nx.draw_networkx(g,pos,with_labels=True)
    ec = nx.draw_networkx_edges(g, pos, alpha=0.2)
    nc = nx.draw_networkx_nodes(g, pos, nodelist=nodes, node_color=colors, node_size=100 * p, cmap=plt.cm.magma)
    ax.set_xticks([])
    ax.set_yticks([])
    cb = plt.colorbar(nc)
    return ax, cb


def plot_from_graph(g, lines=None, ax=None, color=None, prob=False, **kwargs):
    s_idx, leaves = retrieve_scenarios_indexes(g)
    values = np.array(list(nx.get_node_attributes(g, 'v').values()))
    times = np.array(list(nx.get_node_attributes(g, 't').values()))
    p = np.array(list(nx.get_node_attributes(g, 'p').values()))
    v = np.array(list(nx.get_node_attributes(g, 'v').values()))
    t = np.array(list(nx.get_node_attributes(g, 't').values()))
    cmap = plt.get_cmap('Set1')
    if color is None:
        color = cmap(np.arange(3))[1, :]
    if ax is None:
        fig, ax = plt.subplots(1, 1)
    if lines is not None:
        for s, l in zip(np.arange(s_idx.shape[1]), lines):
            l.set_data(times[s_idx[:, s]], values[s_idx[:, s]])
    else:
        lines = []
        for s in np.arange(s_idx.shape[1]):
            l = ax.plot(values[s_idx[:, s]], color=color, **kwargs)
            lines.append(l[0])
    if prob:
        ax.scatter(t, v, s=100 * p, c=color, alpha=0.5)
    return lines


def plot_scen(S_s, y=None):
    '''

    :param S_s:
    :return:
    '''
    if S_s.shape[2] == 2:
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        for i in np.arange(S_s.shape[1]):
            ax.plot(np.arange(S_s.shape[0]), np.squeeze(S_s[:, i, 0]), np.squeeze(S_s[:, i, 1]), color='k', alpha=0.1)
        if y is not None:
            ax.plot(np.arange(S_s.shape[0]), y[:, 0], y[:, 1], linewidth=1.5)
    elif S_s.shape[2] == 1:
        fig, ax = plt.subplots(1)
        plt.plot(np.squeeze(S_s), color='k', alpha=0.1)
        if y is not None:
            plt.plot(y, linewidth=1.5)
    else:
        assert S_s.shape[2] > 2, 'Error: cannot visualize more than bivariate scenarios'
    return fig, ax


def plot_vars(g, v, ax=None, color=None, dyn_offset=False, **kwargs):
    if ax is None:
        fig, ax = plt.subplots(1, 1)
    replace_var(g, v, dyn_offset)
    ax = plot_from_graph(g, ax=ax, color=color, **kwargs)
    return ax


def plot_results(df, x, y, effect_1, effect_2=None, subplot_effect=None, figsize=(4.5, 3), basepath='results',
                 textpos=(0.05, 0.6), semilog=False, ax=None, legend=True, linewidth=0.8):

    x_plots = 1
    if subplot_effect is not None:
        col_effect_names = df[subplot_effect].value_counts().index
        figsize = (figsize[0], figsize[1] * len(col_effect_names))
        x_plots = len(col_effect_names)
        sb.set_style('darkgrid')
        fig, ax = plt.subplots(x_plots, 1, figsize=figsize)
        for i, se in enumerate(col_effect_names):
            sb.lineplot(x=x, y=y, hue=effect_1, style=effect_2, data=df[df[subplot_effect] == se], ax=ax[i], legend=legend, linewidth=linewidth)
            ax[i].text(*textpos, '{}={}'.format(subplot_effect, se), transform=ax[i].transAxes, fontsize=9)
    else:
        if ax is None:
            fig, ax = plt.subplots(x_plots, 1, figsize=figsize)
        sb.lineplot(x=x, y=y, hue=effect_1, style=effect_2, data=df, ax=ax, legend=legend, linewidth=linewidth)

    if glob(basepath) == []:
        mkdir(basepath)

    if legend:
        [a.legend(fontsize='x-small', ncols=2) for a in np.atleast_1d(ax)]

    if semilog:
       [a.semilogy() for a in plt.gcf().axes]
    # plt.savefig(join(basepath, '{}_{}_{}_{}_{}.pdf'.format(strftime("%Y-%m-%d_%H"), x, y,
    # effect_1, effect_2, subplot_effect)))


def rankplot(df, key=r'$\tilde{d}(\xi^{sc}, \xi^{tr})$', ax=None, savepath=None, **kwargs):
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