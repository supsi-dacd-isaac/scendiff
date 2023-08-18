import numpy as np
import networkx as nx


def retrieve_scenarios_indexes(g, dyn_offset=False):
    """
    Retrieve the indexes of the scenarios in the tree
    :param g: a networkx graph (a tree)
    :param dyn_offset: whether the control uses more than one observations for the first step to model initial variance
    :return: scen_idxs_hist: a matrix of size (T, n_s) where T is the number of time steps and n_s the number of
             scenarios
             leaves: a vector of size n_s containing the indexes of the leaves
    """
    n_n = len(g.nodes)
    node_set = np.linspace(0, n_n - 1, n_n, dtype=int)
    all_t = np.array(list(nx.get_node_attributes(g, 't').values()))
    t = np.unique(all_t)
    leaves = np.array([n for n in node_set[all_t == np.max(t)]])
    scen_idxs_hist = np.zeros((max(t) + 1, len(leaves)), dtype=int)
    for s in np.arange(len(leaves)):
        scen_idxs = np.sort(np.array(list(nx.ancestors(g, leaves[s]))))
        scen_idxs = np.asanyarray(np.insert(scen_idxs, len(scen_idxs), leaves[s], 0), int)
        scen_idxs_hist[:, s] = scen_idxs
    if dyn_offset:
        scen_idxs_hist = scen_idxs_hist[1:, :]
        scen_idxs_hist -= 1
        leaves -= 1
    return scen_idxs_hist, leaves


def replace_var(tree, variable, dyn_offset=False):
    if dyn_offset:
        variable = np.hstack([np.nan, variable])
    nx.set_node_attributes(tree, {i: v for i, v in enumerate(variable)}, name='v')


def superimpose_signal_to_scens(x, w, perms):
    return w + x[:, perms]


def superimpose_signal_to_tree(x, tree):
    times = np.array(list(nx.get_node_attributes(tree, 't').values()))
    tree_vals = np.array(list(nx.get_node_attributes(tree, 'v').values()))
    for t in np.unique(times):
        tree_vals[times == t] += np.atleast_1d(x[t])
    replace_var(tree, tree_vals)
    return tree


def get_nodes_per_time_from_tree(g):
    times = np.array(list(nx.get_node_attributes(g, 't').values()))
    return [np.sum(times==t) for t in np.unique(times)]


def get_scenarios_from_tree(tree):
    tree_idxs, leaves = retrieve_scenarios_indexes(tree)
    tree_vals = np.hstack(list(dict(tree.nodes('v')).values()))
    tree_scens = np.vstack([tree_vals[idx] for idx in tree_idxs])
    return tree_scens, tree_vals, tree_idxs