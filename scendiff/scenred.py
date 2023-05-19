import numpy as np
from scipy.spatial.distance import pdist, squareform, cdist
import networkx as nx
import logging


def get_dist(X, metric):
    D = squareform(pdist(X, metric))
    return D


def scenred(samples_0, tol=10, metric='cityblock', nodes=None, logger=None):
    """
    Build a scenario tree reducing N observed samplings from a multivariate
    distribution. The algorithm is described in "Nicole Growe-Kuska,
    Holger Heitsch and Werner Romisch - Scenario Reduction Scenario Tree
    Construction for Power Management Problems".
     Inputs:
             samples: n cell array, where n is the number of signals observed,
             each cell of dimension (n_steps, n_scen), where n_steps is the number of observed
             timesteps and n_scen is the original number of observed scenarios.
             The scenarios in the different cells are supposed to be
             statistically dependent (sampled from historical data or from
             a copula)
             metric: in {'euclidean','cityblock'}, used for
                     Kantorovich-Rubinstein metric
             kwargs: {'nodes','tol'} methods for building the tree
                       -nodes vector with length n_steps, increasingly monotone,
                       with the specified number of nodes per timestep
                       -tol: tolerance on Dist/Dist_0, where Dist is the
                             distance after aggregating scenarios, and Dist_0
                             is the distance between all the scenarios and a
                             tree with only one scenario. In any case, at
                             least one scenario is selected.
     :return:
              S: n_steps*n_scen*n matrix, containing all the path in the scenario
              tree
              P: n_steps*n_scen matrix, containing evolviong probabilities for each
              scenario
              J: n_steps*n_scen matrix, indicating non-zero entries of P (if a
              scenario is present at a given timestep)
              Me: n_steps*n_scen*n_scen matrix, containing which scenario is linked
              with which at time t
    """

    if logger is None:
        logger = logging.getLogger()
        logging.basicConfig(format='%(asctime)-15s::%(levelname)s::%(funcName)s::%(message)s')
        logger.setLevel(logging.WARNING)

    samples = np.copy(samples_0)

    # expand dimensions if needed (univariate case)
    if len(samples.shape) == 2:
        samples = np.expand_dims(samples, -1)

    n_steps, n_scen, n_var = samples.shape
    default_nodes = np.ones((n_steps, 1))

    if nodes is None:
        nodes = default_nodes

    # pars kwargs
    pars = {'nodes': nodes,
            'tol': tol,
            'metric': metric}

    # Obtain the observation matrix, size (n*n_steps) x n_scen, from which to compute distances. Normalize observations
    X = []
    S = samples
    for i in np.arange(np.shape(S)[2]):
        V = S[:, :, i]
        V_norm = (V - np.mean(V, 1).reshape(-1, 1)) / (np.std(V, 1) + 1e-6).reshape(-1, 1)
        X.append(V_norm)

    X = np.vstack(X).T

    D = get_dist(X, pars['metric'])

    # set diagonal element to max distance
    D = D + np.eye(D.shape[0]) * (1 + np.max(D.ravel()))
    infty = 1e12

    # generate the tolerance vector
    if np.all(pars['nodes'].ravel() == default_nodes.ravel()):
        # Tol = np.tile(pars['tol'], (1, T))[0]
        Tol = np.fliplr(pars['tol'] / (1.5 ** (n_steps - np.arange(n_steps).reshape(1, -1) + 1))).ravel()
        Tol[0] = infty
    else:
        Tol = infty * np.ones((1, n_steps)).ravel()

    if pars['nodes'][0] != 1:
        print(
            'The first number of scenarios is not 1. This will cause an error on graph retrieval. I am forcing it to 1')
        pars['nodes'][0] = 1

    J = np.asanyarray(np.ones((n_steps, n_scen)), bool)
    L = np.zeros((n_scen, n_scen))

    # probability matrix, initialized to the terminal probabilities of the leaves, 1/n_scen
    P = np.ones((n_steps, n_scen)) / n_scen
    branches = n_scen
    for i in np.arange(n_steps-1, -1, -1):
        delta_rel = 0
        delta_p = 0
        D_i = D

        # select scenarios which survived, up to the ith step
        basic_idx = np.asanyarray(np.hstack([np.ones((1, i + 1)), np.zeros((1, n_steps - i - 1))]), bool)
        sel_idx = np.tile(basic_idx, (1, samples.shape[2])).ravel()
        X_filt = X[J[i, :], :]
        X_filt = X_filt[:, sel_idx.ravel()]

        D_j = get_dist(X_filt, pars['metric'])
        D_j[np.asanyarray(np.eye(np.shape(D_j)[0]), bool)] = 0  # do not consider self - distance
        delta_max = np.min(np.sum(D_j, 0))

        while (delta_rel < Tol[i]) and (branches > pars['nodes'][i]):
            D_i[~J[i, :], :] = infty    # set distance of discarded scenarios to infinity, ignoring them
            D_i[:, ~J[i, :]] = infty

            # idx_s is a matrix containing in the ith element of first row the nearest neighbors of the ith scenario
            # in the second row the 2nd nearest neighbours and so on
            d_s = np.sort(D_i, 0)       # sort distances with respect to the non-discarded scenarios
            idx_s = np.argsort(D_i, 0)

            # create vector of weighted probabilities. The euristic for the merge is the following: merge the scenario
            # which cost less to be merged with its nearest neighbor in terms of weighted probability z
            z = d_s[0, :] * P[i, :]
            z[~J[i, :]] = infty         # set prob. of removed scenario to inf in order to ignore them
            idx_rem = np.argmin(z)      # find the scenario which cause the smallest p*d-deviation when merged,
            # and its index
            dp_min = np.min(z)
            idx_aug = idx_s[0, idx_rem]     # retrieve who's being augmented with the probability of idx_rem
            J[i, idx_rem] = False           # mark it as a removed scenarion in the current timestep

            # add the probability of the removed scenario to the closest scenario
            P[i, idx_aug] = P[i, idx_rem] + P[i, idx_aug]
            P[i, idx_rem] = 0                                # set probability of removed scenarios to 0
            branches = np.sum(P[i, :] > 0)                   # count remaining branches
            L[idx_aug, idx_rem] = 1                          # keep track of which scenario has merged

            # the scenario who's been augmented inherit all the scenarios previously merged with the removed scenario
            L[idx_aug, L[idx_rem, :] > 0] = 1

            # make the merged scenarios equal up to the root node
            S[0: i + 1, idx_rem, :] = S[0: i + 1, idx_aug, :]

            # make all the scenario previously merged with the removed one equal to the one in idx_aug,
            # up to the root node
            to_merge_idx = np.argwhere(L[[idx_rem], :])
            for j in np.arange(np.shape(to_merge_idx)[0]):
                S[: i + 1, to_merge_idx[j, 1], :] = S[: i + 1, idx_aug, :]

            # update the differential accuracy
            if not Tol[i] == infty:
                delta_p = delta_p + dp_min
            delta_rel = delta_p / delta_max

        if i > 0:
            # Update available scenarios in the previous timestep
            J[i - 1, :] = J[i, :]

            # update previous timestep probabilities
            P[i - 1, :] = P[i, :]
            D[~J[i, :], ~J[i, :]] = infty

        # print('Branches t=%i: %i' %  (i, branches))

    S = S[:, J[-1, :] > 0, :]
    P = P[:, J[-1, :] > 0]

    # obtain the match matrix
    Me = np.zeros((S.shape[1], S.shape[1], n_steps))
    for i in np.arange(n_steps):
        D = get_dist(S[i, :, 0].reshape(-1, 1), pars['metric'])
        # D[np.eye(D.shape[0],dtype=bool)] = 1
        Me[:, :, i] = D == 0

    Me = np.swapaxes(Me, 2, 0)

    if np.shape(S)[1] <= 500:
        g = get_network(S, P)
    else:
        logger.warning('Warning: automatic retrieval of the network graph has been disabled because the number of '
                       'nodes is high. If you want to get the network graph anyway, call the get_network(S,P)')
        g = []

    return S, P, J, Me, g


def get_network(S_s, P_s):
    '''
    Get a network representation from the S_s and P_p matrices. The network is encoded in a networkx graph, each node
    has the following attribute:
    t: time of the node
    p: probability of the node
    v: array of values associated with the node

    :param S_s: T*n_scen*n matrix, containing all the path in the scenario
                tree
    :param P_s: T*n_scen matrix, containing evolviong probabilities for each
                scenario
    :return: g: a networkx graph with time, probability and values encoded with the connectivity of the nodes
    '''

    assert np.sum(P_s[0, :] > 0) == 1, 'there is more than one node at root with P>0. Something wrong, ' \
                                       'most likely some observations at first step are equal down to epsilon'
    g = nx.DiGraph()
    g.add_node(0, t=0, p=1, v=S_s[0, P_s[0, :] > 0, :].ravel())

    for t in 1 + np.arange(P_s.shape[0] - 1):
        for s in np.arange(P_s.shape[1]):
            # span all the times, starting from second point (the root node is already defined)
            # consider current point mu, and its previous point in its scenario
            mu = S_s[t, s, :]
            p = P_s[t, s]
            # if probability of current point is zero, just go on
            if p == 0:
                continue
            mu_past = S_s[t - 1, s, :]
            # get current times in the tree
            times = np.array(list(nx.get_node_attributes(g, 't').values()))
            # get values, filtered by current time
            values = np.array(list(nx.get_node_attributes(g, 'v').values()))
            values_t = values[times == t, :]
            # values_past = values[times == t - 1, :]
            # check if values of current point s are present in the tree at current time
            if np.any([np.array_equal(mu, a) for a in values_t]):
                continue
            else:
                # find parent node
                try:
                    parent_node = [x for x, y in g.nodes(data=True) if
                                   y['t'] == t - 1 and np.array_equal(y['v'], mu_past)]
                except:
                    print(
                        'The tree does not have a root! It is likely that you did required more than one node as first node')
                # create the node and add an edge from current point to its parent
                new_node = len(g.nodes())
                g.add_node(new_node, t=t, p=p, v=mu, label=new_node)
                g.add_edge(parent_node[0], new_node)
                # print('node %i added, with values' % (new_node), mu)

    # approximated probabilities
    for i in np.arange(len(g.nodes)):
        g.nodes[i]['p2'] = np.sum([g.nodes[n]['t'] == t for n in nx.descendants(g, i)]) / np.sum(
            [g.nodes[n]['t'] == t for n in nx.descendants(g, 0)])
        if g.nodes[i]['p2'] == 0:
            g.nodes[i]['p2'] = 1 / len(nx.descendants(g, 0))

    return g


def set_distance(alphas, samples, metric):
    D = np.zeros((samples.shape[0], alphas.shape[0]))
    for i in np.arange(alphas.shape[0]):
        D[:, [i]] = cdist(samples, alphas[[i], :], metric=metric)
    # d = np.sum(np.minimum(D,0))
    d = np.sum(np.min(D, 1))
    return d, D


