import cvxpy as cp
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sb
from scipy.spatial.distance import  cdist

alpha_xi = 0.3
np.random.seed(10)
colors = plt.get_cmap('viridis', 100)
colors = [colors(4), colors(84)]

#colors = sb.color_palette('viridis', n_colors=20)
#colors = [colors[16], colors[1]]

n = 100*4
m = 2


def kantorovich_xitr(xi, xi_tr, relaxed=False):
    """
    Formulate the relaxed Kantorovich problem, given some initial values for xi_tr
    :param xi: scenarios for P
    :param m:
    :param n:
    :param xi_tr:
    :param relaxed:
    :return:
    """
    m = xi_tr.shape[0]
    n = xi.shape[0]
    pi = cp.Variable((m, n))

    if not relaxed:
        p_tr = cp.Parameter(m)
    else:
        p_tr = cp.Variable(m)

    distances = cdist(xi_tr, xi, 'sqeuclidean')
    cost = cp.sum(cp.multiply(pi, distances))
    constraints = [cp.sum(pi, axis=1) == p_tr,
                   cp.sum(pi, axis=0) == 1/n,
                    pi >= 0]
    prob = cp.Problem(cp.Minimize(cost), constraints)
    return prob, pi, p_tr


def rule_based_xitr(xi, xi_tr):
    """
    Rule-based solutions for the relaxed Kantorovich problem, given some initial values for xi_tr
    :param xi:
    :param xi_tr:
    :return:
    """
    m = xi_tr.shape[0]
    n = xi.shape[0]
    distances = cdist(xi_tr, xi, 'sqeuclidean')
    pi = np.zeros((m, n))
    pi[np.argmin(distances, axis=0), range(n)] = 1
    pi /= n
    return np.sum(distances*pi), pi, pi.sum(axis=1)


xi = np.sort(np.hstack([np.random.randn(int(3*n/4)), (np.random.randn(int(n/4))/2+4)])).reshape(-1, 1)

outer = [['a', 'a'],
         ['b', 'b'],
        ['c', 'd'],
        ['c', 'd']]

fig, ax = plt.subplot_mosaic(outer, layout="constrained", figsize=(5, 6))


xi_tr = np.vstack([-2.4, 3]).reshape(-1, 1)
prob, pi, p_tr = kantorovich_xitr(xi, xi_tr, relaxed=True)
prob.solve()
distopt, pi_rb, p_tr_rb = rule_based_xitr(xi, xi_tr)

c = [colors[0] if pi.value[0, i] > pi.value[1, i] else colors[1] for i in range(n)]
[ax['a'].vlines(xi_tr.ravel()[i], 0, p_tr.value[i], colors=colors[i], linewidth=2) for i in range(m)]
sb.kdeplot(xi, alpha=0.2, ax=ax['a'])
sb.rugplot(np.squeeze(xi), alpha=alpha_xi, c=c, height=0.05, linewidth=0.5, ax=ax['a'])
# put text on the upper left corner with the value of the objective function
ax['a'].text(0.02, 0.85, r'cvx, $\tilde{d}$:' + ' {:.2f}'.format(prob.value), transform=ax['a'].transAxes)
ax['a'].text(0.02, 0.65, r'rel diff $\tilde{d}$:' + ' {:.2e}'.format((prob.value-distopt)/prob.value), transform=ax['a'].transAxes)
ax['a'].get_legend().remove()

xi_tr = np.vstack([0, 4]).reshape(-1, 1)
prob, pi, p_tr = kantorovich_xitr(xi, xi_tr, relaxed=True)
prob.solve()
distopt, pi_rb, p_tr_rb = rule_based_xitr(xi, xi_tr)

c = [colors[0] if pi.value[0, i] > pi.value[1, i] else colors[1] for i in range(n)]

[ax['b'].vlines(xi_tr.ravel()[i], 0, p_tr.value[i]/np.sum(p_tr.value), colors=colors[i], linewidth=2) for i in range(m)]
sb.kdeplot(xi, alpha=0.2, label='kde', ax=ax['b'])
sb.rugplot(np.squeeze(xi), alpha=alpha_xi, label='xi', c=c, height=0.05, linewidth=0.5, ax=ax['b'])
# put text on the upper left corner with the value of the objective function
ax['b'].text(0.02, 0.85, r'cvx, $\tilde{d}$:' + ' {:.2f}'.format(prob.value), transform=ax['b'].transAxes)
ax['b'].text(0.02, 0.65, r'rel diff $\tilde{d}$:' + ' {:.2e}'.format((prob.value-distopt)/prob.value), transform=ax['b'].transAxes)
ax['b'].get_legend().remove()
ax['a'].set_ylim(0, 0.8)
ax['b'].set_ylim(0, 0.8)

xi = np.vstack([np.random.randn(int(3*n/4), 2), (np.random.randn(int(n/4), 2)/2+4)])
xi_tr = np.array([[-3, -2], [2, 2]])
prob, pi, p_tr = kantorovich_xitr(xi, xi_tr, relaxed=True)
prob.solve()
distopt, pi_rb, p_tr_rb = rule_based_xitr(xi, xi_tr)

c = [colors[0] if pi.value[0, i] > pi.value[1, i] else colors[1] for i in range(n)]

ax['c'].scatter(*xi.T, c=c, s=1, alpha=alpha_xi)
ax['c'].scatter(*xi_tr.T, c=colors, s=p_tr.value*150)

sb.kdeplot(x=xi[:, 0], y=xi[:, 1], ax=ax['c'], alpha=0.2)
ax['c'].text(0.02, 0.9, r'cvx, $\tilde{d}$:' + ' {:.2f}'.format(prob.value), transform=ax['c'].transAxes)
ax['c'].text(0.02, 0.8, r'rel diff $\tilde{d}$:' + ' {:.2e}'.format((prob.value-distopt)/prob.value), transform=ax['c'].transAxes)

xi_tr = np.array([[0, 0], [4, 4]])
prob, pi, p_tr = kantorovich_xitr(xi, xi_tr, relaxed=True)
prob.solve()
distopt, pi_rb, p_tr_rb = rule_based_xitr(xi, xi_tr)
c = [colors[0] if pi.value[0, i] > pi.value[1, i] else colors[1] for i in range(n)]

sb.kdeplot(x=xi[:, 0], y=xi[:, 1], ax=ax['d'], alpha=0.2)
ax['d'].scatter(*xi.T, c=c, s=1, alpha=alpha_xi)
ax['d'].scatter(*xi_tr.T, c=colors, s=p_tr.value*150)
ax['d'].text(0.1, 0.9, r'cvx, $\tilde{d}$:' + ' {:.2f}'.format(prob.value), transform=ax['d'].transAxes)
ax['d'].text(0.1, 0.8, r'rel diff $\tilde{d}$:' + ' {:.2e}'.format((prob.value-distopt)/prob.value), transform=ax['d'].transAxes)

# remove spines
[a.spines[['right', 'top']].set_visible(False) for a in ax.values()]

plt.savefig('results/relaxed_Kantorovich.pdf')

