import cvxpy as cp
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sb
from scipy.spatial.distance import  cdist

np.random.seed(10)
colors = plt.get_cmap('viridis', 10)
colors = [colors(2), colors(8)]

n = 50*4
m = 2


def kantorovich_xitr(xi, m, n, xi_tr, relaxed=False):
    # Formulate the Kantorovich problem, given some initial values for xi_tr

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


def rule_based_xitr(xi, m, n, xi_tr):
    distances = cdist(xi_tr, xi, 'sqeuclidean')
    pi = np.zeros((m, n))
    pi[np.argmin(distances, axis=0), range(n)] = 1
    pi /= n
    return np.sum(distances*pi), pi, pi.sum(axis=1)


xi = np.sort(np.hstack([np.random.randn(int(3*n/4)), (np.random.randn(int(n/4))/2+4)])).reshape(-1, 1)

fig, ax = plt.subplots(2, 2, figsize=(8, 5), layout='tight')

xi_tr = np.vstack([-1, 3]).reshape(-1, 1)
prob, pi, p_tr = kantorovich_xitr(xi, m, n, xi_tr, relaxed=True)
prob.solve()
distopt, pi_rb, p_tr_rb = rule_based_xitr(xi, m, n, xi_tr)

c = [colors[0] if pi.value[0, i] > pi.value[1, i] else colors[1] for i in range(n)]
[ax[0, 0].vlines(xi_tr.ravel()[i], 0, p_tr.value[i], colors=colors[i], linewidth=2) for i in range(m)]
sb.kdeplot(xi, alpha=0.5, ax=ax[0, 0])
sb.rugplot(np.squeeze(xi), alpha=0.5, c=c, height=0.1, linewidth=1.5, ax=ax[0, 0])
# put text on the upper left corner with the value of the objective function
ax[0, 0].text(0.5, 0.9, 'cvx, $d_K$: {:.2f}'.format(prob.value), transform=ax[0, 0].transAxes)
ax[0, 0].text(0.5, 0.8, 'rel diff $d_K$: {:.2e}'.format((prob.value-distopt)/prob.value), transform=ax[0, 0].transAxes)
ax[0, 0].get_legend().remove()

xi_tr = np.vstack([0, 4]).reshape(-1, 1)
prob, pi, p_tr = kantorovich_xitr(xi, m, n, xi_tr, relaxed=True)
prob.solve()
distopt, pi_rb, p_tr_rb = rule_based_xitr(xi, m, n, xi_tr)

c = [colors[0] if pi.value[0, i] > pi.value[1, i] else colors[1] for i in range(n)]

[ax[1, 0].vlines(xi_tr.ravel()[i], 0, p_tr.value[i]/np.sum(p_tr.value), colors=colors[i], linewidth=2) for i in range(m)]
sb.kdeplot(xi, alpha=0.5, label='kde', ax=ax[1, 0])
sb.rugplot(np.squeeze(xi), alpha=0.5, label='xi', c=c, height=0.1, linewidth=1.5, ax=ax[1, 0])
# put text on the upper left corner with the value of the objective function
ax[1, 0].text(0.5, 0.9, 'cvx, $d_K$: {:.2f}'.format(prob.value), transform=ax[1, 0].transAxes)
ax[1, 0].text(0.5, 0.8, 'rel diff $d_K$: {:.2e}'.format((prob.value-distopt)/prob.value), transform=ax[1, 0].transAxes)
ax[1, 0].get_legend().remove()
ax[0, 0].set_ylim(0, 0.8)
ax[1, 0].set_ylim(0, 0.8)


xi = np.vstack([np.random.randn(int(3*n/4), 2), (np.random.randn(int(n/4), 2)/2+4)])

xi_tr = np.array([[-3, -2], [2, 4]])
prob, pi, p_tr = kantorovich_xitr(xi, m, n, xi_tr, relaxed=True)
prob.solve()
distopt, pi_rb, p_tr_rb = rule_based_xitr(xi, m, n, xi_tr)

c = [colors[0] if pi.value[0, i] > pi.value[1, i] else colors[1] for i in range(n)]

ax[0, 1].scatter(*xi.T, c=c, s=1)
ax[0, 1].scatter(*xi_tr.T, c=colors, s=p_tr.value*150)
ax[0, 1].text(0.1, 0.9, 'cvx, $d_K$: {:.2f}'.format(prob.value), transform=ax[0, 1].transAxes)
ax[0, 1].text(0.1, 0.8, 'rel diff $d_K$: {:.2e}'.format((prob.value-distopt)/prob.value), transform=ax[0, 1].transAxes)

xi_tr = np.array([[0, 0], [4, 4]])
prob, pi, p_tr = kantorovich_xitr(xi, m, n, xi_tr, relaxed=True)
prob.solve()
distopt, pi_rb, p_tr_rb = rule_based_xitr(xi, m, n, xi_tr)
c = [colors[0] if pi.value[0, i] > pi.value[1, i] else colors[1] for i in range(n)]

ax[1, 1].scatter(*xi.T, c=c, s=1)
ax[1, 1].scatter(*xi_tr.T, c=colors, s=p_tr.value*150)
ax[1, 1].text(0.1, 0.9, 'cvx, $d_K$: {:.2f}'.format(prob.value), transform=ax[1, 1].transAxes)
ax[1, 1].text(0.1, 0.8, 'rel diff $d_K$: {:.2e}'.format((prob.value-distopt)/prob.value), transform=ax[1, 1].transAxes)

# remove spines
[ax[i, j].spines[['right', 'top']].set_visible(False) for i in range(2) for j in range(2)]

plt.savefig('results/relaxed_kantorovich.pdf')