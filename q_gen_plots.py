import multiprocessing
import pandas as pd
import numpy as np
from scendiff.trees import  QuantileTree
from scendiff.plot_utils import plot_from_graph
from synthetic_processes import sin_process, random_walk
import matplotlib.pyplot as plt
import seaborn as sb
import networkx as nx

# ----------------------------------------------------------------------------------------------------------------------
# ------------------------------------ Define parameters ---------------------------------------------------------------
# ----------------------------------------------------------------------------------------------------------------------
np.random.seed(5)
colors = sb.color_palette('viridis', n_colors=100)
c1 = colors[84]
c2 = colors[4]
alpha = 0.4
s = 3               # number of steps
n = 120             # number of scenarios
scen = sin_process(steps=s, n_scens=n, double=True) # generate scenarios


# ----------------------------------------------------------------------------------------------------------------------
# ------------------------------------ Plot scenarios and bins ---------------------------------------------------------
# ----------------------------------------------------------------------------------------------------------------------
fig, ax = plt.subplots(1, 1, layout='tight', figsize=(4.5, 3))
sb.set_style('white')
plt.plot(scen, color=c1, alpha=0.12, linewidth=0.15)
plt.plot(scen, color=c1,  marker='.', linestyle='None', markersize=1)
# remove upper and right spines
ax.spines[['top', 'right', 'left']].set_visible(False)

times = np.arange(s)
# plot a transprent vertical patch from -05 to 0.5 and from 1.5 to 2.5
colors = plt.get_cmap('tab10').colors
qs_0 = np.quantile(scen[0, :], [0, 1], axis=0)
patches = []
patches.append(ax.fill_betweenx(qs_0, times[0]-0.1, times[0]+0.1, color=colors[0], alpha=alpha))

qs_1 = np.quantile(scen[1, :], np.linspace(0, 1, 3), axis=0)
patches.append(ax.fill_betweenx(qs_1[:2], times[1]-0.1, times[1]+0.1, color=colors[1], alpha=alpha))
patches.append(ax.fill_betweenx(qs_1[1:], times[1]-0.1, times[1]+0.1, color=colors[2], alpha=alpha))


scen_11 = scen[:, scen[1, :] < qs_1[1]]
qs_21 = np.quantile(scen_11[2, :], np.linspace(0, 1, 3), axis=0)
patches.append(ax.fill_betweenx(qs_21[:2], times[2]-0.1, times[2]+0.1, color=colors[3], alpha=alpha))
patches.append(ax.fill_betweenx(qs_21[1:], times[2]-0.1, times[2]+0.1, color=colors[4], alpha=alpha))


scen_12 = scen[:, scen[1, :] >= qs_1[1]]
qs_22 = np.quantile(scen_12[2, :], np.linspace(0, 1, 3), axis=0)
patches.append(ax.fill_betweenx(qs_22[:2], times[2]-0.1, times[2]+0.1, color=colors[5], alpha=alpha))
patches.append(ax.fill_betweenx(qs_22[1:], times[2]-0.1, times[2]+0.1, color=colors[6], alpha=alpha))

# write text in the rigth corner of the patch p1
labels = np.arange(1, 8)
positions = [patches[0].get_paths()[0].get_extents().get_points()[1] + np.array([-0.1, 0.1]),
             patches[1].get_paths()[0].get_extents().get_points()[0] + np.array([+0.05, -0.26]),
             patches[2].get_paths()[0].get_extents().get_points()[1] + np.array([-0.15, +0.05]),
             patches[3].get_paths()[0].get_extents().get_points()[0] + np.array([+0.22, +0.3]),
             patches[4].get_paths()[0].get_extents().get_points()[0] + np.array([+0.22, +0.1]),
             patches[5].get_paths()[0].get_extents().get_points()[1] + np.array([+0.02, -0.15]),
             patches[6].get_paths()[0].get_extents().get_points()[1] + np.array([+0.02, -0.3])
             ]
for p, label, pos, c in zip(patches, labels, positions, colors):
    ax.text(*pos, '$F_{}$'.format(label), fontsize=20, color=c)


# set xticks to the time (integer) values
ax.set_xticks(times)
ax.set_yticks([])

# ----------------------------------------------------------------------------------------------------------------------
# ------------------------------------ Plot q-gen tree -----------------------------------------------------------------
# ----------------------------------------------------------------------------------------------------------------------
tr = QuantileTree().gen_tree(scen,nodes_at_step=[0, 2, 4])[0]
plot_from_graph(tr, ax=ax, color=c2, marker='.', markersize=10, alpha=0.9, linewidth=0.6)
ax.set_xlabel('time step')

for n in tr.nodes:
    ax.text(tr.nodes[n]['t']-0.12, tr.nodes[n]['v']+0.1, '$n_{}$'.format(n+1), fontsize=10, color=c2)

plt.savefig('results/quantile_tree.pdf')
