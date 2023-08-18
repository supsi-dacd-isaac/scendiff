# scendiff 

This repository contains the code for the paper "Scenario Tree Generation via Adaptive Gradient Descent, preprint, 2023"


# Installation
To install the package, run `python setup.py install` in the root directory.


In order to enable tree visualization with graphviz you need to

`sudo apt-get install libgraphviz-dev graphviz`

and

`pip install pygraphviz`

# Usage

### Experiments 
To run experiments run `experiments.py`. This reproduces the main results in the paper. 

### Relaxed Kantorovich problem examples 
To reproduce figure 1 in the paper, run `kantorovich_relaxation.py`. This solves four instances of the relaxed
Kantorovich problem using both cvyxpy and the formulation from theorem 1. 

### q-gen example
To reproduce figure 2 in the paper, run `q_gen_plots.py`. 
