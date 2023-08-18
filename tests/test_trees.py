import unittest
from scendiff.trees import NeuralGas, DiffTree, ScenredTree, QuantileTree
from scendiff.plot_utils import plot_from_graph
import matplotlib.pyplot as plt
from synthetic_processes import sin_process


class TestScenarios(unittest.TestCase):
    def setUp(self) -> None:
        self.ng = NeuralGas(init_vals_method='scenred', init_tree_method='scenred')
        self.ndt = DiffTree(init_vals_method='scenred', init_tree_method='scenred')
        self.srt = ScenredTree()
        self.qt = QuantileTree()
        self.test_scens = sin_process(24, 20, double=True)

    def test_tree_gens(self):

        tree_ng, _, _, _ = self.ng.gen_tree(self.test_scens, k_max=200, do_plot=True)
        tree_d, _, _, _ = self.ndt.gen_tree(self.test_scens, k_max=200, do_plot=True)
        tree_sr, _, _, _ = self.srt.gen_tree(self.test_scens, k_max=10)
        tree_q, _, _, _ = self.qt.gen_tree(self.test_scens, k_max=10)

        plot_from_graph(tree_ng)
        plot_from_graph(tree_d, ax=plt.gca(), color='r')
        plt.pause(0.1)

        plot_from_graph(tree_sr, ax=plt.gca(), linestyle='--')
        plot_from_graph(tree_q)
        plt.pause(0.1)


if __name__ == '__main__':
    unittest.main()
