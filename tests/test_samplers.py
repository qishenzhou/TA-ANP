import unittest
from unittest.mock import patch

import numpy as np
import pandas as pd
import torch
from torch_geometric.data import Data

from data_provider.madrid_dataset import MadridDataset
from data_provider.sampler import NPSampler1, NPSampler1Resilient


class SamplerTest(unittest.TestCase):
    def dataset(self, split, length):
        dataset = MadridDataset.__new__(MadridDataset)
        dataset.split = split
        dataset.len = lambda: length
        dataset.full_graph = Data(
            num_nodes=12, n_id=torch.arange(12), n_idx=torch.arange(12),
            edge_index=torch.stack([torch.arange(12), torch.arange(12)]),
            edge_attr=torch.ones(12),
        )
        dataset.permanent_unobs_nodes_global_idx = np.arange(8, 12)
        return dataset

    def clusters(self):
        return pd.DataFrame({"node_id": np.arange(12), "cluster_id_road_graph": 0})

    def test_training_and_test_partitions_are_preserved(self):
        torch.manual_seed(42)
        np.random.seed(42)
        with patch("data_provider.sampler.pd.read_csv", return_value=self.clusters()):
            training = NPSampler1(self.dataset("train", 5), 2, 1, shuffle=False)
            testing = NPSampler1(self.dataset("test", 5), 2, 1, shuffle=False)
        self.assertEqual(len(training), 3)
        for batch in training:
            graph = batch[0][1]
            self.assertGreater(len(graph.unobs_global_idx), 0)
            self.assertLess(len(graph.unobs_global_idx), graph.num_nodes)
        batches = list(testing)
        self.assertEqual([t for batch in batches for t, _ in batch], list(range(5)))
        for batch in batches:
            torch.testing.assert_close(batch[0][1].unobs_global_idx, torch.arange(8, 12))

    def test_damage_repair_addition_lifecycle_is_preserved(self):
        with patch("data_provider.sampler.pd.read_csv", return_value=self.clusters()), \
             patch("data_provider.sampler.np.save"), \
             patch("data_provider.sampler.np.load", return_value=np.arange(8)):
            sampler = NPSampler1Resilient(self.dataset("test", 12 * 96), 2, 1)
        counts = {}
        with patch("data_provider.sampler.print", create=True):
            for batch in sampler:
                t, graph = batch[0]
                if t in (0, 288, 576, 864):
                    counts[t] = len(graph.unobs_global_idx)
        self.assertEqual(counts, {0: 4, 288: 12, 576: 4, 864: 0})


if __name__ == "__main__":
    unittest.main()
