import random
import numpy as np
import pandas as pd
import torch
from torch.utils.data.sampler import Sampler
from data_provider.madrid_dataset import MadridDataset


class NPSampler1(Sampler):
    """
    Smapler for neural process training.
    This sampler is based on  "Le, T.A., Kim, H., Garnelo, M., n.d. Empirical Evaluation of Neural Process Objectives."
    the ratio of context nodes and target nodes m ~[0.1,0.9], n ~ [m+1, 1]

    Generates batch indices for both temporal and spatial dimensions,
    specifically designed for neural process.
    Each batch contains a combination of temporal indices and spatial indices.
    """

    def __init__(
        self,
        dataset: MadridDataset,
        t_batch_size: int,
        s_batch_size: int,
        shuffle: bool = True,
        drop_last: bool = False,
        num_cluster: int = 10,
    ):
        """_summary_

        Args:
            dataset (MadridDataset): pytorch dataset instance
            t_batch_size (int): batch size for temporal dimension
            s_batch_size (int): batch size for spatial dimension
            num_neighbors (tuple[int]): number of neighbors for each hop of the graph
            shuffle (bool, optional): whether to shuffle temporal indices. Defaults to True.
            drop_last (bool, optional): whether to drop the last temporal batch. Defaults to False.
            num_cluster (int, optional): number of clusters for the graph. Defaults to 10.
        """

        self.dataset = dataset
        self.t_batch_size = t_batch_size
        self.s_batch_size = s_batch_size
        self.shuffle = shuffle
        self.drop_last = drop_last
        self.num_cluster = num_cluster
        # Get dataset length (temporal dimension)
        self.num_time_steps = dataset.len()

        # Calculate number of temporal batches
        if self.drop_last:
            self.num_t_batches = self.num_time_steps // self.t_batch_size
        else:
            self.num_t_batches = (self.num_time_steps + self.t_batch_size -
                                  1) // self.t_batch_size

        if isinstance(self.dataset, MadridDataset):
            dataset_all_nodes_id = self.dataset.full_graph.n_id
            road_cluster_data = pd.read_csv(
                'MadridAllinOne/DLData/train/cluster_id_road_graph.csv')
            cluster_data = []
            for i, df in road_cluster_data.groupby('cluster_id_road_graph'):
                cluster_nodes = torch.tensor(df.node_id.to_list())
                is_cluster_nodes = torch.isin(dataset_all_nodes_id, cluster_nodes)
                cluster_data.append(self.dataset.full_graph.subgraph(is_cluster_nodes))
        else:
            cluster_data = [self.dataset.full_graph]

        self.spatial_loader = cluster_data
        self.num_s_batches = len(self.spatial_loader)

    def __iter__(self):
        """
        Generate iterator for spatio-temporal batch indices.

        Returns:
            Iterator: Each iteration returns a batch containing spatio-temporal indices
        """
        # Generate temporal batches
        t_indices = list(range(self.num_time_steps))
        if self.shuffle:
            # Shuffle temporal indices if required
            random.shuffle(t_indices)

        # Group temporal indices into batches
        t_batches = [
            t_indices[i:i + self.t_batch_size]
            for i in range(0, len(t_indices), self.t_batch_size)
        ]

        # Drop last incomplete batch if required
        if self.drop_last and len(t_batches[-1]) < self.t_batch_size:
            t_batches = t_batches[:-1]

        # For each combination of spatial and temporal batches
        for t_batch in t_batches:
            for _, s_batch in enumerate(self.spatial_loader):
                # Each batch is a list of indices, where each index is a tuple:
                # (temporal_index, current_spatial_batch, observed_nodes)
                if self.dataset.split != "test":
                    dynamic_context_num = max(
                        1, int(np.random.uniform(0.1, 0.9) * s_batch.num_nodes))
                    dynamic_subgraph_num = int(
                        np.random.uniform(dynamic_context_num + 1, s_batch.num_nodes))
                    dynamic_unobs_num = dynamic_subgraph_num - dynamic_context_num
                    sub_graph_node_index = torch.randperm(
                        s_batch.num_nodes)[:dynamic_subgraph_num]
                    # get sub_s_batch based on sub_graph_node_index
                    s_batch = s_batch.subgraph(sub_graph_node_index)
                    s_batch.unobs_global_idx = torch.from_numpy(
                        np.random.choice(
                            s_batch.n_idx,
                            size=dynamic_unobs_num,
                            replace=False,
                        ))  # control each cluster have same unobs rate
                else:
                    s_batch.unobs_global_idx = s_batch.n_idx[torch.isin(
                        s_batch.n_idx,
                        torch.from_numpy(self.dataset.permanent_unobs_nodes_global_idx))]
                yield [(t_idx, s_batch) for t_idx in t_batch]

    def __len__(self):
        """
        Calculate total number of batches.

        Returns:
            int: Total number of batches = number of temporal batches × number of spatial batches
        """

        # Total batches = temporal batches × spatial batches
        return self.num_t_batches * self.num_s_batches


class NPSampler1Resilient(NPSampler1):
    """
    Resilient sampler for neural process training.
    This sampler simulates sensor failure-maintenance-addition lifecycle.

    Key differences from NPSampler1:
    1. In test mode, temporal indices are NOT shuffled (maintain chronological order)
    2. Pre-stores two types of nodes:
       - Failed Nodes: selected from non-permanent_unobs_nodes_global_idx (will fail)
       - New Added Nodes: selected from permanent_unobs_nodes_global_idx (will be added)
    3. Simulates sensor lifecycle: failure (3 days) -> maintenance (3 days) -> addition (3 days)
       Each phase lasts 3 days = 72 time steps (1 time step = 1 hour)
       Rate: 100 nodes per day, accumulating daily (not hourly)
    """

    def __init__(
        self,
        dataset: MadridDataset,
        t_batch_size: int,
        s_batch_size: int,
        shuffle: bool = True,
        drop_last: bool = False,
        num_cluster: int = 10,
    ):
        """
        Args:
            dataset (MadridDataset): pytorch dataset instance
            t_batch_size (int): batch size for temporal dimension
            s_batch_size (int): batch size for spatial dimension
            shuffle (bool, optional): whether to shuffle temporal indices. Defaults to True.
            drop_last (bool, optional): whether to drop the last temporal batch. Defaults to False.
            num_cluster (int, optional): number of clusters for the graph. Defaults to 10.
            failed_nodes_ratio (float, optional): ratio of nodes that will fail. Defaults to 0.1.
            new_added_nodes_ratio (float, optional): ratio of nodes that will be newly added. Defaults to 0.1.
            nodes_per_day (int, optional): number of nodes processed per day. Defaults to 100.
        """
        t_batch_size = 1  #force be 1 for resilient sampler
        super().__init__(dataset, t_batch_size, s_batch_size, shuffle, drop_last,
                         num_cluster)

        # Phase duration: 3 days = 72 time steps (1 time step = 1 hour)
        self.phase_duration = 3 * 24 * 4
        self.nodes_per_day = 100
        self.time_steps_per_day = 24 * 4  # 1 time step = 1 hour, 1 day = 24 hours

        # Pre-store node categories for lifecycle simulation
        if self.dataset.split == "test":
            # Get all nodes from the dataset (use n_idx, which is global_idx)
            dataset_all_nodes_idx = self.dataset.full_graph.n_idx.numpy()
            id_to_idx_dict = dict(
                zip(self.dataset.full_graph.n_id.numpy(),
                    self.dataset.full_graph.n_idx.numpy()))
            # Failed Nodes: selected from non-permanent_unobs_nodes_global_idx
            # (nodes that are currently observable but will fail)
            # Total: 3 days × 100 nodes/day = 300 nodes
            permanent_unobs_global_idx = self.dataset.permanent_unobs_nodes_global_idx
            all_observable_nodes = dataset_all_nodes_idx[
                ~np.isin(dataset_all_nodes_idx, permanent_unobs_global_idx)]
            total_failed_nodes = 3 * self.nodes_per_day  # 300 nodes for 3 days
            num_failed_nodes = min(total_failed_nodes, len(all_observable_nodes))
            # Select nodes and keep the order (for maintenance order)
            np.random.seed(42)
            self.failed_nodes_global_idx = np.random.choice(all_observable_nodes,
                                                            size=num_failed_nodes,
                                                            replace=False).tolist()
            failied_nodes_global_id = self.dataset.full_graph.n_id[np.isin(
                self.dataset.full_graph.n_idx, self.failed_nodes_global_idx)]
            np.save('failed_nodes_global_id.npy', failied_nodes_global_id)

            sorted_failed_nodes_global_id = np.load(
                'sorted_failed_nodes_by_uncertainty.npy')
            sorted_failed_nodes_global_idx = np.array(
                [id_to_idx_dict[n_id] for n_id in sorted_failed_nodes_global_id])
            self.sorted_failed_nodes_global_idx = sorted_failed_nodes_global_idx.tolist()

            # New Added Nodes: selected from permanent_unobs_nodes_global_idx
            # (nodes that are currently unobservable but will be added)
            # Total: 3 days × 100 nodes/day = 300 nodes
            total_new_added_nodes = 3 * self.nodes_per_day  # 300 nodes for 3 days
            num_new_added_nodes = min(total_new_added_nodes,
                                      len(self.dataset.permanent_unobs_nodes_global_idx))
            # Select nodes and keep the order (for addition order)
            np.random.seed(42)
            self.new_added_nodes_global_idx = np.random.choice(
                self.dataset.permanent_unobs_nodes_global_idx,
                size=num_new_added_nodes,
                replace=False).tolist()

            # permanent_unobs_nodes_global_id = self.dataset.full_graph.n_id[np.isin(
            #     self.dataset.full_graph.n_idx,
            #     self.dataset.permanent_unobs_nodes_global_idx)]
            # np.save('permanent_unobs_nodes_global_id.npy',
            #         permanent_unobs_nodes_global_id)

            # sorted_new_added_nodes_global_id = np.load('top_new_nodes_by_uncertainty.npy')
            # # Create a mapping from n_id to n_idx using dictionary for O(1) lookup
            # # Get idx in the same order as sorted_new_added_nodes_global_id
            # sorted_new_added_nodes_global_idx = np.array(
            #     [id_to_idx_dict[n_id] for n_id in sorted_new_added_nodes_global_id])
            # self.new_added_nodes_global_idx = sorted_new_added_nodes_global_idx.tolist()

        else:
            # For train/val, use empty tensors (not used)
            self.failed_nodes_global_idx = []
            self.new_added_nodes_global_idx = []

    def __iter__(self):
        """
        Generate iterator for spatio-temporal batch indices.
        In test mode, maintains chronological order and simulates sensor lifecycle.

        Returns:
            Iterator: Each iteration returns a batch containing spatio-temporal indices
        """
        # Generate temporal batches
        t_indices = list(range(self.num_time_steps))

        # In test mode, do NOT shuffle to maintain chronological order
        if self.shuffle and self.dataset.split != "test":
            random.shuffle(t_indices)

        # For each combination of spatial and temporal batches
        for t_idx in t_indices:
            for _, s_batch in enumerate(self.spatial_loader):
                if self.dataset.split != "test":
                    # Training/validation mode: use original logic
                    dynamic_context_num = max(
                        1, int(np.random.uniform(0.1, 0.9) * s_batch.num_nodes))
                    dynamic_subgraph_num = int(
                        np.random.uniform(dynamic_context_num + 1, s_batch.num_nodes))
                    dynamic_unobs_num = dynamic_subgraph_num - dynamic_context_num
                    sub_graph_node_index = torch.randperm(
                        s_batch.num_nodes)[:dynamic_subgraph_num]
                    s_batch = s_batch.subgraph(sub_graph_node_index)
                    s_batch.unobs_global_idx = torch.from_numpy(
                        np.random.choice(
                            s_batch.n_idx,
                            size=dynamic_unobs_num,
                            replace=False,
                        ))
                else:
                    # Test mode: simulate sensor lifecycle
                    # For each temporal index in the batch, determine unobserved nodes
                    # Determine which phase we're in based on time step
                    phase = t_idx // self.phase_duration
                    phase_progress = t_idx % self.phase_duration  # Progress within current phase (0-71)
                    # Initialize unobserved nodes with permanent unobserved nodes
                    current_unobs_global_idx = self.dataset.permanent_unobs_nodes_global_idx.tolist(
                    )

                    # Phase 0: Normal operation (0-71 time steps)
                    # All nodes are in their original state
                    if phase == 0:
                        # No additional nodes fail in phase 0
                        current_unobs_global_idx = torch.from_numpy(
                            self.dataset.permanent_unobs_nodes_global_idx)
                    # Phase 1: Failure period (72-143 time steps)
                    # Failed nodes accumulate daily: day 1 = 100, day 2 = 200, day 3 = 300
                    elif phase == 1:
                        # Calculate which day within the failure phase (0, 1, or 2)
                        day_in_phase = phase_progress // self.time_steps_per_day

                        # Number of nodes that have failed by this day
                        # Day 0: 100 nodes, Day 1: 200 nodes, Day 2: 300 nodes
                        total_failed_nodes = (day_in_phase + 1) * self.nodes_per_day
                        total_failed_nodes = min(total_failed_nodes,
                                                 len(self.failed_nodes_global_idx))

                        # Get the first total_failed_nodes nodes (maintain order)
                        current_failed_global_idx = self.failed_nodes_global_idx[:
                                                                                 total_failed_nodes]

                        current_unobs_global_idx = torch.tensor(list(
                            set(current_unobs_global_idx + current_failed_global_idx)),
                                                                dtype=torch.long)
                    # Phase 2: Maintenance period (144-215 time steps)
                    # Failed nodes are gradually repaired daily: day 1 = 100, day 2 = 200, day 3 = 300
                    elif phase == 2:
                        # Calculate which day within the maintenance phase (0, 1, or 2)
                        day_in_phase = phase_progress // self.time_steps_per_day

                        # Number of nodes that have been repaired by this day
                        # Day 0: 100 nodes repaired, Day 1: 200 nodes repaired, Day 2: 300 nodes repaired
                        total_repaired_nodes = (day_in_phase + 1) * self.nodes_per_day
                        total_repaired_nodes = min(total_repaired_nodes,
                                                   len(self.failed_nodes_global_idx))

                        # Get the first total_repaired_nodes nodes (same order as failure)\
                        still_failed_nodes = self.failed_nodes_global_idx[
                            total_repaired_nodes:]

                        # still_failed_nodes = self.sorted_failed_nodes_global_idx[
                        #     total_repaired_nodes:]

                        current_unobs_global_idx = torch.tensor(list(
                            set(current_unobs_global_idx + still_failed_nodes)),
                                                                dtype=torch.long)

                    # Phase 3: Addition period (216+ time steps)
                    # New nodes are gradually added daily: day 1 = 100, day 2 = 200, day 3 = 300
                    elif phase >= 3:
                        if phase == 3:
                            # Phase 3: days 0, 1, 2
                            day_in_phase = phase_progress // self.time_steps_per_day
                        else:
                            # Phase 4+: all nodes have been added (all 3 days completed)
                            day_in_phase = 2
                        # Number of nodes that have been added by this day
                        # Day 0: 100 nodes added, Day 1: 200 nodes added, Day 2: 300 nodes added
                        total_added_nodes = (day_in_phase + 1) * self.nodes_per_day
                        total_added_nodes = min(total_added_nodes,
                                                len(self.new_added_nodes_global_idx))

                        # Get the first total_added_nodes nodes (in selection order)
                        added_nodes = self.new_added_nodes_global_idx[:total_added_nodes]

                        current_unobs_global_idx = list(
                            set(current_unobs_global_idx) - set(added_nodes))
                        current_unobs_global_idx = torch.tensor(current_unobs_global_idx,
                                                                dtype=torch.long)
                    print(
                        f"phase: {phase}, phase_progress: {phase_progress}, len_of_unobs: {len(current_unobs_global_idx)}"
                    )
                    # Set unobserved nodes for this time step
                    # Only include nodes that are in the current s_batch
                    s_batch.unobs_global_idx = s_batch.n_idx[torch.isin(
                        s_batch.n_idx, current_unobs_global_idx)]

                    # s_batch.unobs_global_idx = s_batch.n_idx[torch.isin(
                    #     s_batch.n_idx,
                    #     torch.from_numpy(self.dataset.permanent_unobs_nodes_global_idx))]
                yield [(t_idx, s_batch)]
