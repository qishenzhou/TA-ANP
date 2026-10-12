from pathlib import Path
from typing import List, Tuple
import h5py
import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import torch
from sklearn.preprocessing import StandardScaler
from torch_geometric.data import Data, Dataset
from torch_geometric.utils import subgraph
from utils.time_features_tool import time_features
from rich import print
from rich.progress import track
from datetime import datetime, timezone


class MadridDataset(Dataset):
    """
    A dataset for Madrid network data using PyTorch Geometric format.

    This dataset handles:
    - Static graph structure (distance matrix)
    - Dynamic node features (multiple sources with missing data)
    - Static node features
    """

    def __init__(
            self,
            distance_matrix_path: str,
            static_features_path: str,
            target_features: str,
            input_features: str,
            dynamic_features_paths: dict[str, str],
            input_window=12,
            look_back_window=0,
            pred_window=12,
            stride=1,
            split="train",
            train_ratio=0.6,
            val_ratio=0.2,
            train_unobs_ratio=0.05,
            permanent_unobs_ratio=0.1,
            threshold_distance=10,  # Distance threshold for adjacency matrix (km)
            seed4unob=42,
            not_scale=False,
            stack_time=False,
            add_node_list: list[int] = None,
            inf_exclude_num: int = 0,
            add_by_betweenness: int = 0,
            add_by_closeness: int = 0):
        """
        Initialize the MadridDataset.

        Args:
            distance_matrix_path (str): Path to distance matrix HDF5 file
            static_features_path (str): Path to static features HDF5 file
            dynamic_features_paths (dict[str, str]): Dictionary of paths to dynamic features HDF5 files
            input_window_size (int, optional): Size of the input temporal window. Defaults to 12.
            look_back_window_size (int, optional): Size of the loop back temporal window. Defaults to 0.
            pred_window_size (int, optional): Size of the output temporal window for prediction. Defaults to 12.
            stride (int, optional): Stride for sliding window. Defaults to 1.
            split (str, optional): 'train', 'val', or 'test'. Defaults to "train".
            train_ratio (float, optional): Ratio of training data. Defaults to 0.6.
            val_ratio (float, optional): Ratio of validation data. Defaults to 0.2.
            train_unobs_ratio (float, optional): Ratio of unobserved nodes in training data. Defaults to 0.05.
            permanent_unobs_ratio (float, optional): Ratio of unobserved nodes in permanent nodes. Defaults to 0.1.
            threshold_distance (int, optional): Threshold to create adjacency matrix from distance matrix. Defaults to 10.
            not_scale (bool, optional): Whether to scale the data. Defaults to False.
            stack_time (bool, optional): Whether to stack the time step dimension. Defaults to False.
            add_node_list (list[int], optional): List of add node ids. Defaults to None.
            inf_exclude_num (int, optional): Number of nodes to exclude for inference. Defaults to 0.
            add_by_betweenness (int, optional): Number of nodes to add by betweenness. Defaults to 0.
            add_by_closeness (int, optional): Number of nodes to add by closeness. Defaults to 0.
        """
        assert look_back_window >= 0 and look_back_window <= input_window, f"look_back_window_size {look_back_window} must be between 0 and {input_window}"
        super().__init__(log=False)

        self.distance_matrix_path = distance_matrix_path
        self.static_features_path = static_features_path
        self.dynamic_features_paths = dynamic_features_paths
        self.add_node_list = add_node_list
        self.inf_exclude_num = inf_exclude_num
        self.add_by_betweenness = add_by_betweenness
        self.add_by_closeness = add_by_closeness
        fea_dict = {
            'q': [0],
            'o': [1],
            'c': [2],
        }
        self.tar_feas = fea_dict[target_features]
        self.input_window = input_window
        self.look_back_window = look_back_window
        self.pred_window = pred_window
        self.total_window_size = input_window + pred_window
        self.stride = stride

        self.split = split
        self.train_ratio = train_ratio
        self.val_ratio = val_ratio

        self.train_unobs_ratio = train_unobs_ratio
        self.permanent_unobs_ratio = permanent_unobs_ratio
        self.seed4unobgen = seed4unob

        self.threshold_distance = threshold_distance
        self.not_scale = not_scale
        self.stack_time = stack_time
        # Initialize data structures
        self._init_data()

        # sensor_has_feature_mask = self.dynamic_info["sensor"]["has_feature_mask"].copy()
        # sensor_has_feature_mask[self.permanent_unobs_nodes_global_idx] = True
        # tom_has_feature_mask = self.dynamic_info["tomtom"]["has_feature_mask"]
        # tom_in_sensor = np.where((tom_has_feature_mask) & (sensor_has_feature_mask))[0]
        # tom_global_idx_to_source_idx = self.dynamic_info["tomtom"][
        #     "global_idx_to_source_idx"]
        # tom_in_sensor = tom_global_idx_to_source_idx[tom_in_sensor].astype(int)
        # tom_flow = self.dynamic_info["tomtom"]["data"][:, tom_in_sensor, 2]
        # max_tom_flow = np.nanmax(tom_flow, axis=0, keepdims=True)
        # self.dynamic_info['sensor'][
        #     'data'][:, :,
        #             -1] = self.dynamic_info['sensor']['data'][:, :, -1] / max_tom_flow

        self._scale_data()

        self.global_node_ids = torch.tensor(self.global_node_ids).long()
        self.full_graph = Data(
            edge_index=self.full_edge_index,
            edge_attr=self.full_weights,
            num_nodes=len(self.train_node_global_idx)
            if self.split != "test" else len(self.global_node_ids),
            n_idx=self.train_node_global_idx if self.split != "test" else torch.tensor(
                range(len(self.global_node_ids))),
            n_id=self.global_node_ids[self.train_node_global_idx]
            if self.split != "test" else self.global_node_ids,
        )

        self.causal_time_edges(self.input_window, self.pred_window + self.input_window)

    def _init_data(self):
        """
        Initialize dataset-specific data structures.

        This method:
        1. Builds or loads the graph structure
        2. Creates node ID mappings
        3. Identifies active and unobserved nodes
        4. Loads dynamic data information
        5. Sets up dataset splits
        """

        # Get all node information and create mappings
        with h5py.File(self.distance_matrix_path, "r") as f:
            self.global_node_ids = f["metadata/nodes"][:].astype(np.int64)

        # Create bidirectional mappings between physical IDs and continuous indices
        self.node_id_to_global_idx = {
            node_id: idx
            for idx, node_id in enumerate(self.global_node_ids)
        }
        self.global_idx_to_node_id = {
            idx: node_id
            for node_id, idx in self.node_id_to_global_idx.items()
        }

        # Set dataset size
        with h5py.File(self.dynamic_features_paths["sensor"], "r") as f:
            self.timestamps = f["metadata/times"][:]
            self.timestamps = pd.to_datetime(
                [ts.decode('utf-8') for ts in self.timestamps])
            self.num_timestamps = len(self.timestamps)
        self.train_start_idx = 0
        self.train_end_idx = int(self.num_timestamps *
                                 self.train_ratio)  # we keep this for scale data
        # set dataset split
        if self.split == "train":
            self.start_idx = 0
            self.end_idx = int(self.num_timestamps * self.train_ratio)
        elif self.split == "val":
            self.start_idx = int(self.num_timestamps * self.train_ratio)
            self.end_idx = int(self.num_timestamps * (self.train_ratio + self.val_ratio))
        else:
            self.start_idx = int(self.num_timestamps *
                                 (self.train_ratio + self.val_ratio))
            self.end_idx = self.num_timestamps

        self.split_timestamps = self.timestamps[self.start_idx:self.end_idx]
        self.valid_indices = []
        for i in range(
                0,
                len(self.split_timestamps) - self.input_window - self.pred_window + 1,
                self.stride):
            # window_timestamps = self.split_timestamps[i:i + self.input_window_size +
            #                                           self.pred_window_size]
            # if (window_timestamps[self.input_window_size -
            #                       self.look_back_window_size:].hour
            #         >= 22).any() or (window_timestamps[self.input_window_size -
            #                                            self.look_back_window_size:].hour
            #                          < 6).any():
            #     continue
            self.valid_indices.append(i)
        self.valid_indices = np.array(self.valid_indices)
        # Build or load the graph structure
        self._prepare_graph_data()
        # self.fourier_features = get_fourier(self.full_edge_index,
        #                                     self.full_weights,
        #                                     len(self.global_node_ids),
        #                                     k=32)
        # self.fourier_features = torch.from_numpy(self.fourier_features).float()

        # Get active nodes (nodes with fixed sensors)
        # and split active nodes into permanent unobserved and observable nodes
        with h5py.File(self.dynamic_features_paths["sensor"], "r") as f:
            self.active_node_ids = f["metadata/nodes"][:]
            # if self.tar_feas == [2]:
            # if True:  #TODO, remove this
            #     selected_nodes = np.load('MadridAllinOne/DLData/selected_nodes_frc2.npy')
            #     self.active_node_ids = np.array(
            #         [x for x in self.active_node_ids if x in selected_nodes])
            all_active_node_global_idx = np.array(
                [self.node_id_to_global_idx[nid] for nid in self.active_node_ids])
            num_permanent_unobs = int(
                len(all_active_node_global_idx) * self.permanent_unobs_ratio)
            np.random.seed(self.seed4unobgen)
            permanent_unobs_local_idx = np.random.choice(
                len(all_active_node_global_idx), size=num_permanent_unobs,
                replace=False)  # used to slect nodes in all_active_node_global_idx
            # For training phase: remove permanent unobserved nodes from active nodes
            mask = np.ones(len(all_active_node_global_idx), dtype=bool)
            mask[permanent_unobs_local_idx] = False
            if self.add_node_list is not None:
                mask[np.isin(
                    self.active_node_ids, np.array(self.add_node_list)
                )] = True  # some permanently unobserved nodes will be added manually, for the experiment of uncertainty-guided active sensing
            elif self.inf_exclude_num > 0:  # for the experiment of inference with addition unobs nodes, to see if uncertainty will be higher or not
                # Randomly select num_inf_exclude nodes with mask=True and set their mask to False.
                inf_exclude_local_idx = np.random.choice(np.where(mask)[0],
                                                         size=self.inf_exclude_num,
                                                         replace=False)
                mask[inf_exclude_local_idx] = False
            elif self.add_by_betweenness > 0:
                with h5py.File(self.static_features_path, "r") as f:
                    self.static_features = f["data/values"][:]
                    self.static_features_name = [
                        f.decode() for f in f["metadata/features"][:]
                    ]
                betweenness = self.static_features[:,
                                                   self.static_features_name.
                                                   index("BetweennessCentrality")]
                unobs_now_global_idx = all_active_node_global_idx[~mask]
                unobs_now_betweenness = betweenness[unobs_now_global_idx]
                add_unobs_global_idx = unobs_now_global_idx[np.argsort(
                    unobs_now_betweenness)[-self.add_by_betweenness:]]
                mask[np.isin(all_active_node_global_idx, add_unobs_global_idx)] = True
            elif self.add_by_closeness > 0:
                with h5py.File(self.static_features_path, "r") as f:
                    self.static_features = f["data/values"][:]
                    self.static_features_name = [
                        f.decode() for f in f["metadata/features"][:]
                    ]
                closeness = self.static_features[:,
                                                 self.static_features_name.
                                                 index("ClosenessCentrality")]
                unobs_now_global_idx = all_active_node_global_idx[~mask]
                unobs_now_closeness = closeness[unobs_now_global_idx]
                add_unobs_global_idx = unobs_now_global_idx[np.argsort(
                    unobs_now_closeness)[-self.add_by_closeness:]]
                mask[np.isin(all_active_node_global_idx, add_unobs_global_idx)] = True
            self.permanent_unobs_nodes_global_idx = all_active_node_global_idx[~mask]

        # For training phase: remove permanent unobserved nodes from active nodes
        # and filter out edges that related to permanent unobserved nodes
        if self.split != "test":
            self.active_node_global_idx = all_active_node_global_idx[mask]
            self.train_node_global_idx = torch.tensor(
                list(
                    set(self.global_idx_to_node_id.keys()) -
                    set(self.permanent_unobs_nodes_global_idx)))
            self.full_edge_index, self.full_weights = subgraph(self.train_node_global_idx,
                                                               self.full_edge_index,
                                                               self.full_weights,
                                                               relabel_nodes=True)
        else:
            # For test phase: use permanent unobserved nodes as inference targets
            self.active_node_global_idx = all_active_node_global_idx

        # Get static features
        with h5py.File(self.static_features_path, "r") as f:
            self.static_features = f["data/values"][:]
            self.static_features_name = [f.decode() for f in f["metadata/features"][:]]

        # Get dynamic data information
        self.dynamic_info = {}
        for data_source, path in self.dynamic_features_paths.items():
            with h5py.File(path, "r") as f:
                data = f["data/values"][self.start_idx:self.end_idx]
                if data_source == "sensor":
                    # type of data is numpy.ndarray, shape is (T,N,F), we need to replace the row with nan if the first feature is bigger than 300
                    data[data[:, :, 0] > 300, :] = np.nan
                    data[data[:, :, 0] < 10, :] = np.nan
                    # if self.tar_feas == [2]:  # congestion level
                    #     data[data[..., -1] <= 5, :] = np.nan
                elif data_source == "tomtom":
                    # type of data is numpy.ndarray, shape is (T,N,F), we need to replace the row with nan if the any of features is bigger than 120
                    data[(data[:, :, 4:] > 120).any(axis=2), :] = np.nan

                times = f["metadata/times"][self.start_idx:self.end_idx]
                features_name = [f.decode() for f in f["metadata/features"][:]]
                source_node_ids = f["metadata/nodes"][:]

                # Create mask and mapping for this data source
                has_feature_mask = np.zeros(len(self.global_node_ids), dtype=bool)
                global_idx_to_source_idx = np.full(
                    len(self.global_node_ids),
                    np.nan)  # np.nan denotes there is no source in this global ind
                source_node_id_to_source_idx = {
                    nid: idx
                    for idx, nid in enumerate(source_node_ids)
                }

                # create a mask for nodes that have source data and a mapping from global index to source index
                for global_idx, node_id in enumerate(self.global_node_ids):
                    if node_id in source_node_id_to_source_idx:
                        has_feature_mask[global_idx] = True
                        global_idx_to_source_idx[global_idx] = (
                            source_node_id_to_source_idx[node_id])

                # For training phase, mask out permanent unobserved nodes in sensor data
                if self.split != "test" and data_source == "sensor":
                    has_feature_mask[self.permanent_unobs_nodes_global_idx] = False
                    global_idx_to_source_idx[
                        self.permanent_unobs_nodes_global_idx] = np.nan
                if data_source == "tomtom":
                    # norm speed with speed limit
                    data[...,
                         4:] = data[..., 4:] / self.static_features[has_feature_mask][
                             ..., 4][np.newaxis, :, np.newaxis]
                self.dynamic_info[data_source] = {
                    "data": data,
                    "times": times,
                    "features_name": features_name,
                    "has_feature_mask": has_feature_mask,
                    "global_idx_to_source_idx": global_idx_to_source_idx,
                }

    def _prepare_graph_data(self) -> None:
        """
        Prepare and cache graph structure data.
        """
        base_path = Path(self.distance_matrix_path).parent
        self.graph_data_path = base_path.joinpath(
            f"train/graph_data_{self.threshold_distance}km.pt")
        self.graph_data_path.parent.mkdir(parents=True, exist_ok=True)
        edge_index_path = base_path.joinpath(
            f"train/edge_index_{self.threshold_distance}km.pt")
        weights_path = base_path.joinpath(f"train/weights_{self.threshold_distance}km.pt")

        if self.graph_data_path.exists():
            self.full_edge_index = torch.load(edge_index_path)
            self.full_weights = torch.load(weights_path)
            return

        print("Building and caching graph structure...")
        image_path_1 = base_path.joinpath(
            f"train/valid_distances_{self.threshold_distance}km.png")
        image_path_2 = base_path.joinpath(
            f"train/valid_weights_{self.threshold_distance}km.png")

        with h5py.File(self.distance_matrix_path, "r") as f:
            # Load and process one batch of distances
            distances = f["data/values"][:]
            # Find valid edges in this batch
            mask = distances <= self.threshold_distance
            rows, cols = np.where(mask)
            valid_distances = distances[rows, cols]
            self.sigma = valid_distances.std()
        plt.hist(valid_distances, bins=100)
        plt.figure(figsize=(8, 6))
        plt.hist(valid_distances, bins=50, color="blue", alpha=0.7, edgecolor="black")
        plt.title("Distribution of Valid Distances", fontsize=16)
        plt.xlabel("Distance", fontsize=14)
        plt.ylabel("Frequency", fontsize=14)
        plt.grid(True, linestyle="--", alpha=0.7)
        plt.savefig(image_path_1)
        plt.close()
        #  Define schema for parquet file
        schema = pa.schema([
            ("source_idx", pa.int64()),
            ("target_idx", pa.int64()),
            ("weight", pa.float32()),
        ])

        # Create parquet writer
        writer = pq.ParquetWriter(self.graph_data_path, schema)

        with h5py.File(self.distance_matrix_path, "r") as f:
            num_nodes = len(self.global_node_ids)
            batch_size = 1000  # Process 1000 nodes at a time

            for start_idx in track(range(0, num_nodes, batch_size),
                                   description="Processing distance matrix"):
                end_idx = min(start_idx + batch_size, num_nodes)

                # Load and process one batch of distances
                distances = f["data/values"][start_idx:end_idx]

                # Find valid edges in this batch
                mask = distances <= self.threshold_distance
                rows, cols = np.where(mask)

                if len(rows) > 0:
                    # Calculate weights for valid edges
                    valid_distances = distances[rows, cols]
                    weights = np.exp(
                        -(valid_distances**2) / (2 * self.sigma**2)
                    )  # TODO, It is possiable to use neural network to learn the weights

                    # Adjust row indices to account for batch offset
                    rows = rows + start_idx

                    # Create and write Arrow table
                    batch_table = pa.Table.from_arrays(
                        [
                            pa.array(rows),
                            pa.array(cols),
                            pa.array(weights, type=pa.float32()),
                        ],
                        schema=schema,
                    )

                    writer.write_table(batch_table)

        writer.close()

        # Only load edge connectivity data
        df = pd.read_parquet(self.graph_data_path)
        if len(df) > 0:
            self.full_edge_index = torch.tensor(df[["source_idx", "target_idx"]].values.T,
                                                dtype=torch.long)
            torch.save(self.full_edge_index, edge_index_path)
            self.full_weights = torch.tensor(df["weight"].values, dtype=torch.float)
            plt.figure(figsize=(8, 6))
            plt.hist(self.full_weights,
                     bins=50,
                     color="blue",
                     alpha=0.7,
                     edgecolor="black")
            plt.title("Distribution of Valid Weights", fontsize=16)
            plt.xlabel("Weight", fontsize=14)
            plt.ylabel("Frequency", fontsize=14)
            plt.grid(True, linestyle="--", alpha=0.7)
            plt.savefig(image_path_2)
            plt.close()

            torch.save(self.full_weights, weights_path)
            print(
                f"Graph built and cached with {len(self.global_node_ids)} nodes and {self.full_edge_index.size(1)} edges"
            )
        else:
            raise ValueError("No valid edges found in the graph")

    def causal_time_edges(self, t_k, t_q):
        """
        construct causal time edges
        Args:
            t_k: Number of historical time steps (T_k = 0, 1, ..., t_k-1)
            t_q: Number of query time steps (T_q = 0, 1, ..., t_q-1), where t_q >= t_k

        Returns:
            time_edge_index: [2, E_t] Temporal edge indices
        """
        edges = []
        assert t_q >= t_k
        for t_q_idx in range(t_q):
            max_t_k_idx = min(t_q_idx, t_k - 1)

            for t_k_idx in range(max_t_k_idx + 1):
                edges.append([t_q_idx, t_k_idx])
        time_edge_index = torch.tensor(edges).T  # [2, E_t]
        self.causal_time_edge_index = time_edge_index

        # Also construct temporal edge indices without causal constraints.
        self.full_time_edge_index = torch.tensor([[t_q_idx, t_k_idx]
                                                  for t_q_idx in range(t_q)
                                                  for t_k_idx in range(t_q)],
                                                 dtype=torch.long)

    def _scale_data(self):

        if self.not_scale:  # WE SHOULD FIT ON NODES THAT PERMANENTLY OBSEVED
            return
        save_path = Path(self.distance_matrix_path).parent.joinpath(
            f"train/scaler_train_ratio{int(self.train_ratio*100)}.pkl")
        save_path.parent.mkdir(parents=True, exist_ok=True)
        if save_path.exists():
            # static features
            static_scaler = joblib.load(save_path)["static_scaler"]
            # IQR scale ['Length','SpeedLimit','BetweennessCentrality','ClosenessCentrality','Heading','x','y']
            static_scale_idx = [
                self.static_features_name.index("Length"),
                self.static_features_name.index("SpeedLimit"),
                self.static_features_name.index("BetweennessCentrality"),
                self.static_features_name.index("ClosenessCentrality"),
                self.static_features_name.index("Heading"),
                self.static_features_name.index("x"),
                self.static_features_name.index("y"),
                self.static_features_name.index("poi_density"),
                self.static_features_name.index("work_density"),
                self.static_features_name.index("residential_density"),
                self.static_features_name.index("education_density"),
                self.static_features_name.index("medical_density"),
                self.static_features_name.index("food_density"),
                self.static_features_name.index("shopping_density"),
                self.static_features_name.index("entertainment_density"),
                self.static_features_name.index("transport_density"),
                self.static_features_name.index("other_density"),
                self.static_features_name.index("poi_diversity"),
                self.static_features_name.index("poi_mix_entropy"),
                self.static_features_name.index("Curviness"),
                self.static_features_name.index("CurvatureRatio"),
            ]
            self.static_features[:, static_scale_idx] = static_scaler.transform(
                self.static_features[:, static_scale_idx])
            # # min-max scale ['Curviness','CurvatureRatio']
            # static_scale_idx2 = [
            #     self.static_features_name.index("Curviness"),
            #     self.static_features_name.index("CurvatureRatio"),
            # ]
            # self.static_features[:, static_scale_idx2] = (
            #     self.static_features[:, static_scale_idx2] -
            #     self.static_features[:, static_scale_idx2].min(axis=0)) / (
            #         self.static_features[:, static_scale_idx2].max(axis=0) -
            #         self.static_features[:, static_scale_idx2].min(axis=0)) - 0.5
            self.static_features = torch.from_numpy(self.static_features).float()
            # dynamic features
            self.dynamic_scaler = joblib.load(save_path)["dynamic_scaler"]
            for data_source in self.dynamic_info.keys():
                self.dynamic_info[data_source]["data"] = torch.from_numpy(
                    self.dynamic_scaler[data_source].transform(
                        self.dynamic_info[data_source]["data"].reshape(
                            -1, len(self.dynamic_info[data_source]["features_name"])
                        )).reshape(self.dynamic_info[data_source]["data"].shape)).float()
            return
        # -----------------------------------------------------
        # Original static features distribution
        df = pd.DataFrame(self.static_features, columns=self.static_features_name)
        df.hist(bins=100, edgecolor="black", figsize=(15, 15))
        plt.title(
            f"Distribution of Static Features train_ratio{int(self.train_ratio*100)}")
        plt.xlabel("Feature Value")
        plt.ylabel("Frequency")
        fig_path = Path(self.distance_matrix_path).parent.joinpath(
            f"train/static_features_distribution_train_ratio{int(self.train_ratio*100)}.png"
        )
        plt.savefig(fig_path)
        plt.close()
        # -----------------------------------------------------
        # static features
        static_scaler = StandardScaler()  #Z-score scale
        static_scale_idx = [
            self.static_features_name.index("Length"),
            self.static_features_name.index("SpeedLimit"),
            self.static_features_name.index("BetweennessCentrality"),
            self.static_features_name.index("ClosenessCentrality"),
            self.static_features_name.index("Heading"),
            self.static_features_name.index("x"),
            self.static_features_name.index("y"),
            self.static_features_name.index("poi_density"),
            self.static_features_name.index("work_density"),
            self.static_features_name.index("residential_density"),
            self.static_features_name.index("education_density"),
            self.static_features_name.index("medical_density"),
            self.static_features_name.index("food_density"),
            self.static_features_name.index("shopping_density"),
            self.static_features_name.index("entertainment_density"),
            self.static_features_name.index("transport_density"),
            self.static_features_name.index("other_density"),
            self.static_features_name.index("poi_diversity"),
            self.static_features_name.index("poi_mix_entropy"),
            self.static_features_name.index("Curviness"),
            self.static_features_name.index("CurvatureRatio"),
        ]

        self.static_features[:, static_scale_idx] = static_scaler.fit_transform(
            self.static_features[:, static_scale_idx])
        # min-max scale ['Curviness','CurvatureRatio']
        # static_scale_idx2 = [
        #     self.static_features_name.index("Curviness"),
        #     self.static_features_name.index("CurvatureRatio"),
        # ]
        # self.static_features[:, static_scale_idx2] = (
        #     self.static_features[:, static_scale_idx2] -
        #     self.static_features[:, static_scale_idx2].min(axis=0)) / (
        #         self.static_features[:, static_scale_idx2].max(axis=0) -
        #         self.static_features[:, static_scale_idx2].min(axis=0)) - 0.5
        # -----------------------------------------------------
        # plot scaled static features distribution
        df = pd.DataFrame(self.static_features, columns=self.static_features_name)
        df.hist(bins=100, edgecolor="black", figsize=(15, 15))
        plt.title(
            f"Distribution of Static Features train_ratio{int(self.train_ratio*100)}")
        plt.xlabel("Feature Value")
        plt.ylabel("Frequency")
        fig_path = Path(self.distance_matrix_path).parent.joinpath(
            f"train/static_features_distribution_train_ratio{int(self.train_ratio*100)}_scaled.png"
        )
        plt.savefig(fig_path)
        plt.close()
        # -----------------------------------------------------
        self.static_features = torch.from_numpy(self.static_features).float()
        # -----------------------------------------------------
        print(
            f"Static features name: {[self.static_features_name[i] for i in static_scale_idx]}"
        )
        print(
            f"Static features scaled with scale: {static_scaler.scale_.round(1)} and center: {static_scaler.mean_.round(1)}"
        )
        # -----------------------------------------------------
        # dynamic features
        self.dynamic_scaler = {}
        for data_source, path in self.dynamic_features_paths.items():
            self.dynamic_scaler[data_source] = StandardScaler()
            with h5py.File(path, "r") as f:
                dynamic_features = f["data/values"][self.train_start_idx:
                                                    self.train_end_idx]  # (T, N, F)
                dynamic_features_name = [f.decode() for f in f["metadata/features"][:]]
                # change to (T*N, F)
                dynamic_features = dynamic_features.reshape(-1,
                                                            len(dynamic_features_name))
                self.dynamic_scaler[data_source].fit(dynamic_features)

            # -----------------------------------------------------
            # Original dynamic features distribution
            df = pd.DataFrame(
                self.dynamic_info[data_source]["data"].reshape(
                    -1, len(dynamic_features_name)),
                columns=dynamic_features_name,
            )
            df.hist(bins=100, edgecolor="black", figsize=(15, 15))
            plt.title(f"Distribution of {data_source} Dynamics Features")
            plt.xlabel("Feature Value")
            plt.ylabel("Frequency")
            fig_path = Path(self.distance_matrix_path).parent.joinpath(
                f"train/dynamic_features_distribution_{data_source}_train_ratio{int(self.train_ratio*100)}.png"
            )
            plt.savefig(fig_path)
            plt.close()
            # -----------------------------------------------------
            # Scaled dynamic features distribution

            self.dynamic_info[data_source]["data"] = (
                self.dynamic_scaler[data_source].transform(
                    self.dynamic_info[data_source]["data"].reshape(
                        -1, len(dynamic_features_name))).reshape(
                            self.dynamic_info[data_source]["data"].shape))

            df = pd.DataFrame(
                self.dynamic_info[data_source]["data"].reshape(
                    -1, len(dynamic_features_name)),
                columns=dynamic_features_name,
            )
            df.hist(bins=100, edgecolor="black", figsize=(15, 15))
            plt.title(f"Distribution of {data_source} Dynamics Features")
            plt.xlabel("Feature Value")
            plt.ylabel("Frequency")
            fig_path = Path(self.distance_matrix_path).parent.joinpath(
                f"train/dynamic_features_distribution_{data_source}_scaled_train_ratio{int(self.train_ratio*100)}.png"
            )
            plt.savefig(fig_path)
            plt.close()
            # -----------------------------------------------------
            self.dynamic_info[data_source]["data"] = torch.from_numpy(
                self.dynamic_info[data_source]["data"]).float()
            # -----------------------------------------------------
            print(f"Dynamic {data_source} features name: {dynamic_features_name}")
            print(
                f"Dynamic {data_source} features scaled with scale: {self.dynamic_scaler[data_source].scale_} and center: {self.dynamic_scaler[data_source].mean_}"
            )
            # -----------------------------------------------------

        # save scaler
        joblib.dump(
            {
                "static_scaler": static_scaler,
                "dynamic_scaler": self.dynamic_scaler,
            },
            save_path,
        )

    def inverse_transform(self, y):
        if not self.not_scale:
            y = y * self.scale + self.center
        return y

    def forward_transform(self, y):
        if not self.not_scale:
            y = (y - self.center) / self.scale
        return y

    def tom_inverse_transform(self, y):
        if not self.not_scale:
            y = y * self.tom_scale + self.tom_center
        return y

    def tom_forward_transform(self, y):
        if not self.not_scale:
            y = (y - self.tom_center) / self.tom_scale
        return y

    def len(self):
        """
        Return the number of examples in the dataset.

        Returns:
            int: Number of temporal windows in the dataset
        """
        return max(0, len(self.valid_indices))

    def __getitem__(
        self,
        idx: Tuple[int, List[int]],
    ):
        """
        Extend __getitem__ method to support spatial and temporal indexing.

        Args:
            idx (Tuple[int, List[int]]): A tuple containing:
                - idx[0] (int): temporal index
                - idx[1] (Data): pyg data object
        Returns:
            Data: A PyG data object containing the requested spatio-temporal data
        """
        data = self.get(self.indices()[idx[0]], idx[1])
        return data

    def get(self, t_idx: int, spatial_batch: Data) -> Data:
        """ method to get part of data from the dataset
        Args:
            t_idx (int): temporal index
            spatial_batch (Data): spatial indexes for unobserved nodes

        Returns:
            Data: A PyG data object containing:
                - x: Node features from fixed sensors
                - y: Target features within input and output window
                - edge_index: Edge connectivity
                - edge_attr: Edge weights
                - pos: Node positions
                - time: Temporal features
                - static_aux: Static auxiliary features
                - dynamic_aux: Dynamic auxiliary features
                - label_obs_mask: Mask for observed labels
                - label_unobs_mask: Mask for unobserved labels
                - graph_mask: Overall graph mask
                - num_nodes: Number of nodes in the subgraph
        """
        # Convert relative index to global index
        input_start_idx = self.valid_indices[t_idx]
        input_end_idx = input_start_idx + self.input_window
        output_start_idx = input_end_idx - self.look_back_window
        output_end_idx = input_end_idx + self.pred_window
        all_window = self.input_window + self.pred_window
        static_features = self.static_features[spatial_batch.n_idx].unsqueeze(1).repeat(
            1, all_window, 1)

        input_dynamic_features = {}
        for data_source in self.dynamic_info.keys():
            # inputs
            # Get valid input nodes and their mappings
            if data_source == "sensor":
                global_has_feature_mask = self.dynamic_info[data_source][
                    "has_feature_mask"].copy()
                global_has_feature_mask[spatial_batch.unobs_global_idx] = False
            else:
                global_has_feature_mask = self.dynamic_info[data_source][
                    "has_feature_mask"]  # tomtom is avaiable for these positions.

            subgraph_input_mask = np.atleast_1d(
                global_has_feature_mask[spatial_batch.n_idx])
            subgraph_source_input_idx = np.atleast_1d(
                self.dynamic_info[data_source]["global_idx_to_source_idx"][
                    spatial_batch.n_idx])[subgraph_input_mask]

            inputs = (torch.zeros((
                all_window,
                spatial_batch.num_nodes,
                len(self.dynamic_info[data_source]["features_name"]),
            )) * torch.nan)
            inputs[:self.input_window,
                   subgraph_input_mask] = self.dynamic_info[data_source]["data"][
                       input_start_idx:input_end_idx, subgraph_source_input_idx]
            inputs = inputs.permute(1, 0, 2)  # [N, T, C]

            input_dynamic_features.update({
                data_source: inputs,
            })
            if data_source == "tomtom":
                hits = self.dynamic_info[data_source]["data"][
                    input_start_idx:output_end_idx, subgraph_source_input_idx, 2:3]
                hits = hits.permute(1, 0, 2)  # [N, T, C]
                input_dynamic_features.update({
                    'tomtom_hits': hits,
                })

            # labels
            if data_source == "sensor":
                subgraph_label_mask = np.atleast_1d(
                    self.dynamic_info[data_source]["has_feature_mask"][
                        spatial_batch.n_idx])
                subgraph_source_label_idx = np.atleast_1d(
                    self.dynamic_info[data_source]["global_idx_to_source_idx"][
                        spatial_batch.n_idx])[subgraph_label_mask]

                labels = self.dynamic_info[data_source]["data"][
                    input_start_idx:output_end_idx, subgraph_source_label_idx,
                    self.tar_feas].permute(1,
                                           0).unsqueeze(-1)  # [N_sensor, T_in+T_out, C]
                subgraph_label_mask = torch.from_numpy(subgraph_label_mask).bool()
                subgraph_label_unobs_mask = torch.isin(spatial_batch.n_idx,
                                                       spatial_batch.unobs_global_idx)

                # time features
                time = self.dynamic_info[data_source]["times"][
                    input_start_idx:output_end_idx]
                decode_vec = np.vectorize(lambda x: x.decode())
                time = decode_vec(time)
                time_enc = time_features(time, freq="min")
                time_enc = (torch.from_numpy(time_enc).long().unsqueeze(0)
                            )  # Indicates a time stamp series for one graph

                # time id
                time_id = self.dynamic_info[data_source]["times"][
                    output_start_idx].decode()
                time_id = int(
                    datetime.strptime(
                        time_id,
                        "%Y-%m-%d %H:%M:%S").replace(tzinfo=timezone.utc).timestamp())
        # Create PyG data object
        data = Data(
            x=input_dynamic_features[
                "sensor"],  # Nodes monitored by fixed sensor[no unobserved nodes]
            y=labels,  # Target features: sensor data within output window
            tom_hits=input_dynamic_features["tomtom_hits"],
            edge_index=spatial_batch.edge_index,  # Edge connectivity
            edge_attr=spatial_batch.edge_attr,  # Edge weights
            pos=static_features[..., 12:14],
            # spectral_features=self.fourier_features[spatial_batch.n_idx],
            t_id=torch.tensor([time_id] * spatial_batch.num_nodes,
                              dtype=torch.int64).reshape(1, -1),
            n_id=spatial_batch.n_id,
            time_enc=time_enc.repeat(spatial_batch.num_nodes, 1, 1),
            static_aux=static_features[..., :12],
            poi_aux=static_features[..., [15, 17, 19, 21, 23, 25, 27, 29, 31, 33]],
            dynamic_aux=input_dynamic_features["tomtom"][
                ..., 2:],  # first two features [AvgTt, MedTt] are dropped
            label_unobs_mask=subgraph_label_unobs_mask,
            graph_mask=subgraph_label_mask,
            num_nodes=spatial_batch.num_nodes,
        )
        return data

    @property
    def scale(self):
        # NOTE: index with [self.tar_feas[0]] (a scalar), NOT [self.tar_feas]
        # (a length-1 list). The list form returns a float64 numpy *array*,
        # which promotes float32 tensors to float64 under `y * scale`,
        # doubling memory (OOM) and breaking torch.quantile's dtype check.
        return self.dynamic_scaler["sensor"].scale_[self.tar_feas[0]]

    @property
    def center(self):
        return self.dynamic_scaler["sensor"].mean_[self.tar_feas[0]]

    @property
    def tom_scale(self):
        return self.dynamic_scaler["tomtom"].scale_[2]

    @property
    def tom_center(self):
        return self.dynamic_scaler["tomtom"].mean_[2]

    def get_metadata(self):
        """
        Get dataset metadata.

        Returns:
            dict: Dictionary containing:
                - num_nodes: Number of active nodes
                - num_timestamps: Total number of timestamps
                - window_size: Size of temporal window
                - stride: Window stride
                - total_windows: Total number of windows
                - global_node_ids: List of all node IDs
                - {source}_features: Feature names for each data source
                - {source}_node_coverage: Number of nodes with data for each source
                - {source}_feature_dim: Feature dimension for each source
        """
        metadata = {
            "num_nodes": len(self.global_node_ids),
            "num_timestamps": self.num_timestamps,
            "window_size": self.total_window_size,
            "stride": self.stride,
            "total_windows": self.length,
        }

        # Add information for each data source
        for source, info in self.dynamic_info.items():
            metadata[f"{source}_features_name"] = info["features_name"]
            metadata[f"{source}_node_coverage"] = np.sum(info["has_feature_mask"])
            metadata[f"{source}_feature_dim"] = len(info["features_name"])

        return metadata

class MadridFSDataset4NP(MadridDataset):
    """Dataset that only model the road segment that has fixed sensors as graph nodes. organized for spatio-temporal neural processing

    Args:
        MadridDataset (MadridDataset): The original dataset that contains all road segments [segments that have at least one dynamicsource of data] as graph nodes.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        # only keep edges between active nodes (fixed sensors)
        if self.split != "test":
            # note, permanent unobserved nodes are already removed from active nodes in father class
            # so the globalid4subset is the same as active_node_global_idx
            sub_graph_node_id = torch.tensor(range(len(self.train_node_global_idx)))
            new_subset = sub_graph_node_id[torch.isin(
                self.train_node_global_idx,
                torch.from_numpy(self.active_node_global_idx))]
            self.full_edge_index, self.full_weights = subgraph(new_subset,
                                                               self.full_edge_index,
                                                               self.full_weights,
                                                               relabel_nodes=True)
            self.train_node_global_idx = torch.from_numpy(self.active_node_global_idx)

        else:
            # note, all nodes are keep in father class, so here we only need to sample subgraph from active nodes
            self.full_edge_index, self.full_weights = subgraph(torch.from_numpy(
                self.active_node_global_idx),
                                                               self.full_edge_index,
                                                               self.full_weights,
                                                               relabel_nodes=True)

        self.full_graph = Data(
            edge_index=self.full_edge_index,
            edge_attr=self.full_weights,
            num_nodes=len(self.active_node_global_idx),
            n_idx=torch.from_numpy(self.active_node_global_idx).long(),
            n_id=self.global_node_ids[self.active_node_global_idx].long(),
        )
