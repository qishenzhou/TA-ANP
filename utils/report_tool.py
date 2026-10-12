import csv
import os
import pickle
import numpy as np
import pandas as pd
from pathlib import Path
from rich import print
import h5py
from utils.sci_plot import plot_multi_timeseries_shared_axes, plot_roadtype_r2, plot_node_r2, plot_node_type_bar, calibration_plot, caculate_qice, node_calibration_distribution, plot_road_uncertainty_analysis, plot_cv_vs_error, plot_z_distribution
from scipy.stats import norm
import matplotlib.pyplot as plt


def visual_part_results(args, data_name: str = "best_total_y.pkl"):
    # y_label = "Flow (veh/lane/hour)" if args.target_features == "q" else "Speed (km/h)" if args.target_features == "v" else "Occupancy (%)"
    y_label = "流量（辆/车道/小时）" if args.target_features == "q" else "速度（公里/小时）" if args.target_features == "v" else "拥堵指数（%）"

    max_y_global = 300 if args.target_features == "q" else 120 if args.target_features == "v" else 50
    # if 'madrid' in args.dataset_class.lower():
    #     y_label = "Flow (vehicles/lane/hour)"
    #     max_y_global = 300

    args.images_path = args.best_images_path + args.project_name + "/" + args.exp_id + "/"
    Path(args.images_path).mkdir(parents=True, exist_ok=True)
    if 'inference' in data_name:
        mode = 'inference'
    elif args.val_ratio > 0:
        mode = 'val'
    else:
        mode = 'train'
    all_horizon = args.look_back_window_size + args.pred_window_size
    df, has_alea, has_epist, has_tom_hits, to_obs_min, adj = read_results(
        args.best_prediction_path + data_name,
        mode=mode,
        horizons=all_horizon,
        distance_matrix_path=args.distance_matrix_path)
    if not 'pems' in args.dataset_class.lower():
        with h5py.File(args.static_features_path, "r") as f:
            static_features = f["data/values"][:]
            node_ids = f["metadata/nodes"][:]
            node2road = dict(zip(node_ids, static_features[:, 1]))
            df['roadtype'] = df.node.map(node2road)
    else:
        df['roadtype'] = 'Motorway'
    for horizon_name, horizon_len in [('hist', args.look_back_window_size),
                                      ('pred', args.pred_window_size)]:
        if horizon_len == 0:
            continue
        for mode, flag in [("unobs", True), ("obs", False)]:
            if horizon_name == 'hist':
                current_horizons = list(range(horizon_len))
            else:
                current_horizons = list(range(args.look_back_window_size, all_horizon))
            if mode == 'obs' and horizon_name == 'hist':
                continue
            output_dir = args.images_path + f"{mode}-{horizon_name}/"
            Path(output_dir).mkdir(parents=True, exist_ok=True)

            for h, horizon in enumerate(current_horizons):
                sub_df = df.loc[df.unobs_mask == flag].copy()
                sub_df2 = df.copy()
                keep_cols = ['node', 'node_idx', 'time', 'roadtype']
                for col in sub_df.columns:
                    if '_' in col and f'{horizon}' in col:
                        keep_cols.append(col)
                sub_df = sub_df[keep_cols]
                sub_df2 = sub_df2[keep_cols]
                if has_alea:
                    alea_p_values = sub_df[f'AleaP_{horizon}']
                    node_qice_scores = sub_df.groupby(
                        'node_idx')[f'AleaP_{horizon}'].apply(caculate_qice)
                    pred_values = sub_df2.groupby(
                        'node_idx')[f'Pred_{horizon}'].mean().values
                    pred_neighbor_diff_matrices = np.abs(pred_values[:, None] -
                                                         pred_values[None, :])
                    avg_mode_neighbor_diff = (adj * pred_neighbor_diff_matrices).sum(
                        axis=1)[node_qice_scores.index]
                    if mode == 'unobs':
                        unobs2obs = to_obs_min[node_qice_scores.index]
                    else:
                        unobs2obs = None
                    node_cv_scores = sub_df.groupby('node')[f'AleaCV_{horizon}'].mean()
                    all_node_cv_scores = sub_df2.groupby(
                        'node_idx')[f'AleaCV_{horizon}'].mean()
                    mode2all = adj[
                        node_qice_scores.index]  #ndarrdy [num_mode_nodes,num_all_nodes]
                    avg_mode_neighbor_cv = mode2all @ all_node_cv_scores.values.reshape(
                        -1, 1)
                    avg_mode_neighbor_cv[avg_mode_neighbor_cv == 0] = np.nan
                    relative_node_cv_scores = node_cv_scores.values - avg_mode_neighbor_cv.flatten(
                    )
                    cv_values = sub_df[f'AleaCV_{horizon}'].values
                    ape_values = sub_df[f'AE_{horizon}'].values / sub_df[
                        f'True_{horizon}'].values
                    plot_cv_vs_error(cv_values, ape_values,
                                     output_dir + f"horizon_{h}_alea_cv_vs_error.pdf")

                    z = (sub_df[f'True_{horizon}'] -
                         sub_df[f'Pred_{horizon}']) / sub_df[f'Alea_{horizon}']
                    plot_z_distribution(z, output_dir + f"horizon_{h}")
                    node_calibration_distribution(
                        node_qice_scores, node_cv_scores, unobs2obs,
                        relative_node_cv_scores, avg_mode_neighbor_diff,
                        output_dir + f"horizon_{h}_node_distribution_alea")
                    calibration_plot(alea_p_values,
                                     output_dir + f"horizon_{h}_calibration_alea")
                    if has_epist:
                        epist_p_values = sub_df[f'EpistP_{horizon}']
                        node_qice_scores = sub_df.groupby(
                            'node')[f'EpistP_{horizon}'].apply(caculate_qice)
                        node_cv_scores = sub_df.groupby(
                            'node')[f'EpistCV_{horizon}'].mean()
                        all_node_cv_scores = sub_df2.groupby(
                            'node_idx')[f'EpistCV_{horizon}'].mean()
                        avg_mode_neighbor_cv = mode2all @ all_node_cv_scores.values.reshape(
                            -1, 1)
                        avg_mode_neighbor_cv[avg_mode_neighbor_cv == 0] = np.nan
                        relative_node_cv_scores = node_cv_scores.values - avg_mode_neighbor_cv.flatten(
                        )
                        cv_values = sub_df[f'EpistCV_{horizon}'].values
                        ape_values = sub_df[f'AE_{horizon}'].values / sub_df[
                            f'True_{horizon}'].values
                        plot_cv_vs_error(
                            cv_values, ape_values,
                            output_dir + f"horizon_{h}_epist_cv_vs_error.pdf")
                        node_calibration_distribution(
                            node_qice_scores, node_cv_scores, unobs2obs,
                            relative_node_cv_scores, avg_mode_neighbor_diff,
                            output_dir + f"horizon_{h}_node_distribution_epist")
                        calibration_plot(epist_p_values,
                                         output_dir + f"horizon_{h}_calibration_epist")

                        total_uncertainty_p_values = sub_df[f'TUP_{horizon}']
                        node_qice_scores = sub_df.groupby('node')[f'TUP_{horizon}'].apply(
                            caculate_qice)
                        node_cv_scores = sub_df.groupby('node')[f'TUCV_{horizon}'].mean()
                        all_node_cv_scores = sub_df2.groupby(
                            'node_idx')[f'TUCV_{horizon}'].mean()
                        avg_mode_neighbor_cv = mode2all @ all_node_cv_scores.values.reshape(
                            -1, 1)
                        avg_mode_neighbor_cv[avg_mode_neighbor_cv == 0] = np.nan
                        relative_node_cv_scores = node_cv_scores.values - avg_mode_neighbor_cv.flatten(
                        )
                        cv_values = sub_df[f'TUCV_{horizon}'].values
                        ape_values = sub_df[f'AE_{horizon}'].values / sub_df[
                            f'True_{horizon}'].values
                        plot_cv_vs_error(
                            cv_values, ape_values,
                            output_dir + f"horizon_{h}_total_uncertainty_cv_vs_error.pdf")
                        node_calibration_distribution(
                            node_qice_scores, node_cv_scores, unobs2obs,
                            relative_node_cv_scores, avg_mode_neighbor_diff, output_dir +
                            f"horizon_{h}_node_distribution_total_uncertainty")
                        calibration_plot(
                            total_uncertainty_p_values,
                            output_dir + f"horizon_{h}_calibration_total_uncertainty")
                    # Plot uncertainty analysis for all road types with subplots
                    if 'madrid' in args.dataset_class.lower():
                        plot_road_uncertainty_analysis(sub_df, horizon,
                                                       output_dir + f"horizon_{h}",
                                                       has_epist)

                plot_roadtype_r2(sub_df, output_dir + f"horizon_{h}_roadtype_r2.pdf",
                                 horizon)
                bad_nodes = plot_node_r2(sub_df, output_dir + f"horizon_{h}_node_r2.pdf",
                                         horizon)

                all_nodes = set(sub_df.node.unique())
                good_nodes = all_nodes - bad_nodes
                num_cols = 5
                num_rows = 10

                plot_node_type_bar([('good', list(good_nodes)), ('bad', list(bad_nodes))],
                                   output_dir + f"horizon_{h}_node_type_bar.pdf")
                for nodes_type, all_nodes in [('bad', list(bad_nodes)),
                                              ('good', list(good_nodes))]:
                    nodes_out_dir = output_dir + f"{nodes_type}/"
                    Path(nodes_out_dir).mkdir(parents=True, exist_ok=True)
                    for date in sub_df.time.dt.date.unique()[1:3]:
                        # Split nodes into batches of num_rows * num_cols nodes.
                        for plot_start in range(0, min(len(all_nodes), 100),
                                                num_rows * num_cols):
                            # Create a new mappings collection for each batch.
                            mappings = []
                            # Get the nodes in the current batch.
                            plot_nodes = all_nodes[plot_start:plot_start +
                                                   num_rows * num_cols]

                            # Process each node in the current batch.
                            for i, node in enumerate(plot_nodes):
                                # Get data for the current node.
                                node_data = sub_df.loc[(sub_df.time.dt.date == date)
                                                       & (sub_df.node == node)]
                                if len(node_data) == 0:
                                    continue

                                # Ensure the data are NumPy arrays.
                                true_sample = node_data.loc[:, f'True_{horizon}'].values
                                pred_sample = node_data.loc[:, f'Pred_{horizon}'].values
                                if has_alea:
                                    alea_sample = node_data.loc[:,
                                                                f'Alea_{horizon}'].values
                                    if has_epist:
                                        epist_sample = node_data.loc[:,
                                                                     f'Epist_{horizon}'].values
                                    else:
                                        epist_sample = None
                                else:
                                    alea_sample = None
                                    epist_sample = None
                                title = f'传感器 #{node}'
                                # Process time data.
                                time_sample = np.array(
                                    node_data.time.dt.time.astype(str).apply(
                                        lambda x: x[:-3]).values)

                                # Check data validity.
                                if len(true_sample) == 0 or len(pred_sample) == 0 or len(
                                        time_sample) == 0:
                                    continue

                                # Add to mappings.
                                mappings.append((i // num_cols, i % num_cols, [{
                                    'x':
                                    time_sample,
                                    'y':
                                    pred_sample,
                                    'epist':
                                    epist_sample,
                                    'alea':
                                    alea_sample,
                                    'label':
                                    '推断值',
                                    'linestyle':
                                    '-',
                                    'color':
                                    '#e6739a'
                                }, {
                                    'x':
                                    time_sample,
                                    'y':
                                    true_sample,
                                    'label':
                                    '观测值',
                                    'color':
                                    '#046c9a',
                                    'linestyle':
                                    '-'
                                }], {
                                    'title': title
                                }))
                            try:
                                plot_multi_timeseries_shared_axes(
                                    nrows=num_rows,
                                    ncols=num_cols,
                                    default_xlabel=None,
                                    max_y_global=max_y_global,
                                    default_ylabel=y_label,
                                    subplot_data_mapping=mappings,
                                    subplot_aspect_ratio=1.7,
                                    figure_width=6.4,
                                    axis_limit_padding=0.1,
                                    num_xticks=2,
                                    subplot_hspace=0.3,
                                    legend_bbox_to_anchor=(0.5, 0.02),
                                    label_pad=0.05,
                                    save_path=nodes_out_dir +
                                    f"date_{date}_horizon_{h}_nodes_{plot_start}_to_{plot_start + len(plot_nodes)}_timeseries",
                                    file_format="pdf")

                            except Exception as e:
                                print(
                                    f"Error plotting batch {plot_start} to {plot_start + len(plot_nodes)}: {str(e)}"
                                )
                                continue
                            finally:
                                # Ensure all Matplotlib resources are released.
                                plt.close('all')
                                import gc
                                gc.collect()


def read_results(path: str,
                 mode: str,
                 horizons: int,
                 distance_matrix_path: str,
                 static_feature_path=None) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Read the results from the path

    Args:
        path (str): The path to the results6

    Returns:
        tuple[pd.DataFrame, pd.DataFrame]: The predictions and the true values
    """
    total_y = np.load(path, allow_pickle=True)[mode]
    true = total_y['true_trans']
    preds = total_y['pred_trans']
    unobs_mask = total_y['unobs_mask']
    t_id = total_y['t_id']
    n_id = total_y['n_id']  #FIXME this is not sensor id, but global node id from edges
    has_alea = False
    has_epist = False
    has_tom_hits = False
    if 'pred_sigma_trans' in total_y.keys():
        alea = total_y['pred_sigma_trans']
        has_alea = True
        if "pred_sigma_trans2" in total_y.keys():
            epist = total_y['pred_sigma_trans2']
            has_epist = True
    if 'tom_true_trans' in total_y.keys():
        tom_hits = total_y['tom_true_trans']
        has_tom_hits = True
    absolute_error = np.abs(true.numpy() - preds.numpy())
    true = pd.DataFrame(true.numpy(), columns=[f'True_{i}' for i in range(horizons)])
    preds = pd.DataFrame(preds.numpy(), columns=[f'Pred_{i}' for i in range(horizons)])
    absolute_error = pd.DataFrame(absolute_error,
                                  columns=[f'AE_{i}' for i in range(horizons)])
    data = pd.concat([true, preds, absolute_error], axis=1)
    data['unobs_mask'] = unobs_mask.numpy()
    data['node'] = n_id.numpy()
    data['time'] = pd.to_datetime(t_id.numpy(), unit='s',
                                  utc=True).strftime("%Y-%m-%d %H:%M:%S")
    data['time'] = pd.to_datetime(data['time'])
    if has_alea:
        alea = pd.DataFrame(alea.numpy(), columns=[f'Alea_{i}' for i in range(horizons)])
        data = pd.concat([data, alea], axis=1)
        alea_p_values = norm.cdf(true.values, loc=preds.values, scale=alea.values)
        alea_p_values = pd.DataFrame(alea_p_values,
                                     columns=[f'AleaP_{i}' for i in range(horizons)])
        data = pd.concat([data, alea_p_values], axis=1)
        safe_preds = preds.values.copy()
        safe_preds[safe_preds < 5] = 5
        alea_cv = alea.values / safe_preds
        alea_cv = pd.DataFrame(alea_cv, columns=[f'AleaCV_{i}' for i in range(horizons)])
        alea_cv = alea_cv.replace([np.inf, -np.inf], np.nan)
        data = pd.concat([data, alea_cv], axis=1)
        if has_epist:
            epist = pd.DataFrame(epist.numpy(),
                                 columns=[f'Epist_{i}' for i in range(horizons)])
            data = pd.concat([data, epist], axis=1)
            epist_p_values = norm.cdf(true.values, loc=preds.values, scale=epist.values)
            epist_p_values = pd.DataFrame(
                epist_p_values, columns=[f'EpistP_{i}' for i in range(horizons)])
            data = pd.concat([data, epist_p_values], axis=1)
            epist_cv = epist.values / preds.values
            epist_cv = pd.DataFrame(epist_cv,
                                    columns=[f'EpistCV_{i}' for i in range(horizons)])
            epist_cv = epist_cv.replace([np.inf, -np.inf], np.nan)
            data = pd.concat([data, epist_cv], axis=1)

            total_uncertainty = (epist.values**2 + alea.values**2)**0.5
            total_uncertainty = pd.DataFrame(total_uncertainty,
                                             columns=[f'TU_{i}' for i in range(horizons)])
            data = pd.concat([data, total_uncertainty], axis=1)
            total_uncertainty_p_values = norm.cdf(true.values,
                                                  loc=preds.values,
                                                  scale=total_uncertainty.values)
            total_uncertainty_p_values = pd.DataFrame(
                total_uncertainty_p_values, columns=[f'TUP_{i}' for i in range(horizons)])
            data = pd.concat([data, total_uncertainty_p_values], axis=1)
            total_uncertainty_cv = total_uncertainty.values / preds.values
            total_uncertainty_cv = pd.DataFrame(
                total_uncertainty_cv, columns=[f'TUCV_{i}' for i in range(horizons)])
            total_uncertainty_cv = total_uncertainty_cv.replace([np.inf, -np.inf], np.nan)
            data = pd.concat([data, total_uncertainty_cv], axis=1)
    if has_tom_hits:
        tom_hits = pd.DataFrame(tom_hits.numpy(),
                                columns=[f'TomHits_{i}' for i in range(horizons)])
        data = pd.concat([data, tom_hits], axis=1)
        tom_p = tom_hits.values / true.values
        tom_p = pd.DataFrame(tom_p, columns=[f'TomP_{i}' for i in range(horizons)])
        data = pd.concat([data, tom_p], axis=1)
    data = data.sort_values(by=['time', 'node'])
    with h5py.File(distance_matrix_path, "r") as f:
        distance_matrix = f['data/values'][:]
        nodes_id = f['metadata/nodes'][:]
        nodes_id = nodes_id.astype(np.int64)
    if static_feature_path is not None:
        with h5py.File(static_feature_path, "r") as f:
            static_features_name = [f.decode() for f in f['metadata/features'][:]]
            x = f['data/values'][:static_features_name.index("x")]
            y = f['data/values'][:static_features_name.index("y")]
    distance_matrix[distance_matrix > 30] = np.nan
    nid2idx = {nid: idx for idx, nid in enumerate(nodes_id)}
    data['node_idx'] = data.node.map(nid2idx)
    if static_feature_path is not None:
        data['x'] = x[data['node_idx']]
        data['y'] = y[data['node_idx']]
    used_nodes = sorted(data.node_idx.unique())
    distance_matrix = distance_matrix[used_nodes][:, used_nodes]
    oidx2nidx = {old_idx: new_idx for new_idx, old_idx in enumerate(used_nodes)}
    data['node_idx'] = data.node_idx.map(oidx2nidx)
    obs_nodes_idx = data.loc[data.unobs_mask == False].node_idx.unique()
    to_obs_min = np.nanmin(distance_matrix[:, obs_nodes_idx], axis=1)
    adj = distance_matrix.copy()
    if 'madrid' in distance_matrix_path.lower():
        adj[(adj != 0) & (adj <= 1)] = 1
        adj[np.isnan(adj)] = 0
        adj[adj > 1] = 0
        d = adj.sum(axis=1, keepdims=True)
        d[d == 0] = 1
        adj = adj / d
    else:
        raise ValueError(f"distance_matrix_path {distance_matrix_path} is not valid")
    return data, has_alea, has_epist, has_tom_hits, to_obs_min, adj
