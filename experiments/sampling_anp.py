"""
Training functions and helpers
"""

import pickle
from argparse import Namespace
import torch
from torch import nn
from data_provider.loader import STDataLoader
from .test import read_out
from rich import print
from rich.progress import track
from einops import rearrange
from itertools import chain
import time


def sampling_inference(
    model: nn.Module,
    dataloader: STDataLoader,
    args: Namespace,
):
    total_y_true_trans = []
    total_y_pred_trans = []
    total_unobs_mask = []
    total_nodes_id = []
    total_time_id = []
    total_y_pred_sigma_trans2 = []
    total_y_pred_sigma_trans = []
    args.best_total_y = {}  # save best epoch test results
    model.eval()
    total_t = 0
    start_t = time.time()
    with torch.no_grad():
        for batch in track(dataloader, description="🚀🙏🍀 Sampling"):
            if (~batch.label_unobs_mask).sum() == 0:
                # continue
                raise Exception("No valid label")
            batch.to(args.device)
            y_mask_nan = ~batch.y.isnan()
            if torch.isnan(
                    batch.y[~batch.label_unobs_mask]
                [:, :args.input_window_size]).all() or torch.isnan(
                    batch.y[~batch.label_unobs_mask][:, -(args.look_back_window_size +
                                                          args.pred_window_size):]).all():
                raise Exception("No valid label")
                # continue
            y = batch.y.clone()
            y_pred, alea, epist = model(batch, num_sampling=args.num_sampling)
            total_t += time.time() - start_t
            y_pred = y_pred.detach().cpu()
            epist = epist.detach().cpu()
            alea = alea.detach().cpu()
            y = y.detach().cpu()
            y_mask_nan = y_mask_nan.detach().cpu()
            # T denotes the batchsize, in our context, we collect mutiple samples along time dimension
            # t denotes the time steps, in our context, it means how many time steps we consider in each target
            y_from_shape = '(T n) t 1'
            y = rearrange(y,
                          f'{y_from_shape} -> (T n) t',
                          t=args.input_window_size + args.pred_window_size,
                          T=len(batch.t_id))
            y = y[:, args.input_window_size - args.look_back_window_size:]
            y_pred = rearrange(y_pred,
                               f'{y_from_shape} -> (T n) t',
                               t=args.input_window_size + args.pred_window_size,
                               T=len(batch.t_id))
            y_pred = y_pred[:, args.input_window_size - args.look_back_window_size:]
            epist = rearrange(epist,
                              f'{y_from_shape} -> (T n) t',
                              t=args.input_window_size + args.pred_window_size,
                              T=len(batch.t_id))
            epist = epist[:, args.input_window_size - args.look_back_window_size:]
            alea = rearrange(alea,
                             f'{y_from_shape} -> (T n) t',
                             t=args.input_window_size + args.pred_window_size,
                             T=len(batch.t_id))
            alea = alea[:, args.input_window_size - args.look_back_window_size:]
            label_unobs_mask = batch.label_unobs_mask.detach().cpu()
            inverse_y, inverse_y_pred = dataloader.inverse_transform(
                y), dataloader.inverse_transform(y_pred)
            epist = epist * dataloader.scale
            alea = alea * dataloader.scale
            total_y_true_trans.append(inverse_y)
            total_y_pred_trans.append(inverse_y_pred)
            total_unobs_mask.append(label_unobs_mask)
            total_nodes_id.append(batch.n_id.detach().cpu())
            total_time_id.append(batch.t_id.detach().cpu().reshape(-1))
            total_y_pred_sigma_trans.append(alea)
            total_y_pred_sigma_trans2.append(epist)
            start_t = time.time()
    total_y = {
        "true_trans": torch.cat(total_y_true_trans, dim=0),
        "pred_trans": torch.cat(total_y_pred_trans, dim=0),
        "unobs_mask": torch.cat(total_unobs_mask, dim=0),
        "n_id": torch.cat(total_nodes_id, dim=0),
        "t_id": torch.cat(total_time_id, dim=0),
        "pred_sigma_trans": torch.cat(total_y_pred_sigma_trans, dim=0),
        "pred_sigma_trans2": torch.cat(total_y_pred_sigma_trans2, dim=0),
    }
    args.best_total_y["inference"] = total_y
    print("-> save predictions and metrics...")
    if 'resilient' in args.sampler_class.lower():
        exp_id = args.exp_id + "resilient"
        save_name = "inference_total_y_resilient.pkl"
    elif args.add_node_number > 0 and not args.random_add_node:
        exp_id = args.exp_id + f"add_{args.add_node_number}_{args.sorted_nodes_path.split('/')[-1].split('-')[1].split('.')[0]}"
        save_name = f"inference_total_y_add_{args.add_node_number}_{args.sorted_nodes_path.split('/')[-1].split('-')[1].split('.')[0]}.pkl"
    elif args.random_add_node and args.add_node_number > 0:
        exp_id = args.exp_id + f"random_add_{args.add_node_number}"
        save_name = f"inference_total_y_random_add_{args.add_node_number}.pkl"
    elif args.inference_unobs_num > 0:
        exp_id = args.exp_id + f"inference_unobs_inf_exclude_{args.inference_unobs_num}"
        save_name = f"inference_total_y_inference_unobs_inf_exclude_{args.inference_unobs_num}.pkl"
    elif args.add_by_betweenness > 0:
        exp_id = args.exp_id + f"add_by_betweenness_{args.add_by_betweenness}"
        save_name = f"inference_total_y_add_by_betweenness_{args.add_by_betweenness}.pkl"
    elif args.add_by_closeness > 0:
        exp_id = args.exp_id + f"add_by_closeness_{args.add_by_closeness}"
        save_name = f"inference_total_y_add_by_closeness_{args.add_by_closeness}.pkl"
    else:
        exp_id = args.exp_id
        save_name = "inference_total_y.pkl"

    save_name = save_name.removesuffix(".pkl") + f"_K{args.num_sampling}.pkl"
    result_exp_id = f"{exp_id}_K{args.num_sampling}"
    args.inference_data_name = save_name

    read_out({"inference": dataloader}, result_exp_id, args)
    print(f"save to {args.best_prediction_path + save_name}")
    with open(args.best_prediction_path + save_name, "wb") as f:
        pickle.dump(args.best_total_y, f)
    print(f"Total sampling time: {total_t:.2f}s")
    print(f"Average sampling time per batch: {total_t/len(dataloader):.2f}s")
