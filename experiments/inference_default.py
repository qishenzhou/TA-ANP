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
import time


def default_inference(
    model: nn.Module,
    dataloader: STDataLoader,
    args: Namespace,
):
    have_alea = True if args.output_path != "deterministic" else False
    have_epist = True if args.output_path == "nig" else False
    have_kl = True if args.np_path != "deterministic" else False
    total_y_true_trans = []
    total_y_pred_trans = []
    total_unobs_mask = []
    total_nodes_id = []
    total_time_id = []
    total_y_pred_sigma_trans = []
    total_y_pred_sigma_2_trans = []

    args.best_total_y = {}  # save best epoch test results
    model.eval()
    total_t = 0
    start_t = time.time()
    with torch.no_grad():
        for batch in track(dataloader, description="🚀🙏🍀 Inferencing"):
            if (~batch.label_unobs_mask).sum() == 0:
                # continue
                raise Exception("No valid label")
            batch.to(args.device)
            if torch.isnan(
                    batch.y[~batch.label_unobs_mask]
                [:, :args.input_window_size]).all() or torch.isnan(
                    batch.y[~batch.label_unobs_mask][:, -(args.look_back_window_size +
                                                          args.pred_window_size):]).all():
                # continue
                raise Exception("No valid label")
            y = batch.y.clone()
            preds = model(batch)
            total_t += time.time() - start_t
            if have_alea:
                if not have_kl and args.output_path == "normal":
                    alea = preds.scale.detach().cpu()
                    y_pred = preds.loc
                elif have_kl and args.output_path == "normal":
                    alea = preds[0].scale.detach().cpu()
                    y_pred = preds[0].loc
                elif not have_kl and args.output_path == 'nig':
                    y_pred = preds[0]
                    alea = torch.sqrt(
                        preds[-1] /
                        (preds[1] * (preds[-2] - 1))).detach().cpu()  # beta/((alpha-1)*v)
                    epist = torch.sqrt(
                        (preds[-1] / (preds[-2] - 1))).detach().cpu()  # beta/(alpha-1)
                elif have_kl and args.output_path == 'nig':
                    raise NotImplementedError("Not implemented")
            else:
                y_pred = preds

            y = y.detach().cpu()
            y_pred = y_pred.detach().cpu()
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

            if have_alea:
                alea = rearrange(alea,
                                 f'{y_from_shape} -> (T n) t',
                                 t=args.look_back_window_size + args.pred_window_size,
                                 T=len(batch.t_id))
                alea = alea[:, args.input_window_size - args.look_back_window_size:]
            if have_epist:
                epist = rearrange(epist,
                                  f'{y_from_shape} -> (T n) t',
                                  t=args.look_back_window_size + args.pred_window_size,
                                  T=len(batch.t_id))
                epist = epist[:, args.input_window_size - args.look_back_window_size:]

            label_unobs_mask = batch.label_unobs_mask.detach().cpu()
            inverse_y, inverse_y_pred = dataloader.inverse_transform(
                y), dataloader.inverse_transform(y_pred)
            if have_alea:
                inverse_y_pred_sigma = alea * dataloader.scale
            if have_epist:
                inverse_y_pred_sigma_2 = epist * dataloader.scale
            total_y_true_trans.append(inverse_y)
            total_y_pred_trans.append(inverse_y_pred)
            total_unobs_mask.append(label_unobs_mask)
            total_nodes_id.append(batch.n_id.detach().cpu())
            total_time_id.append(batch.t_id.detach().cpu().reshape(-1))
            if have_alea:
                total_y_pred_sigma_trans.append(inverse_y_pred_sigma)
            if have_epist:
                total_y_pred_sigma_2_trans.append(inverse_y_pred_sigma_2)

    total_y = {
        "true_trans": torch.cat(total_y_true_trans, dim=0),
        "pred_trans": torch.cat(total_y_pred_trans, dim=0),
        "unobs_mask": torch.cat(total_unobs_mask, dim=0),
        "n_id": torch.cat(total_nodes_id, dim=0),
        "t_id": torch.cat(total_time_id, dim=0),
    }
    if have_alea:
        total_y["pred_sigma_trans"] = torch.cat(total_y_pred_sigma_trans, dim=0)
    if have_epist:
        total_y["pred_sigma_trans2"] = torch.cat(total_y_pred_sigma_2_trans, dim=0)

    args.best_total_y["inference"] = total_y
    print("-> save predictions and metrics...")
    print(f"save to {args.best_prediction_path}")
    if 'resilient' in args.sampler_class.lower():
        exp_id = args.exp_id + "resilient"
        save_name = "inference_total_y_resilient.pkl"
    elif args.add_node_number > 0:
        exp_id = args.exp_id + f"add_{args.add_node_number}"
        save_name = f"inference_total_y_add_{args.add_node_number}.pkl"
    elif args.inference_unobs_num > 0:
        exp_id = args.exp_id + f"inference_unobs_{args.inference_unobs_num}"
        save_name = f"inference_total_y_inference_unobs_{args.inference_unobs_num}.pkl"
    else:
        exp_id = args.exp_id
        save_name = "inference_total_y.pkl"
    read_out({"inference": dataloader}, exp_id, args)

    with open(args.best_prediction_path + save_name, "wb") as f:
        pickle.dump(args.best_total_y, f)
    print(f"Total inference time: {total_t:.2f}s")
    print(f"Average inference time per batch: {total_t/len(dataloader):.2f}s")
