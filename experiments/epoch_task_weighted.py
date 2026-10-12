"""Task-weighted training epoch for the three manuscript subtasks.

This module intentionally mirrors :mod:`experiments.epoch_default`.  The only
training change is the ELBO construction: NLL is averaged independently for
observed-location forecasting (OF), unobserved-location estimation (UE), and
unobserved-location forecasting (UF), then combined using ``args.task_weights``.
Observed-location history (OH) remains in the objective with a fixed weight of
one; only the three evaluated subtasks are perturbed.
"""

from argparse import Namespace

import torch
from einops import rearrange
from rich.progress import track
from torch import nn
from torch.optim import Optimizer

from data_provider.loader import STDataLoader
from utils.log_tool import log_performance


TASK_NAMES = ("OF", "UE", "UF")


def build_task_masks(
    valid_mask: torch.Tensor,
    label_unobs_mask: torch.Tensor,
    input_window_size: int,
    look_back_window_size: int,
    pred_window_size: int,
) -> dict[str, torch.Tensor]:
    """Build masks aligned with targets shaped ``[batch*nodes, time, channels]``."""
    if valid_mask.ndim < 2:
        raise ValueError("valid_mask must have node and time dimensions")
    if label_unobs_mask.numel() != valid_mask.shape[0]:
        raise ValueError(
            "label_unobs_mask must contain one value per target node; "
            f"got {label_unobs_mask.numel()} for {valid_mask.shape[0]} targets"
        )
    if not 0 < look_back_window_size <= input_window_size:
        raise ValueError(
            "look_back_window_size must be positive and no larger than "
            "input_window_size"
        )
    required_steps = input_window_size + pred_window_size
    if valid_mask.shape[1] < required_steps:
        raise ValueError(
            f"targets contain {valid_mask.shape[1]} time steps, but the task "
            f"objective requires at least {required_steps}"
        )

    valid_mask = valid_mask.bool()
    node_shape = (valid_mask.shape[0],) + (1,) * (valid_mask.ndim - 1)
    unobs_nodes = label_unobs_mask.bool().reshape(node_shape)
    obs_nodes = ~unobs_nodes

    hist_start = input_window_size - look_back_window_size
    hist_end = input_window_size
    future_end = input_window_size + pred_window_size

    hist_window = torch.zeros_like(valid_mask)
    hist_window[:, hist_start:hist_end] = True
    future_window = torch.zeros_like(valid_mask)
    future_window[:, input_window_size:future_end] = True

    return {
        "OH": valid_mask & obs_nodes & hist_window,
        "OF": valid_mask & obs_nodes & future_window,
        "UE": valid_mask & unobs_nodes & hist_window,
        "UF": valid_mask & unobs_nodes & future_window,
    }


def task_weighted_beta_elbo(
    preds,
    targets: torch.Tensor,
    valid_mask: torch.Tensor,
    label_unobs_mask: torch.Tensor,
    task_weights,
    beta: float,
    input_window_size: int,
    look_back_window_size: int,
    pred_window_size: int,
) -> torch.Tensor:
    """Compute a sample-normalized, task-weighted NLL plus one KL term.

    A sampled mini-batch can contain no valid labels for one task because the
    traffic-data validity mask is applied after spatial sampling. Empty tasks
    are omitted for that batch and the remaining active weights are
    renormalized.
    """
    if len(task_weights) != len(TASK_NAMES):
        raise ValueError(
            f"task_weights must contain {len(TASK_NAMES)} values in OF, UE, UF order"
        )
    if any(weight < 0 for weight in task_weights) or sum(task_weights) <= 0:
        raise ValueError("task_weights must be non-negative with a positive sum")

    if isinstance(preds, tuple):
        preds_dist, prior, posterior = preds
    else:
        preds_dist = preds
        prior = posterior = None

    task_masks = build_task_masks(
        valid_mask=valid_mask,
        label_unobs_mask=label_unobs_mask,
        input_window_size=input_window_size,
        look_back_window_size=look_back_window_size,
        pred_window_size=pred_window_size,
    )
    safe_targets = torch.where(valid_mask, targets, torch.zeros_like(targets))
    nll_elements = -preds_dist.log_prob(safe_targets)

    weighted_nll = targets.new_zeros(())
    active_weight = 0.0
    weighted_tasks = (("OH", 1.0), *zip(TASK_NAMES, task_weights))
    for task_name, weight in weighted_tasks:
        if weight == 0:
            continue
        task_mask = task_masks[task_name]
        valid_count = task_mask.sum()
        if valid_count == 0:
            continue
        task_nll = (nll_elements * task_mask).sum() / valid_count
        weighted_nll = weighted_nll + float(weight) * task_nll
        active_weight += float(weight)
    if active_weight == 0:
        raise ValueError("no task has valid targets in the current batch")
    weighted_nll = weighted_nll / active_weight

    if prior is None or posterior is None:
        kl = targets.new_zeros(())
    else:
        kl = torch.distributions.kl_divergence(posterior, prior).mean(dim=0).sum()
    return weighted_nll + beta * kl


def shared_epoch(
    model: nn.Module,
    dataloader: STDataLoader,
    optimizer: Optimizer,
    criterions: dict[str, nn.Module],
    args: Namespace,
    mode: str,
    need_train: bool,
    epoch: int,
):
    # Save step-wise metrics
    metrics = {}
    for metric_name, criterion in criterions.items():
        if metric_name == "log_metrics":
            for sub_metric_name in criterion.log_metrics:
                metrics[sub_metric_name] = 0.0  # log metrics
        else:
            metrics[f'loss/{metric_name}'] = 0.0  # loss
    if mode == "train" and need_train:
        model.train()
        model.zero_grad()
        grad_enabled = True
    else:
        model.eval()
        grad_enabled = False

    # Save predictions

    total_y_true_trans = []
    total_y_pred_trans = []
    total_y_pred_sigma_trans = []
    total_y_pred_sigma_2_trans = []
    total_unobs_mask = []
    total_nodes_id = []
    total_time_id = []
    have_alea = True if args.output_path != "deterministic" else False
    have_epist = True if args.output_path == "nig" else False
    have_kl = True if args.np_path != "deterministic" else False
    inverse_y_pred_sigma = None
    inverse_y_pred_sigma_2 = None
    y_pred_sigma = None
    y_pred_sigma_2 = None
    with torch.set_grad_enabled(grad_enabled):
        max_bacth_num = (min(args.max_train_batch_num, len(dataloader))
                         if mode == "train" else len(dataloader))
        for batch_ix, batch in track(enumerate(dataloader),
                                     description=f"🚀🙏🍀 Processing {mode} epoch {epoch}",
                                     transient=True):
            if (~batch.label_unobs_mask).sum() == 0 and mode == "train":
                continue
            batch.to(args.device)
            batch.epoch = epoch
            y_mask_nan = ~batch.y.isnan()
            if torch.isnan(batch.y[~batch.label_unobs_mask][:, :args.input_window_size]
                           ).all() or torch.isnan(batch.y[~batch.label_unobs_mask][:, -(
                               args.look_back_window_size +
                               args.pred_window_size):]).all() and mode == "train":
                continue
            y = batch.y.clone()
            preds = model(batch)
            # Exclude samples whose observed history is entirely missing.
            context_y_mask = batch.y.isnan(
            )[~batch.label_unobs_mask][:, :args.input_window_size]
            context_y_mask = rearrange(context_y_mask,
                                       '(b n) t d -> b (n t) d',
                                       b=len(batch.t_id))
            all_invalid = context_y_mask.all(dim=1).squeeze(-1)
            y_mask_nan = rearrange(y_mask_nan,
                                   '(b n) t d -> b n t d',
                                   b=len(batch.t_id))
            y_mask_nan[all_invalid] = False
            y_mask_nan = rearrange(y_mask_nan,
                                   'b n t d -> (b n) t d',
                                   b=len(batch.t_id))

            loss = task_weighted_beta_elbo(
                preds=preds,
                targets=y,
                valid_mask=y_mask_nan,
                label_unobs_mask=batch.label_unobs_mask,
                task_weights=args.task_weights,
                beta=args.beta,
                input_window_size=args.input_window_size,
                look_back_window_size=args.look_back_window_size,
                pred_window_size=args.pred_window_size,
            )
            if have_alea:
                if have_kl and args.output_path == "normal":
                    y_pred_sigma = preds[0].scale.detach().cpu()
                    y_pred = preds[0].loc
                elif not have_kl and args.output_path == "normal":
                    y_pred_sigma = preds.scale.detach().cpu()
                    y_pred = preds.loc
                elif not have_kl and args.output_path == 'sqt':
                    y_pred = preds.mean(dim=-1)
                    y_pred_sigma = preds.detach().cpu(
                    )  # not, it actually is quantiles, not sigma, but we use this name to keep consistent
                elif not have_kl and args.output_path == 'nig':
                    y_pred = preds[0]
                    y_pred_sigma = torch.sqrt(
                        preds[-1] /
                        (preds[1] * (preds[-2] - 1))).detach().cpu()  # beta/((alpha-1)*v)
                    y_pred_sigma_2 = torch.sqrt(
                        (preds[-1] / (preds[-2] - 1))).detach().cpu()  # beta/(alpha-1)
                else:
                    raise NotImplementedError("Not implemented")
            else:
                y_pred = preds
            # add a clip loss for y_pred < threshold
            # loss += 100 * torch.relu(dataloader.transform(10) - y_pred).mean()
            if torch.isnan(loss):
                raise Exception("NaN loss")
            if torch.isinf(loss):
                raise Exception("Inf loss")
            y_pred = y_pred.detach().cpu()
            if grad_enabled:
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()
                optimizer.zero_grad()
            y = y.detach().cpu()
            # T denotes the batchsize, in our context, we collect mutiple samples along time dimension
            # t denotes the time steps, in our context, it means how many time steps we consider in each target
            y_from_shape = '(T n) t 1'
            y = rearrange(y,
                          f'{y_from_shape} -> (T n) t',
                          t=args.look_back_window_size + args.pred_window_size,
                          T=len(batch.t_id))
            y_pred = rearrange(y_pred,
                               f'{y_from_shape} -> (T n) t',
                               t=args.look_back_window_size + args.pred_window_size,
                               T=len(batch.t_id))

            if y.shape[1] == args.input_window_size + args.pred_window_size:
                y = y[:, args.input_window_size - args.look_back_window_size:]
                y_pred = y_pred[:, args.input_window_size - args.look_back_window_size:]

            if have_alea:
                if args.output_path != 'sqt':
                    y_pred_sigma = rearrange(y_pred_sigma,
                                             f'{y_from_shape} -> (T n) t',
                                             t=args.look_back_window_size +
                                             args.pred_window_size,
                                             T=len(batch.t_id))

                else:
                    y_pred_sigma = rearrange(y_pred_sigma,
                                             f'(T n) t 1 q -> (T n) t (1 q)',
                                             t=args.look_back_window_size +
                                             args.pred_window_size,
                                             T=len(batch.t_id))
                y_pred_sigma = y_pred_sigma[:, args.input_window_size -
                                            args.look_back_window_size:]
            if have_epist:
                y_pred_sigma_2 = rearrange(y_pred_sigma_2,
                                           f'{y_from_shape} -> (T n) t',
                                           t=args.look_back_window_size +
                                           args.pred_window_size,
                                           T=len(batch.t_id))
                y_pred_sigma_2 = y_pred_sigma_2[:, args.input_window_size -
                                                args.look_back_window_size:]
            label_unobs_mask = batch.label_unobs_mask.detach().cpu()
            inverse_y, inverse_y_pred = dataloader.inverse_transform(
                y), dataloader.inverse_transform(y_pred)
            if have_alea:
                inverse_y_pred_sigma = y_pred_sigma * dataloader.scale
            if have_epist:
                inverse_y_pred_sigma_2 = y_pred_sigma_2 * dataloader.scale
            # log
            if mode == "train":
                metrics = log_performance(
                    dataloader=dataloader,
                    criterions=criterions,
                    inverse_y=inverse_y,
                    inverse_y_pred=inverse_y_pred,
                    inverse_y_pred_sigma=inverse_y_pred_sigma,
                    inverse_y_pred_sigma_2=inverse_y_pred_sigma_2,
                    label_unobs_mask=label_unobs_mask,
                    args=args,
                    mode=mode,
                    metrics=metrics,
                    loss=loss,
                    max_bacth_num=max_bacth_num,
                    print_enabled=grad_enabled,
                    batch_ix=batch_ix,
                )
            total_y_true_trans.append(inverse_y)
            total_y_pred_trans.append(inverse_y_pred)
            total_unobs_mask.append(label_unobs_mask)
            total_nodes_id.append(batch.n_id.detach().cpu())
            total_time_id.append(batch.t_id.detach().cpu().reshape(-1))
            if have_alea:
                total_y_pred_sigma_trans.append(inverse_y_pred_sigma)
            if have_epist:
                total_y_pred_sigma_2_trans.append(inverse_y_pred_sigma_2)
            if (batch_ix + 1) >= max_bacth_num:
                break
    total_y = {
        "true_trans": torch.cat(total_y_true_trans, dim=0),
        "pred_trans": torch.cat(total_y_pred_trans, dim=0),
        "unobs_mask": torch.cat(total_unobs_mask, dim=0),
        "n_id": torch.cat(total_nodes_id, dim=0),
        "t_id": torch.cat(total_time_id, dim=0),
    }
    if have_alea:
        total_y.update({"pred_sigma_trans": torch.cat(total_y_pred_sigma_trans, dim=0)})
    if have_epist:
        total_y.update(
            {"pred_sigma_trans2": torch.cat(total_y_pred_sigma_2_trans, dim=0)})

    if mode == "train":
        for metric_name in metrics.keys():
            metrics[metric_name] /= batch_ix + 1  # average over batches
    # it is important to know that, the way to caculate train metrics is wrong,
    # because in each batch, the number of valid labels is different, so the metrics
    # from each batch can not be averaged [also rmse can not be averaged even number of valid samples are same in each batch],
    # we use the wrong way for train metrics, as it is more efficient, and it is enough for us to see the trending of train metrics
    # but for val and test, we use the correct way as followsto calculate metrics
    else:
        inverse_y, inverse_y_pred, unobs_mask = (
            total_y["true_trans"],
            total_y["pred_trans"],
            total_y["unobs_mask"],
        )
        metrics = log_performance(
            dataloader=dataloader,
            criterions=criterions,
            inverse_y=inverse_y,
            inverse_y_pred=inverse_y_pred,
            inverse_y_pred_sigma=total_y["pred_sigma_trans"] if have_alea else None,
            inverse_y_pred_sigma_2=total_y["pred_sigma_trans2"] if have_epist else None,
            label_unobs_mask=unobs_mask,
            args=args,
            mode=mode,
            metrics=metrics,
            loss=loss,
            max_bacth_num=max_bacth_num,
            print_enabled=grad_enabled,
            batch_ix=batch_ix,
        )
    return model, metrics, total_y
