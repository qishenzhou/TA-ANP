from argparse import Namespace

import torch
from torch import nn
from torch.optim import Optimizer

from data_provider.loader import STDataLoader
from utils.log_tool import log_performance2
from rich.progress import track
from einops import rearrange


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
    total_y_true = []
    total_alpha = []
    total_beta = []
    with torch.set_grad_enabled(grad_enabled):
        max_bacth_num = (min(args.max_train_batch_num, len(dataloader))
                         if mode == "train" else len(dataloader))
        for batch_ix, batch in track(
                enumerate(dataloader),
                description=f"🚀🙏🍀 Processing {mode} epoch {epoch}",
                transient=True):
            if (~batch.label_unobs_mask).sum() == 0:
                continue
            batch.to(args.device)
            y_mask_nan = ~batch.y.isnan()
            if torch.isnan(batch.y[~batch.label_unobs_mask]
                           [:, :args.input_window_size]).all() or torch.isnan(
                               batch.y[~batch.label_unobs_mask]
                               [:, -(args.look_back_window_size +
                                     args.pred_window_size):]).all():
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

            loss = criterions[args.loss](preds, y, y_mask_nan, beta=args.beta)

            alpha = preds[0].concentration.detach().cpu(
            )  #alpha in gamma distribution
            beta = preds[0].rate.detach().cpu()  #beta in gamma distribution
            y_pred = preds[0].mean

            # add a clip loss for y_pred < threshold
            # loss += 100 * torch.relu(dataloader.transform(10) - y_pred).mean()
            if torch.isnan(loss):
                raise Exception("NaN loss")
            if torch.isinf(loss):
                raise Exception("Inf loss")
            if grad_enabled:
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(),
                                               max_norm=1.0)
                optimizer.step()
                optimizer.zero_grad()
            y_pred = y_pred.detach().cpu()
            y = y.detach().cpu()
            # T denotes the batchsize, in our context, we collect mutiple samples along time dimension
            # t denotes the time steps, in our context, it means how many time steps we consider in each target
            y_from_shape = '(T n) t 1'
            y = rearrange(y,
                          f'{y_from_shape} -> (T n) t',
                          t=args.look_back_window_size + args.pred_window_size,
                          T=len(batch.t_id))
            y = y[:, args.input_window_size - args.look_back_window_size:]
            y_pred = rearrange(y_pred,
                               f'{y_from_shape} -> (T n) t',
                               t=args.look_back_window_size +
                               args.pred_window_size,
                               T=len(batch.t_id))
            y_pred = y_pred[:, args.input_window_size -
                            args.look_back_window_size:]

            alpha = rearrange(alpha,
                              f'{y_from_shape} -> (T n) t',
                              t=args.look_back_window_size +
                              args.pred_window_size,
                              T=len(batch.t_id))
            alpha = alpha[:,
                          args.input_window_size - args.look_back_window_size:]
            beta = rearrange(beta,
                             f'{y_from_shape} -> (T n) t',
                             t=args.look_back_window_size +
                             args.pred_window_size,
                             T=len(batch.t_id))
            beta = beta[:,
                        args.input_window_size - args.look_back_window_size:]
            gamma_dist = torch.distributions.Gamma(alpha, beta)
            label_unobs_mask = batch.label_unobs_mask.detach().cpu()
            inverse_y, inverse_y_pred = dataloader.inverse_transform(
                y), dataloader.inverse_transform(y_pred)

            low_bound = gamma_dist.icdf(torch.tensor(0.025))
            up_bound = gamma_dist.icdf(torch.tensor(0.975))
            # log
            if mode == "train":
                metrics = log_performance2(
                    dataloader=dataloader,
                    criterions=criterions,
                    inverse_y=inverse_y,
                    inverse_y_pred=inverse_y_pred,
                    y=y,
                    gamma=gamma_dist,
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
            total_y_pred_sigma_trans.append(low_bound)
            total_y_pred_sigma_2_trans.append(up_bound)
            total_y_true.append(y)
            total_alpha.append(alpha)
            total_beta.append(beta)
            if (batch_ix + 1) >= max_bacth_num:
                break
    total_y = {
        "true_trans": torch.cat(total_y_true_trans, dim=0),
        "pred_trans": torch.cat(total_y_pred_trans, dim=0),
        "unobs_mask": torch.cat(total_unobs_mask, dim=0),
        "n_id": torch.cat(total_nodes_id, dim=0),
        "t_id": torch.cat(total_time_id, dim=0),
    }
    total_y.update(
        {"low_bound_trans": torch.cat(total_y_pred_sigma_trans, dim=0)})
    total_y.update(
        {"up_bound_trans": torch.cat(total_y_pred_sigma_2_trans, dim=0)})

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
        torch_y = torch.cat(total_y_true, dim=0)
        torch_alpha = torch.cat(total_alpha, dim=0)
        torch_beta = torch.cat(total_beta, dim=0)
        gamma_dist = torch.distributions.Gamma(torch_alpha, torch_beta)
        metrics = log_performance2(
            dataloader=dataloader,
            criterions=criterions,
            inverse_y=inverse_y,
            inverse_y_pred=inverse_y_pred,
            y=torch_y,
            gamma=gamma_dist,
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
