import os
from datetime import datetime

import pytz
import rich.syntax
import rich.tree
import torch
from omegaconf import DictConfig, ListConfig, OmegaConf
from rich import print


def get_formatted_time(timezone="Etc/GMT-1"):
    dt = datetime.now()
    if timezone is None:
        return dt.strftime("-%m-%d-%H-%M-%S-")
    tz = pytz.timezone(timezone)
    dt = dt.astimezone(tz)
    return dt.strftime("-%m-%d-%H-%M-%S-")


def print_header(x):
    print("-" * len(x))
    print(x)
    print("-" * len(x))


def print_epoch_metrics(metrics):
    for mode in metrics.keys():
        print("-" * 4, f"{mode}", "-" * 4)
        for k, v in metrics[mode].items():
            print(f"- {k}: {v:.5f}")


def print_train(args):
    print_header("*** Training Configurations ***")
    print(
        f"├── Input/Look back/Prediction/Stride Horizon: {args.input_window_size,args.look_back_window_size,args.pred_window_size,args.stride}"
    )
    print(f"├── Number of trainable parameters: {args.num_params}")
    print(f"├── Experiment name: {args.exp_id}")
    print(f"├── Checkpoints save to: {args.best_checkpoint_path}")
    print(f"├── Predictions save to: {args.best_prediction_path}")
    print(f"├── Metrics save to: {args.best_metrics_path}")


def set_wandb(args):
    if not os.path.exists(args.log_path):
        os.makedirs(args.log_path)
    if args.wandb:
        import swanlab as wandb
        wandb.init(
            config={},
            # entity=args.wandb_entity,  # user id
            experiment_name=args.exp_id,  # experiment name
            project=args.project_name,
            logdir=args.log_path,
            mode="cloud",
        )
        wandb.config.update(args)  # record experiment parameters

        return wandb
    return None


def print_args(args, resolve: bool = True, name: str = "ARGPARSE ARGS") -> None:
    """Prints content of DictConfig using Rich library and its tree structure.
    Args:
        args (Namespace):
        resolve (bool, optional): Whether to resolve reference fields of DictConfig.
        name (str, optional): Name of the tree.
    """

    tree = rich.tree.Tree(
        name,
        style="bold red",
    )
    config = OmegaConf.create(vars(args))
    fields = config.keys()
    for field in fields:
        branch = tree.add(field, style="blue")

        config_section = config.get(field)
        branch_content = str(config_section)
        if isinstance(config_section, DictConfig):
            branch_content = OmegaConf.to_yaml(config_section, resolve=resolve)
        elif isinstance(config_section, ListConfig):
            branch_content = OmegaConf.to_yaml(config_section, resolve=resolve)

        branch.add(rich.syntax.Syntax(branch_content, "yaml"), style="green")

    rich.print(tree)


def log_performance(
    dataloader,
    criterions,
    inverse_y,
    inverse_y_pred,
    label_unobs_mask,
    args,
    mode,
    metrics,
    loss,
    inverse_y_pred_sigma=None,
    inverse_y_pred_sigma_2=None,
    max_bacth_num=None,
    print_enabled=False,
    batch_ix=0,
):
    max_bacth_num = len(dataloader) if max_bacth_num is None else max_bacth_num
    description = (
        f"└── {mode} batch {batch_ix+1:{len(str(max_bacth_num)) + 1}d}/{max_bacth_num}")
    for metric_name, criterion in criterions.items():
        if metric_name == args.loss:
            metric_value = loss.item()
        elif metric_name == "log_metrics":
            metric_value = criterion(
                preds=inverse_y_pred,
                targets=inverse_y,
                label_unobs_mask=label_unobs_mask,
                preds_sigma=inverse_y_pred_sigma,
                preds_sigma_2=inverse_y_pred_sigma_2,
            )
        else:
            raise NotImplementedError
        if metric_name == "log_metrics":
            for sub_metric_name, sub_metric_value in metric_value.items():
                metrics[sub_metric_name] += sub_metric_value  # log metrics
        else:
            metrics[f'loss/{metric_name}'] += metric_value  # loss

        if ((batch_ix + 1) % args.log_interval == 0
                and print_enabled) or batch_ix == max_bacth_num - 1:
            if metric_name == "log_metrics":
                if "train" in mode:
                    for sub_metric_name, sub_metric_value in metric_value.items():
                        description += f" | {sub_metric_name}: {(metrics[sub_metric_name]/(batch_ix+1)):.4f}"
                else:
                    for sub_metric_name, sub_metric_value in metric_value.items():
                        description += f" | {sub_metric_name}: {sub_metric_value:.4f}"
            else:
                if "train" in mode:
                    description += (
                        f" | loss/{metric_name}: {(metrics[f'loss/{metric_name}']/(batch_ix+1)):.4f}"
                    )
                else:
                    description += f" | {metric_name}: {metric_value:.4f}"

    if ((batch_ix + 1) % args.log_interval == 0
            and print_enabled) or (batch_ix == max_bacth_num - 1 and print_enabled):
        print(description)
    return metrics


def better_metric(metric_a, metric_b):
    return metric_a < metric_b


def save_checkpoint(model, optimizer, args, epoch, mode, now_metric, total_y,
                    metric_name):
    checkpoint_path = args.best_checkpoint_path + f"best_{mode}_checkpoint.pth"
    best_metric = getattr(args, f"best_{mode}_metric")
    if better_metric(now_metric, best_metric) or epoch == 0:
        setattr(args, f"best_{mode}_metric", now_metric)
        setattr(args, f"best_{mode}_metric_epoch", epoch)
        args.best_total_y.update({f"{mode}": total_y["test"]})
        print(
            f"-> 📉 ✨ New best {mode} loss at epoch {epoch}! ({metric_name}: {now_metric:.4f})"
        )
        torch.save(
            {
                "epoch": epoch,
                "val_metric": now_metric,
                "state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
            },
            checkpoint_path,
        )


def log_performance2(
    dataloader,
    criterions,
    inverse_y,
    inverse_y_pred,
    y,
    gamma,
    label_unobs_mask,
    args,
    mode,
    metrics,
    loss,
    max_bacth_num=None,
    print_enabled=False,
    batch_ix=0,
):
    max_bacth_num = len(dataloader) if max_bacth_num is None else max_bacth_num
    description = (
        f"└── {mode} batch {batch_ix+1:{len(str(max_bacth_num)) + 1}d}/{max_bacth_num}")
    for metric_name, criterion in criterions.items():
        if metric_name == args.loss:
            metric_value = loss.item()
        elif metric_name == "log_metrics":
            metric_value = criterion(
                preds_inverse=inverse_y_pred,
                targets_inverse=inverse_y,
                label_unobs_mask=label_unobs_mask,
                targets_not_inverse=y,
                gamma_dist=gamma,
            )
        else:
            raise NotImplementedError
        if metric_name == "log_metrics":
            for sub_metric_name, sub_metric_value in metric_value.items():
                metrics[sub_metric_name] += sub_metric_value  # log metrics
        else:
            metrics[f'loss/{metric_name}'] += metric_value  # loss

        if ((batch_ix + 1) % args.log_interval == 0
                and print_enabled) or batch_ix == max_bacth_num - 1:
            if metric_name == "log_metrics":
                if "train" in mode:
                    for sub_metric_name, sub_metric_value in metric_value.items():
                        description += f" | {sub_metric_name}: {(metrics[sub_metric_name]/(batch_ix+1)):.4f}"
                else:
                    for sub_metric_name, sub_metric_value in metric_value.items():
                        description += f" | {sub_metric_name}: {sub_metric_value:.4f}"
            else:
                if "train" in mode:
                    description += (
                        f" | loss/{metric_name}: {(metrics[f'loss/{metric_name}']/(batch_ix+1)):.4f}"
                    )
                else:
                    description += f" | {metric_name}: {metric_value:.4f}"

    if ((batch_ix + 1) % args.log_interval == 0
            and print_enabled) or (batch_ix == max_bacth_num - 1 and print_enabled):
        print(description)
    return metrics
