"""
Training functions and helpers
"""

import pickle
import time
from argparse import Namespace
from pathlib import Path

import torch
from torch import nn
from torch.optim import Optimizer
from torch.optim.lr_scheduler import StepLR

from data_provider.loader import STDataLoader
from utils.log_tool import save_checkpoint
from .test import read_out
from rich import print
import math


def get_shared_epoch(epoch_type='default'):
    """Select a TA-ANP training objective without silently changing experiments."""
    from . import epoch_default, epoch_task_weighted, epoch_gamma, epoch_noback
    epochs = {
        'default': epoch_default.shared_epoch,
        'task_weighted': epoch_task_weighted.shared_epoch,
        'gamma': epoch_gamma.shared_epoch,
        'noback': epoch_noback.shared_epoch,
    }
    if epoch_type not in epochs:
        raise ValueError(f"Unsupported TA-ANP epoch type: {epoch_type}")
    return epochs[epoch_type]


def run_epoch(model,
              data_loaders,
              optimizer,
              scheduler,
              criterions,
              args,
              epoch,
              need_train=True,
              shared_epoch_fn=None):
    metrics = {}
    total_y = {mode: {} for mode in data_loaders.keys()}

    for mode, dataloader in data_loaders.items():
        # you can define your own shared_step function here to set the desired training process
        st = time.time()
        model, mode_metrics, total_y_mode = shared_epoch_fn(
            model, dataloader, optimizer, criterions, args, mode, need_train,
            epoch)
        total_time = time.time() - st
        print(f"├── {mode} finished, took {total_time:.2f}s")
        metrics.update({f"{mode}": mode_metrics})
        total_y[mode] = total_y_mode

    if need_train:
        if 'val' in data_loaders.keys():
            save_checkpoint(model, optimizer, args, epoch, "val",
                            metrics["val"][f'loss/{args.loss}'], total_y,
                            args.loss)
        else:
            save_checkpoint(model, optimizer, args, epoch, "train",
                            metrics["train"][f'loss/{args.loss}'], total_y,
                            args.loss)
        # note: we save the best test model only for debugging, not for training or tuning
        save_checkpoint(model, optimizer, args, epoch, "test",
                        metrics["test"][f'loss/{args.loss}'], total_y,
                        args.loss)
        if args.scheduler == "plateau":
            scheduler.step(metrics["train"][f'loss/{args.loss}'])
        elif args.scheduler == "cosine" or args.scheduler == "step":
            scheduler.step()

        print(f'-> Learning rate: {optimizer.param_groups[0]["lr"]:.5f}')
    return metrics


def train_model(
    model: nn.Module,
    optimizer: Optimizer,
    scheduler: StepLR,
    data_loaders: dict[str, STDataLoader],
    criterions: dict[str, nn.Module],
    args: Namespace,
    wandb=None,
):

    if args.read_only:
        args.best_checkpoint_path = args.best_checkpoint_path + args.project_name + "/" + args.exp_id + "/"
        args.best_prediction_path = args.best_prediction_path + args.project_name + "/" + args.exp_id + "/"
        args.best_metrics_path = args.best_metrics_path + args.project_name + "/"
        with open(args.best_prediction_path + "best_total_y.pkl", "rb") as f:
            args.best_total_y = pickle.load(f)
        check_point = args.best_checkpoint_path + "best_train_checkpoint.pth"
        best_model_dict = torch.load(check_point)
        best_epoch = best_model_dict["epoch"]
        read_out(data_loaders, args.exp_id, args, {'train': best_epoch})
        print(f"Returning best train model from epoch {best_epoch}")
        model.load_state_dict(best_model_dict["state_dict"])
        return model

    # set beta scheduler
    if args.beta_scheduler == "None":
        args.beta_scheduler_fn = lambda epoch: args.beta
    elif args.beta_scheduler == "linear":
        args.beta_scheduler_fn = lambda epoch: args.beta_start if epoch < args.beta_start_epoch else args.beta_start + (
            args.beta_end - args.beta_start) * min(1.0, (
                epoch - args.beta_start_epoch) / args.beta_warmup_epochs)
    elif args.beta_scheduler == "cyclic":
        args.beta_scheduler_fn = lambda epoch: (
            args.beta_start + (args.beta_end - args.beta_start) * (
                (epoch % args.beta_cycle) / args.beta_cycle)
            if (epoch // args.beta_cycle) % 2 == 0 else args.beta_end -
            (args.beta_end - args.beta_start) * (
                (epoch % args.beta_cycle) / args.beta_cycle))
    elif args.beta_scheduler == "logistic":
        # Logistic (S-shaped) growth.
        # Required parameters:
        # args.beta_warmup_epochs: Number of epochs over which beta increases from start to finish.
        # args.beta_steepness: Steepness of the S-shaped curve (e.g., 10.0).
        args.beta_scheduler_fn = lambda epoch: args.beta_start + (
            args.beta_end - args.beta_start) * (
                1.0 / (
                    1.0 + math.exp(-args.beta_steepness * (
                        # Map the epoch to [-1, 1] before applying the sigmoid.
                        (2.0 * min(epoch, args.beta_warmup_epochs) / args.
                         beta_warmup_epochs) - 1.0))))
    else:
        raise ValueError(f"Unknown beta scheduler: {args.beta_scheduler}")

    # set epoch type
    epoch_type = getattr(args, 'epoch_type', 'default')
    shared_epoch_fn = get_shared_epoch(epoch_type)
    print(f"-> Using epoch type: {epoch_type}")

    # set file path
    args.best_checkpoint_path = args.best_checkpoint_path + args.project_name + "/" + args.exp_id + "/"
    Path(args.best_checkpoint_path).mkdir(parents=True, exist_ok=True)
    args.best_prediction_path = args.best_prediction_path + args.project_name + "/" + args.exp_id + "/"
    Path(args.best_prediction_path).mkdir(parents=True, exist_ok=True)
    args.best_metrics_path = args.best_metrics_path + args.project_name + "/"
    Path(args.best_metrics_path).mkdir(parents=True, exist_ok=True)
    args.best_val_metric = 1e10
    args.best_val_metric_epoch = -1
    args.best_train_metric = 1e10  # Interpolation / fitting also good to test
    args.best_train_metric_epoch = -1
    args.best_test_metric = 1e10
    args.best_test_metric_epoch = -1
    args.best_total_y = {}  # save best epoch test results
    print("-> Start training...")
    wait_count = 0
    for epoch in range(1, args.max_epoch + 1):
        st = time.time()
        # Update beta value using the scheduler function
        args.beta = args.beta_scheduler_fn(epoch)

        metrics = run_epoch(model,
                            data_loaders,
                            optimizer,
                            scheduler,
                            criterions,
                            args,
                            epoch,
                            shared_epoch_fn=shared_epoch_fn)

        # Reset early stopping count if epoch improved
        if 'val' in data_loaders.keys(
        ) and args.best_val_metric_epoch == epoch or epoch < args.warmup_epoch:
            wait_count = 0
        elif 'val' not in data_loaders.keys(
        ) and args.best_train_metric_epoch == epoch or epoch < args.warmup_epoch:
            wait_count = 0
        else:
            wait_count += 1
            print(f"-> Early stopping count: {wait_count}/{args.patience}")

        if wandb is not None:
            log_metrics = {}
            for mode in metrics.keys():
                for metric_name, metric_value in metrics[f"{mode}"].items():
                    log_metrics[f"{mode}/{metric_name}"] = metric_value
            wandb.log(log_metrics, step=epoch)
            wandb.log({"lr": optimizer.param_groups[0]["lr"]}, step=epoch)
            wandb.log({"beta": args.beta}, step=epoch)  # Log beta value

        for mode in metrics:
            if "train" in mode:
                continue
            description = f"├── {mode} metrics:"
            for metric_name, metric in metrics[f"{mode}"].items():
                description += f"{metric_name}: {metric:.4f} | "
            print(description)

        print(f"├── Epoch {epoch} finished, took {time.time() - st:.2f}s")
        print(f"├── Current beta: {args.beta:.4f}")

        if wait_count == args.patience:
            print(f"Early stopping at epoch {epoch}!")
            break  # Exit for loop and do early stopping

    print("-> Training finished!")
    if 'val' in data_loaders.keys():
        print(
            f"-> Saved best val model checkpoint at epoch {args.best_val_metric_epoch}!",
            f"-> Saved best test model checkpoint at epoch {args.best_test_metric_epoch}!",
        )
    else:
        print(
            f"-> Saved best train model checkpoint at epoch {args.best_train_metric_epoch}!",
            f"-> Saved best test model checkpoint at epoch {args.best_test_metric_epoch}!",
        )
    print(f"   - Saved to: {args.best_checkpoint_path}")
    print("-> Save predictions and metrics...")
    with open(args.best_prediction_path + "best_total_y.pkl", "wb") as f:
        pickle.dump(args.best_total_y, f)
    read_out(
        data_loaders, args.exp_id, args, {
            'val': args.best_val_metric_epoch,
            'train': args.best_train_metric_epoch,
            'test': args.best_test_metric_epoch
        })
    if 'val' in data_loaders.keys():
        check_point = args.best_checkpoint_path + "best_val_checkpoint.pth"
    else:
        check_point = args.best_checkpoint_path + "best_train_checkpoint.pth"
    best_model_dict = torch.load(check_point)
    epoch = best_model_dict["epoch"]
    best_epoch = best_model_dict["epoch"]
    print(f"Returning best train model from epoch {best_epoch}")
    model.load_state_dict(best_model_dict["state_dict"])
    return model
