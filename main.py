import os
from pathlib import Path

import numpy as np
import pytorch_lightning as pl
import torch

from args import initialize_args, get_experiment_id
from data_provider.data_factory import data_provider
from experiments import train_model, sampling_anp, default_inference
from model._model_register import ModelRegistry
from utils.evaluation_tool import MadridMetrics, get_loss_function
from utils.log_tool import print_args, print_header, print_train, set_wandb
from utils.train_tool import get_optimizer, get_scheduler
from rich import print
from utils.report_tool import visual_part_results

# os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'
torch.set_default_dtype(torch.float32)
# torch.use_deterministic_algorithms(True)




def inference(args, model, test_loader):
    args.best_checkpoint_path = args.best_checkpoint_path + args.project_name + "/" + args.exp_id + "/"
    args.best_prediction_path = args.best_prediction_path + args.project_name + "/" + args.exp_id + "/"
    args.best_metrics_path = args.best_metrics_path + args.project_name + "/"

    if args.pretrained_model_path is None:
        try:
            model_dict = torch.load(
                os.path.join(args.best_checkpoint_path,
                             "best_val_checkpoint.pth"))["state_dict"]
        except FileNotFoundError:
            model_dict = torch.load(
                os.path.join(args.best_checkpoint_path,
                             "best_train_checkpoint.pth"))["state_dict"]
        print(f"load pretrained from {args.best_checkpoint_path}")
        model.load_state_dict(model_dict)
    else:
        pretrained = torch.load(args.pretrained_model_path)
        if isinstance(pretrained, dict) and "state_dict" in pretrained:
            model.load_state_dict(pretrained["state_dict"])
        else:
            model = pretrained
        print(f"load pretrained from {args.pretrained_model_path}")

    if args.num_sampling > 0:
        sampling_anp(model, test_loader, args)
    else:
        default_inference(model, test_loader, args)
    if args.plot:
        visual_part_results(
            args,
            data_name=getattr(args, 'inference_data_name',
                              'inference_total_y.pkl'))
    exit()


def get_diy_args():
    """get args interactively
    """
    return input("Enter the exp comments: ").strip() or None


if __name__ == "__main__":
    args = initialize_args()
    args.diy = get_diy_args() if args.diy is None else args.diy
    pl.seed_everything(args.seed)
    test_loader = data_provider(args, "test")
    args.dataset_name = test_loader.dataset.__class__.__name__
    args.project_name = f"{args.project_name}-{args.dataset_name}-{(1-args.train_ratio-args.val_ratio):.2f}-{args.input_window_size}-{args.look_back_window_size}-{args.pred_window_size}-{args.stride}-{args.seed4unob}-{int(args.permanent_unobs_ratio*100)}"
    args.exp_id = get_experiment_id(args)
    model = ModelRegistry.get_model(args.model)(args).to(args.device).float()
    if args.inference_only:
        inference(args, model, test_loader)
    train_loader = data_provider(args, "train")
    if args.val_ratio > 0:
        val_loader = data_provider(args, "val")
        data_loaders = dict(
            zip(["train", "val", "test"], [train_loader, val_loader, test_loader]))
    else:
        data_loaders = dict(zip(["train", "test"], [train_loader, test_loader]))
    wandb = set_wandb(args)
    model_parameters = filter(lambda p: p.requires_grad, model.parameters())
    args.num_params = sum([np.prod(p.size()) for p in model_parameters]).item()

    optimizer = get_optimizer(
        model=model,
        optimizer_name=args.optimizer,
        lr=args.learning_rate,
        weight_decay=args.weight_decay,
    )
    scheduler = get_scheduler(
        optimizer=optimizer,
        scheduler_name=args.scheduler,
        scheduler_patience=args.scheduler_patience,
    )

    loss = get_loss_function(args.loss)
    criterions = dict(
        zip(
            [args.loss, "log_metrics"],
            [
                loss,
                MadridMetrics(look_back_window_size=args.look_back_window_size,
                              pred_window_size=args.pred_window_size,
                              have_unobs=args.permanent_unobs_ratio > 0,
                              metric_names=args.metric_names),
            ],
        ))  # loss is for training, metrics is for logging
    if args.verbose:
        print_header("*** Model Configurations ***")
        print(model)
    print_header("*** Experiment Configurations *** ")
    print_args(args)
    print_train(args)
    print(
        f"├── train data length: {len(train_loader.dataset)}, train loader length: {len(train_loader)}"
    )
    if args.val_ratio > 0:
        print(
            f"├── val data length: {len(val_loader.dataset)}, val loader length: {len(val_loader)}"
        )
    print(
        f"└── test data length: {len(test_loader.dataset)}, test loader length: {len(test_loader)}"
    )
    print(
        f"└── number of permanently unoberved nodes: {len(test_loader.dataset.permanent_unobs_nodes_global_idx)}, number of all nodes: {len(test_loader.dataset.active_node_global_idx)}"
    )

    pretrained = train_model(
        model=model,
        optimizer=optimizer,
        scheduler=scheduler,
        data_loaders=data_loaders,
        criterions=criterions,
        args=args,
        wandb=wandb,
    )

    if args.save_model:
        save_path = f"./pretrained/{args.exp_id}.pth"
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        torch.save(pretrained, save_path)
    if args.plot:
        visual_part_results(args)
    print("👏👏👏 Done!")
