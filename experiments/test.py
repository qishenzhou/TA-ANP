import torch
from utils.log_tool import log_performance
from utils.evaluation_tool import MadridMetrics, get_loss_function
from rich import print


def read_out(data_loaders, exp_id, args, best_epoch={}):
    criterions = dict(
        zip(
            [args.loss, "log_metrics"],
            [
                get_loss_function(args.loss),
                MadridMetrics(
                    look_back_window_size=args.look_back_window_size,
                    pred_window_size=args.pred_window_size,
                    have_unobs=args.permanent_unobs_ratio > 0,
                    metric_names=[
                        "mask_mae", "mask_rmse", "mask_mape", "mask_smape",
                        "mask_rrmse", "mask_r2", "mask_crps", "mask_picb",
                        "mask_qice"
                    ],
                ),
            ],
        ))  # loss is for training, metrics is for logging

    for i, mode in enumerate(data_loaders.keys()):
        if mode == 'train' and 'val' in data_loaders.keys():
            continue
        metrics = {}
        for metric_name, criterion in criterions.items():
            if metric_name == 'log_metrics':
                for sub_metric_name in criterion.log_metrics:
                    metrics[sub_metric_name] = 0
            else:
                metrics[f'loss/{metric_name}'] = 0
        # checkpoint = checkpoint_path + f'best_{mode}_checkpoint.pth'
        # epoch = torch.load(checkpoint)['epoch']

        inverse_y = args.best_total_y[mode]['true_trans']
        inverse_y_pred = args.best_total_y[mode]['pred_trans']

        inverse_y, inverse_y_pred, unobs_mask = args.best_total_y[mode][
            'true_trans'], args.best_total_y[mode][
                'pred_trans'], args.best_total_y[mode]['unobs_mask']
        inverse_y_pred_sigma = args.best_total_y[mode].get(
            'pred_sigma_trans', None)
        inverse_y_pred_sigma_2 = args.best_total_y[mode].get(
            'pred_sigma_trans2', None)
        metrics = log_performance(
            dataloader=data_loaders[mode],
            criterions=criterions,
            inverse_y=inverse_y,
            inverse_y_pred=inverse_y_pred,
            inverse_y_pred_sigma=inverse_y_pred_sigma,
            inverse_y_pred_sigma_2=inverse_y_pred_sigma_2,
            label_unobs_mask=unobs_mask,
            args=args,
            mode=mode,
            metrics=metrics,
            loss=torch.tensor(float('nan')),
        )

        with open(args.best_metrics_path + 'results.txt',
                  'a',
                  encoding='utf-8') as f:
            description = f'├── exp_id: {exp_id} | Best Based on {mode}, at Epoch: {best_epoch.get(mode,None)}'
            for metric_name, metric in metrics.items():
                description += f' | {metric_name}: {metric:.4f}'
            f.write(description + '\n')
            print(description)
            if i < len(data_loaders.keys()) - 1:
                f.write(f'{"-"*len(description)}' + '\n')
                print(f'{"-"*len(description)}')
    with open(args.best_metrics_path + 'results.txt', 'a',
              encoding='utf-8') as f:
        f.write(f'{"="*len(description)}' + '\n')
        print(f'{"="*len(description)}')
    print('-> Testing finished!')
