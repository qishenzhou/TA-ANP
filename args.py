import argparse
import numpy as np


def initialize_args():
    parser = argparse.ArgumentParser(description="Arguments")

    parser.add_argument("-p", "--project_name", type=str, default="Lucky")

    # I/O paths
    parser.add_argument("-dr", "--data_root", type=str, default="./MadridAllinOne/DLData")
    parser.add_argument(
        "-dmp",
        "--distance_matrix_path",
        type=str,
        default="distance_matrix.h5",
    )
    parser.add_argument("-sfp",
                        "--static_features_path",
                        type=str,
                        default="road_info.h5")
    parser.add_argument("-sdp",
                        "--sensor_data_path",
                        type=str,
                        default="fixed_sensor_data.h5")
    parser.add_argument("-tdp", "--tomtom_data_path", type=str, default="tomtom_data.h5")
    parser.add_argument(
        "-bcp",
        "--best_checkpoint_path",
        type=str,
        default="./check_points/",
        help="best val checkpoint path",
    )
    parser.add_argument(
        "-bpp",
        "--best_prediction_path",
        type=str,
        default="./predictions/",
        help="results path",
    )
    parser.add_argument("-bmp",
                        "--best_metrics_path",
                        type=str,
                        default="./results/",
                        help="results path")
    parser.add_argument("-bip",
                        "--best_images_path",
                        type=str,
                        default="./images/",
                        help="images path")
    parser.add_argument("-logp",
                        "--log_path",
                        type=str,
                        default="./logs/",
                        help="log path")

    # I/O parameters
    parser.add_argument("-iw", "--input_window_size", type=int, default=4)
    parser.add_argument("-lbw", "--look_back_window_size", type=int, default=4)
    parser.add_argument("-pw", "--pred_window_size", type=int, default=4)
    parser.add_argument("-st", "--stride", type=int, default=1)
    parser.add_argument("-sp", "--split", type=str, default="train")
    parser.add_argument("-tr", "--train_ratio", type=float, default=0.6)
    parser.add_argument("-vr", "--val_ratio", type=float, default=0.2)
    parser.add_argument("-tur", "--train_unobs_ratio", type=float, default=0.25)
    parser.add_argument("-pur", "--permanent_unobs_ratio", type=float, default=0.25)
    parser.add_argument("-pup", "--permanent_unobs_path", type=str, default="")
    parser.add_argument("-pun", "--permanent_unobs_num", type=int, default=50)
    parser.add_argument(
        '-tfs',
        '--target_features',
        type=str,
        default='q',
        choices=['q', 'o', 'v', 'qv', 'qov', 'c'],
        help='q is flow, v is speed, o is occupancy,c is congestion level')
    parser.add_argument(
        '-ifs',
        '--input_features',
        type=str,
        default='q',
        choices=['q', 'o', 'v', 'qv', 'qov', 'c'],
        help='q is flow, v is speed, o is occupancy, c is congestion level')
    parser.add_argument("-dataset",
                        "--dataset_class",
                        type=str,
                        default="MadridFSDataset4NP",
                        choices=["MadridDataset", "MadridFSDataset4NP"],
                        help="dataset class")
    parser.add_argument("-stack_t",
                        "--stack_time",
                        action="store_true",
                        help="stack the time step dimension")
    parser.add_argument(
        "-nc",
        "--num_cluster",
        type=int,
        default=10,
        help="Number of clusters for each hop",
    )
    parser.add_argument(
        "-td",
        "--threshold_distance",
        type=int,
        default=1,
        help="Distance threshold for adjacency matrix (km)",
    )
    parser.add_argument(
        "-sg",
        "--sigma",
        type=float,
        default=5.0,
        help="Gaussian kernel bandwidth parameter",
    )
    parser.add_argument(
        "-su",
        "--seed4unob",
        type=int,
        default=42,
        help="Random seed for unobserved node generation",
    )
    parser.add_argument("-ns",
                        "--not_scale",
                        action="store_true",
                        help="not scale the input data")
    parser.add_argument(
        "-freq",
        "--freq",
        type=str,
        default="min",
        help="frequency for time feature extraction",
    )
    parser.add_argument(
        "-fb",
        "--follow_batch",
        nargs="+",
        type=str,
        default=None,
        help="follow batch for pytorch geometric dataloader",
    )
    parser.add_argument("-notrb",
                        "--not-return-best",
                        action="store_false",
                        dest="return_best",
                        help="not return best model")
    parser.add_argument(
        "-nw",
        "--num_workers",
        type=int,
        default=20,
        help="number of workers for dataloader",
    )
    parser.add_argument("-nosf",
                        "--no-shuffle",
                        action="store_false",
                        dest="shuffle",
                        help="not shuffle the data")
    parser.add_argument("-drop",
                        "--drop_last",
                        action="store_true",
                        help="drop the last batch")
    parser.add_argument(
        "-ni",
        "--not_inverse",
        action="store_true",
        help="not inverse the data for results presentation",
    )
    parser.add_argument(
        "-sm",
        "--save_model",
        action="store_true",
        help="save model not model state dict",
    )

    # Training parameters
    parser.add_argument("-seed",
                        "--seed",
                        type=int,
                        default=42,
                        help="seed for reproducibility")
    parser.add_argument("-lr",
                        "--learning_rate",
                        type=float,
                        default=0.001,
                        help="learning rate")
    parser.add_argument(
        "-loss",
        "--loss",
        type=str,
        default="mask_beta_elbo",
        help="loss function",
    )
    parser.add_argument(
        "-tw",
        "--task_weights",
        nargs=3,
        type=float,
        default=[1.0, 1.0, 1.0],
        metavar=("OF", "UE", "UF"),
        help=("Task weights used by epoch_task_weighted, in the order "
              "observed forecasting, unobserved estimation, and "
              "unobserved forecasting."),
    )
    parser.add_argument("-dev",
                        "--device",
                        type=str,
                        default="cuda:0",
                        help="device to train on")
    parser.add_argument("-tbs",
                        "--t_batch_size",
                        type=int,
                        default=2,
                        help="temporal batch size")
    parser.add_argument("-sbs",
                        "--s_batch_size",
                        type=int,
                        default=50,
                        help="spatial batch size")
    parser.add_argument("-me", "--max_epoch", type=int, default=1, help="max epoch")
    parser.add_argument("-warm",
                        "--warmup_epoch",
                        type=int,
                        default=20,
                        help="warmup epoch")
    parser.add_argument("-wd",
                        "--weight_decay",
                        type=float,
                        default=0.0001,
                        help="weight decay")
    parser.add_argument(
        "-op",
        "--optimizer",
        type=str,
        default="adamw",
        choices=["adam", "sgd", "adamw"],
        help="optimizer",
    )
    parser.add_argument(
        "-sc",
        "--scheduler",
        type=str,
        default="None",
        choices=["plateau", "step", "cosine", "None"],
        help="scheduler",
    )
    parser.add_argument("-sch_p",
                        "--scheduler_patience",
                        type=int,
                        default=1,
                        help="scheduler patience")
    parser.add_argument("-pt",
                        "--patience",
                        type=int,
                        default=3,
                        help="early stopping epochs")
    parser.add_argument(
        "-stb",
        "--stop_based",
        type=str,
        default="train_mask",
        choices=["val_total", "val_mask", "train_total", "train_mask"],
        help="early stopping criteria",
    )
    parser.add_argument("-ro", "--read_only", action="store_true", help="read only")
    parser.add_argument("-io",
                        "--inference_only",
                        action="store_true",
                        help="inference only")
    parser.add_argument("-pmp",
                        "--pretrained_model_path",
                        type=str,
                        default=None,
                        help="pretrained model path")
    parser.add_argument("-num_sampling",
                        "--num_sampling",
                        type=int,
                        default=0,
                        help="number of sampling")
    parser.add_argument('-sn',
                        '--sorted_nodes_path',
                        type=str,
                        default=None,
                        help='sorted nodes path')
    parser.add_argument('-adn',
                        '--add_node_number',
                        type=int,
                        default=0,
                        help='number of add node')
    parser.add_argument('-rand_adn',
                        '--random_add_node',
                        default=False,
                        action="store_true",
                        help='random add node')
    parser.add_argument('-infunobs',
                        '--inference_unobs_num',
                        type=int,
                        default=0,
                        help='inference unobs number')
    parser.add_argument('-adb',
                        '--add_by_betweenness',
                        type=int,
                        default=0,
                        help='add by betweenness number')
    parser.add_argument('-adc',
                        '--add_by_closeness',
                        type=int,
                        default=0,
                        help='add by closeness number')

    parser.add_argument(
        "-mtb",
        "--max_train_batch_num",
        type=int,
        default=500,
        help="max number of train batches for controling the training time of one epoch",
    )
    parser.add_argument(
        "-etype",
        "--epoch_type",
        type=str,
        default="default",
        choices=["default", "task_weighted", "gamma", "noback"],
        help="epoch type",
    )

    # logging
    parser.add_argument("-wb",
                        "--wandb",
                        action="store_true",
                        help="enable wandb logging")
    parser.add_argument(
        "-we",
        "--wandb_entity",
        type=str,
        default="zqslalala",
        help="user name of wandb",
    )
    parser.add_argument(
        "-diy",
        "--diy",
        type=str,
        default="",
        help="Experiment note only; does not control model behavior",
    )
    parser.add_argument("-ver", "--verbose", action="store_true", help="verbose")
    parser.add_argument("-log_int", "--log_interval", type=int, default=1)
    parser.add_argument("-mn",
                        "--metric_names",
                        nargs="+",
                        default=["mask_mae", "mask_rmse", "mask_smape", "mask_rrmse"],
                        help="List of metric names to use")

    # Model
    parser.add_argument("-model", "--model", type=str, default="TAANP",
                        choices=["TAANP"], help="model name (default: TAANP)")
    parser.add_argument("--without_tamqm", action="store_true",
                        help="Use the shared-query attention ablation without TAMQM")
    parser.add_argument("--use-static", action=argparse.BooleanOptionalAction,
                        default=True, help="Use static road features")
    parser.add_argument("--use-hit", action=argparse.BooleanOptionalAction,
                        default=True, help="Use floating-car hit features")
    parser.add_argument("--use-speed", action=argparse.BooleanOptionalAction,
                        default=True, help="Use floating-car speed features")
    parser.add_argument("--use-poi", action=argparse.BooleanOptionalAction,
                        default=False, help="Use POI features")
    parser.add_argument("--print-experiment-id", action="store_true",
                        help=argparse.SUPPRESS)
    parser.add_argument("-h_dim",
                        "--hidden_dim",
                        type=int,
                        default=64,
                        help="hidden dimension")
    parser.add_argument("-z_dim", "--z_dim", type=int, default=64, help="latent dimension")
    parser.add_argument("-num_heads",
                        "--num_heads",
                        type=int,
                        default=4,
                        help="number of heads")
    parser.add_argument("-epsilon", "--epsilon", type=float, default=0.1, help="epsilon")
    parser.add_argument("-npp",
                        "--np_path",
                        type=str,
                        default="both",
                        help="TA-ANP uses both deterministic and latent paths",
                        choices=['both'])
    parser.add_argument("-opp",
                        "--output_path",
                        type=str,
                        default="normal",
                        help="output path",
                        choices=['normal', 'nig', 'sqt', 'deterministic', 'gamma'])
    parser.add_argument("-dropout",
                        "--dropout",
                        type=float,
                        default=0,
                        help="dropout rate")
    parser.add_argument("-plot", "--plot", action="store_true", help="plot the results")

    parser.add_argument("-beta",
                        "--beta",
                        type=float,
                        default=1.0,
                        help="weight for KL divergence in ELBO loss")
    parser.add_argument("-beta_scheduler",
                        "--beta_scheduler",
                        type=str,
                        default="None",
                        choices=["linear", "cyclic", "None", "logistic"],
                        help="scheduler for beta parameter")

    parser.add_argument('-bwe', '--beta_warmup_epochs', type=int,
                        default=20)  # Total duration of the S-shaped growth schedule.
    parser.add_argument('-bss', '--beta_steepness', type=float, default=10.0)  # Steepness of the S-shaped curve.

    parser.add_argument("-beta_start",
                        "--beta_start",
                        type=float,
                        default=0.0,
                        help="Initial value of beta")
    parser.add_argument("-beta_end",
                        "--beta_end",
                        type=float,
                        default=1.0,
                        help="Final value of beta")
    parser.add_argument("-bse",
                        "--beta_start_epoch",
                        type=int,
                        default=5,
                        help="Epoch to start increasing beta")
    parser.add_argument("-beta_cycle",
                        "--beta_cycle",
                        type=int,
                        default=10,
                        help="number of epochs for one beta cycle")
    parser.add_argument("-sampler",
                        "--sampler_class",
                        type=str,
                        default="NPSampler1",
                        choices=["NPSampler1", "NPSampler1Resilient"],
                        help="sampler class")

    args = parser.parse_args()

    if any(weight < 0 for weight in args.task_weights):
        parser.error("--task_weights must be non-negative")
    if sum(args.task_weights) <= 0:
        parser.error("at least one --task_weights value must be positive")

    args.dynamic_features_paths = {
        "sensor": f"{args.data_root}/{args.sensor_data_path}",
        "tomtom": f"{args.data_root}/{args.tomtom_data_path}",
    }
    args.distance_matrix_path = f"{args.data_root}/{args.distance_matrix_path}"
    args.static_features_path = f"{args.data_root}/{args.static_features_path}"
    if args.output_path == 'nig':
        if 'evidential' not in args.loss:
            print("NIG output requires evidential loss, defaulting to evidential loss?")
            # # Read user input.
            # user_input = input("Type 'y' to confirm, anything else to exit: ")
            # if user_input.lower() == 'y':
            args.loss = 'mask_evidential'
            # else:
            #     print(
            #         "Exiting. Please set the correct loss function and try again."
            #     )
            #     exit(0)
    elif args.output_path == 'sqt':
        if args.loss != 'mask_sort_crps':
            print(
                "SortQuantiles output requires mask_sort_crps loss, defaulting to mask_sort_crps loss?"
            )
            # # Read user input.
            # user_input = input("Type 'y' to confirm, anything else to exit: ")
            # if user_input.lower() == 'y':
            args.loss = 'mask_sort_crps'
            # else:
            #     print(
            #         "Exiting. Please set the correct loss function and try again."
            #     )
            #     exit(0)
    elif args.output_path == 'normal' and args.loss != 'mask_beta_elbo':
        print("Normal output requires mask_beta_elbo loss; using mask_beta_elbo.")
        args.loss = 'mask_beta_elbo'
    if args.add_node_number > 0 and args.sorted_nodes_path is not None:
        args.add_node_list = np.load(
            args.sorted_nodes_path)[:args.add_node_number].tolist()
        if args.random_add_node:
            args.add_node_list = np.load(
                'rand_unobs_nodes.npy')[:args.add_node_number].tolist()
    else:
        args.add_node_list = None
    return args


def get_experiment_id(args):
    """Name experiments from explicit settings without interpreting the note."""
    sampler = args.sampler_class.replace('Resilient', '')
    parts = [f"{args.model}-{args.seed}-{args.train_ratio}-{args.val_ratio}-"
             f"{args.threshold_distance}km-{sampler}"]
    if args.diy:
        parts.append(args.diy)
    for feature in ('static', 'hit', 'speed'):
        if not getattr(args, f'use_{feature}'):
            parts.append(f"no_{feature}")
    if args.use_poi:
        parts.append('poi')
    for field, default in [('output_path', 'normal'),
                           ('hidden_dim', 64), ('z_dim', 64), ('num_heads', 4)]:
        value = getattr(args, field)
        if value != default:
            parts.append(f'{field}_{value}')
    if args.without_tamqm:
        parts.append('without_tamqm')
    if args.epoch_type == 'task_weighted':
        parts.append('tw-' + '-'.join(f'{weight:g}' for weight in args.task_weights))
    elif args.epoch_type != 'default':
        parts.append(f'epoch_{args.epoch_type}')
    return '-'.join(parts)


if __name__ == "__main__":
    import contextlib
    import sys
    # Keep stdout machine-readable for the shell launchers.
    with contextlib.redirect_stdout(sys.stderr):
        args = initialize_args()
    print(get_experiment_id(args) if args.print_experiment_id else args)
