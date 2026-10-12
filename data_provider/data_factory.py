from data_provider.loader import STDataLoader
from rich import print


def get_dataset(args, flag):
    """Build the Madrid dataset used for TA-ANP experiments."""
    if args.dataset_class not in {"MadridDataset", "MadridFSDataset4NP"}:
        raise ValueError(f"Unsupported TA-ANP dataset: {args.dataset_class}")
    module = __import__('data_provider.madrid_dataset', fromlist=[args.dataset_class])
    # Instantiate the dataset with provided arguments
    dataset_class = getattr(module, args.dataset_class)
    dataset = dataset_class(
        distance_matrix_path=args.distance_matrix_path,
        static_features_path=args.static_features_path,
        dynamic_features_paths=args.dynamic_features_paths,
        input_window=args.input_window_size,
        look_back_window=args.look_back_window_size,
        pred_window=args.pred_window_size,
        stride=args.stride if flag != 'train' else 1,
        split=flag,
        train_ratio=args.train_ratio,
        val_ratio=args.val_ratio,
        train_unobs_ratio=args.train_unobs_ratio,
        permanent_unobs_ratio=args.permanent_unobs_ratio,
        threshold_distance=args.threshold_distance,
        seed4unob=args.seed4unob,
        target_features=args.target_features,
        input_features=args.input_features,
        not_scale=args.not_scale,
        stack_time=args.stack_time,
        add_node_list=args.add_node_list,
        inf_exclude_num=args.inference_unobs_num,
        add_by_betweenness=args.add_by_betweenness,
        add_by_closeness=args.add_by_closeness,
    )

    return dataset


def get_sampler(args, dataset, drop_last, shuffle):
    module = __import__('data_provider.sampler', fromlist=[args.sampler_class])
    sampler_class = getattr(module, args.sampler_class)

    sampler = sampler_class(
        dataset=dataset,
        t_batch_size=args.t_batch_size,
        s_batch_size=args.s_batch_size,
        shuffle=shuffle,
        drop_last=drop_last,
        num_cluster=args.num_cluster,
    )

    return sampler


def data_provider(args, flag):
    print(f'Preparing {flag} data...')
    dataset = get_dataset(args, flag)
    if flag == "test":
        drop_last = False  # make sure test data is not influenced by batch size
        shuffle = False
    else:
        drop_last = args.drop_last
        shuffle = args.shuffle
    sampler = get_sampler(args, dataset, drop_last, shuffle)

    return STDataLoader(dataset=dataset,
                        sampler=sampler,
                        follow_batch=args.follow_batch,
                        num_workers=args.num_workers)
