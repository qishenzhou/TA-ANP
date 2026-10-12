import torch


# define scheduler factory function
def get_scheduler(optimizer, scheduler_name, scheduler_patience):
    schedulers = {
        'cosine': torch.optim.lr_scheduler.CosineAnnealingLR,
        'step': torch.optim.lr_scheduler.StepLR,
        'plateau': torch.optim.lr_scheduler.ReduceLROnPlateau
    }

    if scheduler_name == 'plateau':
        return schedulers[scheduler_name](optimizer,
                                          mode='min',
                                          factor=0.5,
                                          min_lr=1e-5,
                                          patience=scheduler_patience)
    elif scheduler_name == 'cosine':
        return schedulers[scheduler_name](optimizer, T_max=5, eta_min=1e-7)
    elif scheduler_name == 'step':
        return schedulers[scheduler_name](optimizer,
                                          step_size=scheduler_patience,
                                          gamma=0.5)
    else:

        class NullScheduler:

            def step(self, loss=None):
                pass

        return NullScheduler()


# define optimizer factory function
def get_optimizer(model, optimizer_name, lr, weight_decay):
    optimizers = {
        'adamw': torch.optim.AdamW,
        'adam': torch.optim.Adam,
        'sgd': torch.optim.SGD
    }

    if optimizer_name not in optimizers:
        raise ValueError(f"Unknown optimizer: {optimizer_name}")

    return optimizers[optimizer_name](model.parameters(),
                                      lr=lr,
                                      weight_decay=weight_decay)
