from torch_geometric.loader import DataLoader
from data_provider.madrid_dataset import MadridDataset
from data_provider.sampler import NPSampler1


class STDataLoader(DataLoader):
    """
    Custom DataLoader integrating NeighborLoader for efficient spatial and temporal sampling.

    This loader handles both spatial and temporal dimensions of the data,
    providing efficient batch generation for spatio-temporal graph data.
    """

    dataset: MadridDataset

    def __init__(
        self,
        dataset: MadridDataset,
        sampler: NPSampler1,
        follow_batch: list[str] = None,
        num_workers: int = 0,
        **kwargs,
    ):
        """   Initialize the Madrid DataLoader.
        Args:
            dataset (MadridDataset): pytorch dataset instance
            sampler (SpatioTemporalBatchSampler): spatio-temporal batch sampler
            follow_batch (list[str], optional): pytorch batch attributes to follow. Defaults to None.
            num_workers (int, optional): number of workers for loading data. Defaults to 0.
            **kwargs: additional arguments passed to DataLoader
        """

        super().__init__(
            dataset,
            batch_sampler=sampler,  # Use batch_sampler instead of batch_size
            follow_batch=follow_batch,
            num_workers=num_workers,
            pin_memory=False,
            persistent_workers=num_workers > 0,
            **kwargs,
        )
        self.num_t_batches = sampler.num_t_batches
        self.num_s_batches = sampler.num_s_batches

    def inverse_transform(self, y, *args):
        return self.dataset.inverse_transform(y, *args)

    def transform(self, y):
        return self.dataset.forward_transform(y)

    def tom_inverse_transform(self, y):
        return self.dataset.tom_inverse_transform(y)

    def tom_forward_transform(self, y):
        return self.dataset.tom_forward_transform(y)

    @property
    def scale(self):
        return self.dataset.scale

    @property
    def tom_scale(self):
        return self.dataset.tom_scale

    @property
    def center(self):
        return self.dataset.center

    def get_num_t_batches(self):
        return self.num_t_batches

    def get_num_s_batches(self):
        return self.num_s_batches
