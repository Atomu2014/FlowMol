import torch
from torch.utils.data import Sampler, BatchSampler

from flowmol.data_processing.dataset import MoleculeDataset

class AdaptiveEdgeSampler(BatchSampler):
    def __init__(self, dataset, edges_per_batch: int,
                 sampler = None,
                 distributed: bool = False,
                 rank: int = None,
                 num_replicas: int = None,
                  ):

        self.dataset: MoleculeDataset = dataset
        self.edges_per_batch = edges_per_batch
        self.distributed = distributed

        if self.distributed:
            self.num_replicas = num_replicas if num_replicas is not None else torch.distributed.get_world_size()
            self.rank = rank if rank is not None else torch.distributed.get_rank()

            dataset_frac_per_worker = 1.0 / self.num_replicas
            self.frac_start = self.rank * dataset_frac_per_worker
            self.frac_end = (self.rank + 1) * dataset_frac_per_worker
        else:
            self.rank = 0
            self.num_replicas = 1
            self.frac_start = 0
            self.frac_end = 1

        # Compute local shard once; each worker consumes its shard exactly once per epoch.
        start_idx = int(self.frac_start * len(self.dataset))
        end_idx = int(self.frac_end * len(self.dataset))
        self.local_indices = torch.arange(start_idx, end_idx)
        self.samples_per_epoch = len(self.local_indices)

    def setup_queue(self):
        if len(self.local_indices) == 0:
            self.sample_queue = torch.empty(0, dtype=torch.long)
            return
        self.sample_queue = self.local_indices[torch.randperm(len(self.local_indices))]

    def get_next_batch(self, queue_idx):
        # Build one dynamic batch from queue[queue_idx:], stopping when edge budget is reached.
        # Always include at least one graph per batch.
        batch_idxs = []
        n_edges = 0
        while queue_idx < len(self.sample_queue):
            idx = self.sample_queue[queue_idx]
            idx_edges = int(self.dataset.n_edges_per_graph[idx])

            if len(batch_idxs) > 0 and (n_edges + idx_edges) > self.edges_per_batch:
                break

            batch_idxs.append(idx)
            n_edges += idx_edges
            queue_idx += 1

        return batch_idxs, queue_idx

    def __iter__(self):
        self.setup_queue()
        queue_idx = 0
        while queue_idx < len(self.sample_queue):
            next_batch, queue_idx = self.get_next_batch(queue_idx)
            if len(next_batch) == 0:
                break
            yield next_batch

    def __len__(self):
        # Dynamic batching means exact batch count is data/order dependent.
        # Return a deterministic estimate for progress bars.
        if self.samples_per_epoch == 0:
            return 0
        mean_edges = float(self.dataset.n_edges_per_graph[self.local_indices].float().mean())
        mean_edges = max(mean_edges, 1.0)
        est_batches = int((self.samples_per_epoch * mean_edges) / max(self.edges_per_batch, 1))
        return max(est_batches, 1)