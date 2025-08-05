import torch
import numpy as np
import random

class SNRScheduler:
    def __init__(self,
                 snr_start: float,
                 snr_step: float,
                 step_interval: int,
                 total_epochs: int,
                 min_snr: float = None,
                 schedule_type: str = 'linear'):

        self.snr_start = snr_start
        self.snr_step = snr_step
        self.step_interval = step_interval
        self.total_epochs = total_epochs
        self.min_snr = min_snr
        self.schedule_type = schedule_type
        self.current_epoch = 0
        self.warmup_epochs = 1

    def get_current_snr(self) -> float:
        if self.current_epoch < self.warmup_epochs:
            return float('inf')

        effective_epoch = self.current_epoch - self.warmup_epochs
        progress = effective_epoch / self.total_epochs

        if self.schedule_type == 'linear':
            snr_now = self.snr_start - (self.snr_start - self.min_snr) * progress
        elif self.schedule_type == 'step':
            stage = self.current_epoch // self.step_interval
            snr_now = self.snr_start - stage * self.snr_step
        else:
            raise ValueError("Unsupported schedule type")
        return snr_now

    def step(self):
        self.current_epoch = min(self.current_epoch + 1, self.total_epochs)


def set_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    np.random.seed(seed)
    random.seed(seed)


class BalancedBatchSampler(torch.utils.data.Sampler):
    def __init__(self, dataset, batch_size):
        self.dataset = dataset
        self.batch_size = batch_size
        self.num_batches = len(dataset) // batch_size

    def __iter__(self):
        indices = torch.randperm(len(self.dataset))
        for i in range(self.num_batches):
            start = i * self.batch_size
            end = start + self.batch_size
            yield indices[start:end]

    def __len__(self):
        return self.num_batches
