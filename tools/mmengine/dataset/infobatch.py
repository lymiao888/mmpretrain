'''Save loss and embedding together with data, load data depending on them.
Construct my dataset to get index
Try two version of code for the autothreshold:
v1: with camparison: if one batch's loss has upper confidence bound smaller than loss mean/median/mode' lower confidence bound, we can
prune on it
v2: if moving average loss is stable and gradient small, then a sample is well learned
'''
import os
import torch
import numpy as np
import torchvision
import math
import time
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils
from collections import defaultdict
from torch.utils.data import Dataset, Sampler
from torch.utils.data.distributed import DistributedSampler
from operator import itemgetter
from typing import Iterator, List, Optional, Union

class MyTrainSet(Dataset):
    def __init__(self, dataset, total_steps = None, class_num = None, logger = None, ratio = 0.5):
        self.dataset = dataset
        self.class_num = class_num
        self.logger = logger
        if logger is not None:
            logger.info("initializing mydataset")
        self.scores = np.ones([len(self.dataset)])
        self.class_samples = defaultdict(list)
        self.weights = np.ones(len(self.dataset))
        self.save_num = 0
        self.total_time = 0
        self.ratio = ratio
        self.total_steps = total_steps
        if logger is not None:
            logger.info("initialized mydataset")

    def __setscore__(self, indices, values):
        self.scores[indices] = values

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index):
        data = self.dataset[index]
        weight = self.weights[index]
        return data, index, weight

    def prune(self):
        # prune samples that are well learned, rebalence the weight by scaling up remaining
        # well learned samples' learning rate to keep estimation about the same
        # for the next version, also consider new class balance
        start = time.time()
        if self.logger is not None:
            self.logger.info("running my pruning sampler")
        b = self.scores<self.scores.mean()
        well_learned_samples = np.where(b)[0]
        pruned_samples = []
        pruned_samples.extend(np.where(np.invert(b))[0])
        selected = np.random.choice(well_learned_samples, int(0.5*len(well_learned_samples)))
        self.reset_weights()
        if len(selected)>0:
            self.weights[selected]=2.0
            pruned_samples.extend(selected)
        if self.logger is None:
            print('Cut {} samples for this iteration'.format(len(self.dataset)-len(pruned_samples)))
        else:
            self.logger.info('Cut {} samples for this iteration'.format(len(self.dataset)-len(pruned_samples)))
        self.save_num += len(self.dataset)-len(pruned_samples)
        np.random.shuffle(pruned_samples)
        end = time.time()
        self.total_time += end-start
        return pruned_samples

    def pruning_sampler(self):
        return MyIterator(self.prune, initial_len=len(self.dataset), func2=self.no_cut, total_steps=self.total_steps)

    def total_time_cost(self):
        return self.total_time

    def no_cut(self):
        samples = list(range(len(self.dataset)))
        np.random.shuffle(samples)
        return samples

    def mean_score(self):
        return self.scores.mean()

    def normal_sampler_no_cut(self):
        return MyIterator(self.no_cut, initial_len=len(self.dataset))

    def get_weights(self,indexes):
        return self.weights[indexes]

    def total_save(self):
        return self.save_num

    def reset_weights(self):
        self.weights = np.ones(len(self.dataset))

class MyIterator():
    def __init__(self, func, initial_len = 0, func2=None, total_steps=None):
        self.func = func
        self.func2 = func2
        self.total_steps = total_steps
        self.seq = None
        self.initial_len = initial_len
        self.seed = 0
        self.current_step = 0
        self.reset()

    def reset(self):
        np.random.seed(self.seed)
        if self.total_steps is not None and self.func2 is not None and self.current_step>=int(0.85*self.total_steps):
            self.seq = self.func2()
        else:
            self.seq = self.func()
        self.seed+=1
        self.ite = iter(self.seq)
        self.new_length = len(self.seq)

    def __next__(self):
        try:
            nxt = next(self.ite)
            self.current_step+=1
            return nxt
        except StopIteration:
            self.reset()
            raise StopIteration

    def __len__(self):
        return len(self.seq)

    def __iter__(self):
        self.ite = iter(self.seq)
        return self

class DatasetFromSampler(Dataset):
    """Dataset to create indexes from `Sampler`.
    Args:
        sampler: PyTorch sampler
    """

    def __init__(self, sampler: Sampler):
        """Initialisation for DatasetFromSampler."""
        self.sampler = sampler
        self.sampler_list = None

    def __getitem__(self, index: int):
        """Gets element of the dataset.
        Args:
            index: index of the element in the dataset
        Returns:
            Single element by index
        """
        if self.sampler_list is None:
            self.sampler_list = list(self.sampler)
        return self.sampler_list[index]

    def __len__(self) -> int:
        """
        Returns:
            int: length of the dataset
        """
        return len(self.sampler)

class DistributedSamplerWrapper(DistributedSampler):
    """
    Wrapper over `Sampler` for distributed training.
    Allows you to use any sampler in distributed mode.
    It is especially useful in conjunction with
    `torch.nn.parallel.DistributedDataParallel`. In such case, each
    process can pass a DistributedSamplerWrapper instance as a DataLoader
    sampler, and load a subset of subsampled data of the original dataset
    that is exclusive to it.
    .. note::
        Sampler can change size during training.
    """

    def __init__(
        self,
        sampler,
        num_replicas: Optional[int] = None,
        rank: Optional[int] = None,
        shuffle: bool = True,
    ):
        """
        Args:
            sampler: Sampler used for subsampling
            num_replicas (int, optional): Number of processes participating in
                distributed training
            rank (int, optional): Rank of the current process
                within ``num_replicas``
            shuffle (bool, optional): If true (default),
                sampler will shuffle the indices
        """
        super(DistributedSamplerWrapper, self).__init__(
            DatasetFromSampler(sampler),
            num_replicas=num_replicas,
            rank=rank,
            shuffle=shuffle,
        )
        self.sampler = sampler

    def __iter__(self) -> Iterator[int]:
        """Iterate over sampler.
        Returns:
            python iterator
        """
#         self.sampler.reset()
        self.dataset = DatasetFromSampler(self.sampler)
        if self.drop_last and len(self.dataset) % self.num_replicas != 0:  # type: ignore[arg-type]
            # Split to nearest available length that is evenly divisible.
            # This is to ensure each rank receives the same amount of data when
            # using this Sampler.
            self.num_samples = math.ceil(
                (len(self.dataset) - self.num_replicas) / self.num_replicas  # type: ignore[arg-type]
            )
        else:
            self.num_samples = math.ceil(len(self.dataset) / self.num_replicas)  # type: ignore[arg-type]
        self.total_size = self.num_samples * self.num_replicas
        indexes_of_indexes = super().__iter__()
        subsampler_indexes = self.dataset
        return iter(itemgetter(*indexes_of_indexes)(subsampler_indexes))


@torch.no_grad()
def concat_all_gather(tensor, dim=0):
    """
    Performs all_gather operation on the provided tensors.
    *** Warning ***: torch.distributed.all_gather has no gradient.
    """
    tensors_gather = [torch.ones_like(tensor) for _ in range(torch.distributed.get_world_size())]
    torch.distributed.all_gather(tensors_gather, tensor, async_op=False)
    output = torch.cat(tensors_gather, dim=dim)
    return output

def is_master():
    if not torch.distributed.is_available():
        return True

    if not torch.distributed.is_initialized():
        return True

    if torch.distributed.get_rank()==0:
        return True

    return False

def split_index(t):
    low_mask = 0b111111111111111
    low = torch.tensor([x&low_mask for x in t])
    high = torch.tensor([(x>>15)&low_mask for x in t])
    return low,high

def recombine_index(low,high):
    original_tensor = torch.tensor([(high[i]<<15)+low[i] for i in range(len(low))])
    return original_tensor