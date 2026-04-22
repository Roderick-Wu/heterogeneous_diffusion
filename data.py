import os
import numpy as np
import torch
import json
from torch.utils.data import Dataset, DataLoader, Sampler
from torchvision import datasets, transforms


class MNISTDataset(Dataset):
    def __init__(self, save_path, train=True):
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,))
        ])
        self.dataset = datasets.MNIST(root=save_path, train=train, download=True, transform=transform)

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        image, label = self.dataset[idx]
        return image, label


class StatefulBatchSampler(Sampler):
    def __init__(self, dataset_size, batch_size, seed=0, shuffle=True, drop_last=False):
        self.dataset_size = dataset_size
        self.batch_size = batch_size
        self.seed = seed
        self.shuffle = shuffle
        self.drop_last = drop_last

        self.epoch = 0
        self.cursor = 0
        self._order = self._make_order(self.epoch)

    def _make_order(self, epoch):
        if self.shuffle:
            g = torch.Generator()
            g.manual_seed(self.seed + epoch)
            return torch.randperm(self.dataset_size, generator=g)
        return torch.arange(self.dataset_size)

    def __iter__(self):
        while self.cursor < self.dataset_size:
            end = min(self.cursor + self.batch_size, self.dataset_size)
            batch = self._order[self.cursor:end]
            self.cursor = end

            if self.drop_last and batch.numel() < self.batch_size:
                break

            yield batch.tolist()

        self.epoch += 1
        self.cursor = 0
        self._order = self._make_order(self.epoch)

    def __len__(self):
        if self.drop_last:
            return self.dataset_size // self.batch_size
        return (self.dataset_size + self.batch_size - 1) // self.batch_size

    def state_dict(self):
        return {
            'epoch': self.epoch,
            'cursor': self.cursor,
        }

    def load_state_dict(self, state):
        self.epoch = int(state.get('epoch', 0))
        self.cursor = int(state.get('cursor', 0))
        self._order = self._make_order(self.epoch)

def shuffle_and_split_dataset(dataset, val_split=0.2):
    total_size = len(dataset)
    indices = np.random.permutation(total_size)
    #import pdb; pdb.set_trace()
    val_size = int(val_split * total_size)
    
    val_indices = indices[:val_size]
    train_indices = indices[val_size:]
    
    train_subset = torch.utils.data.Subset(dataset, train_indices)
    val_subset = torch.utils.data.Subset(dataset, val_indices)
    
    return train_subset, val_subset


def take_dataset_shard(dataset, shard_index=0, num_shards=1):
    if num_shards <= 0:
        raise ValueError("num_shards must be > 0")
    if shard_index < 0 or shard_index >= num_shards:
        raise ValueError("shard_index must be in [0, num_shards)")

    if num_shards == 1:
        return dataset

    indices = [i for i in range(len(dataset)) if i % num_shards == shard_index]
    if len(indices) == 0:
        raise ValueError("Shard produced no samples. Check shard_index/num_shards.")
    return torch.utils.data.Subset(dataset, indices)