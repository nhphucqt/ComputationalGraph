import numpy as np
from .dataset import Dataset

class DataLoader:
    def __init__(self, dataset: Dataset, batch_size: int = 1, shuffle: bool = False):
        self.dataset = dataset
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.indices = list(range(len(dataset)))
        if shuffle:
            import random
            random.shuffle(self.indices)

    def __iter__(self):
        for start in range(0, len(self.dataset), self.batch_size):
            end = min(start + self.batch_size, len(self.dataset))
            batch_indices = self.indices[start:end]
            features, labels = zip(*[self.dataset[i] for i in batch_indices])
            features = np.array(features)
            labels = np.array(labels)
            yield features, labels

    def __len__(self):
        return (len(self.dataset) + self.batch_size - 1) // self.batch_size