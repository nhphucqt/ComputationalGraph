"""
References:
- Dataset: https://www.kaggle.com/datasets/hojjatk/mnist-dataset
- Code: https://www.kaggle.com/code/hojjatk/read-mnist-dataset
"""

from ..dataset import Dataset
from array import array
from configurations import CONFIG_CACHED_DATASET_DIR
from pathlib import Path
import numpy as np
import requests
import struct
import zipfile

class MNIST(Dataset):
    URL = "https://www.kaggle.com/api/v1/datasets/download/hojjatk/mnist-dataset"
    DATASET_PATH = Path(CONFIG_CACHED_DATASET_DIR + "/mnist")
    TRAIN_PATH_IMAGE = 'train-images-idx3-ubyte/train-images-idx3-ubyte'
    TRAIN_PATH_LABEL = 'train-labels-idx1-ubyte/train-labels-idx1-ubyte'
    TEST_PATH_IMAGE = 't10k-images-idx3-ubyte/t10k-images-idx3-ubyte'
    TEST_PATH_LABEL = 't10k-labels-idx1-ubyte/t10k-labels-idx1-ubyte'
    FILE_NAME = "mnist.zip"

    def __init__(self, split: str, transform=None, target_transform=None):
        super().__init__()
        self.split = split
        self.transform = transform
        self.target_transform = target_transform

        if not self.DATASET_PATH.exists():
            print(f"Dataset not found at {self.DATASET_PATH}. Downloading...")
            self.DATASET_PATH.mkdir(parents=True, exist_ok=True)
            self.download_dataset()

        if self.split == 'train':
            images_filepath = self.DATASET_PATH / self.TRAIN_PATH_IMAGE
            labels_filepath = self.DATASET_PATH / self.TRAIN_PATH_LABEL
        elif self.split == 'test':
            images_filepath = self.DATASET_PATH / self.TEST_PATH_IMAGE
            labels_filepath = self.DATASET_PATH / self.TEST_PATH_LABEL

        self.images, self.labels = self.read_images_labels(images_filepath, labels_filepath)

    def download_dataset(self) -> None:
        response = requests.get(self.URL)
        if response.status_code == 200:
            with open(self.DATASET_PATH / self.FILE_NAME, 'wb') as file:
                file.write(response.content)
        else:
            raise Exception(f"Failed to download dataset: {response.status_code}")
        
        with zipfile.ZipFile(self.DATASET_PATH / self.FILE_NAME, 'r') as zip_ref:
            zip_ref.extractall(self.DATASET_PATH)

        print("Dataset downloaded and extracted successfully.")

    def read_images_labels(self, images_filepath, labels_filepath):        
        labels = []
        with open(labels_filepath, 'rb') as file:
            magic, size = struct.unpack(">II", file.read(8))
            if magic != 2049:
                raise ValueError('Magic number mismatch, expected 2049, got {}'.format(magic))
            labels = array("B", file.read())        
        
        with open(images_filepath, 'rb') as file:
            magic, size, rows, cols = struct.unpack(">IIII", file.read(16))
            if magic != 2051:
                raise ValueError('Magic number mismatch, expected 2051, got {}'.format(magic))
            image_data = array("B", file.read())        
        images = []
        for i in range(size):
            images.append([0] * rows * cols)
        for i in range(size):
            img = np.array(image_data[i * rows * cols:(i + 1) * rows * cols])
            img = img.reshape(28, 28)
            images[i][:] = img
            images[i] = np.row_stack(images[i])

        print(f"Loaded {len(images)} images and {len(labels)} labels from {self.split} set.")
        
        return images, labels

    def __len__(self):
        return len(self.images)

    def __getitem__(self, index):
        image = self.images[index]
        label = self.labels[index]
        if self.transform:
            image = self.transform(image)
        if self.target_transform:
            label = self.target_transform(label)
        return image, label