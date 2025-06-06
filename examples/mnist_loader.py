import matplotlib.pyplot as plt
import numpy as np
from data.datasets.mnist import MNIST
from data.loader import DataLoader

mnist_dataset = MNIST('train')
mnist_loader = DataLoader(mnist_dataset, batch_size=15, shuffle=True)

data = next(iter(mnist_loader))

plt.figure(figsize=(10, 6))
for idx, (image, label) in enumerate(data):
    plt.subplot(3, 5, idx + 1)
    plt.imshow(image, cmap='gray')
    plt.title(f"Label: {label}")
    plt.axis('off')

plt.tight_layout()
plt.show()