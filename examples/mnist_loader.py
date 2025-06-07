import matplotlib.pyplot as plt
import numpy as np
from cgraph.utils.data.datasets.mnist import MNIST
from cgraph.utils.data.loader import DataLoader

mnist_dataset = MNIST('train')
mnist_loader = DataLoader(mnist_dataset, batch_size=15, shuffle=True)

data = next(iter(mnist_loader))

print(data[0].shape)  # Should print the shape of the images
print(data[1])  # Should print the labels of the images

# Visualize the first 15 images in the batch
fig, axes = plt.subplots(3, 5, figsize=(10, 6))
for i, ax in enumerate(axes.flat):
    ax.imshow(data[0][i].reshape(28, 28), cmap='gray')
    ax.set_title(f'Label: {data[1][i]}')
    ax.axis('off')
plt.tight_layout()
plt.show()
# This code snippet loads the MNIST dataset, retrieves a batch of images and labels,
# and visualizes the first 15 images in that batch using matplotlib.
# It prints the shape of the images and the corresponding labels.
# The images are reshaped to 28x28 pixels for visualization.