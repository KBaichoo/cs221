
import torch
import kornia

import cv2
import numpy as np
from matplotlib import pyplot as plt


# create the operator
laplace = kornia.filters.Laplacian(5)


def plot_transform(image):
    global laplace
    image = cv2.imread(image, 0)

    img = kornia.image_to_tensor(image).float()
    # img = torch.unsqueeze(img.float(), dim=0)  # BxCxHxW

    img_lap = laplace(img)
    # Convert to numpy to display here
    img_lap = kornia.tensor_to_image(img_lap.byte()[0])

    plt.subplot(2, 2, 1), plt.imshow(image, cmap='binary')
    plt.title('Original'), plt.xticks([]), plt.yticks([])
    plt.subplot(2, 2, 2), plt.imshow(img_lap, cmap='binary')
    plt.title('Laplacian'), plt.xticks([]), plt.yticks([])
    plt.show()


with open('./files_imgs', 'r') as f:
    for filename in f:
        print('Opening {}'.format(filename))
        plot_transform(filename.strip())
