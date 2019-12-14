
# BASED ON: https://github.com/pytorch/examples/tree/master/mnist


from __future__ import print_function
import argparse
import kornia
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
import torchvision
import matplotlib.pyplot as plt
import numpy as np

from PIL import Image

def imshow(img, file_name):
    img = img * 255
    # img = (img * 0.3081) + 0.1307     # unnormalize
    npimg = img.numpy()
    plt.imshow(np.transpose(npimg, (1, 2, 0))[:,:,0])
    plt.savefig(file_name)

def imshow2(img, file_name):
    # img = img * 255
    # img = (img * 0.3081) + 0.1307     # unnormalize
    npimg = img.numpy()
    plt.imshow(np.transpose(npimg, (1, 2, 0)))
    plt.savefig(file_name)

def image_loader(image):
    if not isinstance(image, Image.Image):
        image = F.to_pil_image(image)

    transform = transforms.Compose([
        transforms.Grayscale(num_output_channels=1),
        transforms.Resize((64, 64)),
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,)),
        Binarize(0.2165)
    ])

    tensor = transform(image).float()

    return tensor

class Binarize(object):
    """Applies Laplacian. Args - kernel size."""

    def __init__(self, threshold):
        self.threshold = threshold

    def __call__(self, sample):
        y = torch.zeros(sample.size())
        x = torch.ones(sample.size())
        return torch.where(sample > self.threshold, x, y)

def main():
    no_transform = transforms.Compose([transforms.ToTensor()])
    transform = transforms.Compose([
        transforms.Grayscale(num_output_channels=1),
        transforms.Resize((64, 64)),
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,)),
        Binarize(0.2165)
    ])

    path = './frames/train/left/thumb20065.jpg'
    original_image = torch.tensor(no_transform(Image.open(path)).float())
    transformed_image = image_loader(Image.open(path))

    imshow2(original_image, './original.jpg')
    imshow(transformed_image, './transformed.jpg')

if __name__ == '__main__':
    main()
