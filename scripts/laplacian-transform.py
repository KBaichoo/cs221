import cv2
import numpy as np
from matplotlib import pyplot as plt


def plot_transform(image):
    img = cv2.imread(image, 0)

    laplacian = cv2.Laplacian(img, cv2.CV_64F, ksize=5)
    sobelx = cv2.Sobel(img, cv2.CV_64F, 1, 0, ksize=5)
    sobely = cv2.Sobel(img, cv2.CV_64F, 0, 1, ksize=5)

    plt.subplot(2, 2, 1), plt.imshow(img, cmap='binary')
    plt.title('Original'), plt.xticks([]), plt.yticks([])
    plt.subplot(2, 2, 2), plt.imshow(laplacian, cmap='binary')
    plt.title('Laplacian'), plt.xticks([]), plt.yticks([])
    plt.subplot(2, 2, 3), plt.imshow(sobelx, cmap='binary')
    plt.title('Sobel X'), plt.xticks([]), plt.yticks([])
    plt.subplot(2, 2, 4), plt.imshow(sobely, cmap='binary')
    plt.title('Sobel Y'), plt.xticks([]), plt.yticks([])

    plt.show()


with open('./files_imgs', 'r') as f:
    for filename in f:
        print('Opening {}'.format(filename))
        plot_transform(filename.strip())
