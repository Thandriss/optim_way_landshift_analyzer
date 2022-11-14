import cv2
import numpy as np
from matplotlib import pyplot as plt
from matplotlib.pyplot import imshow

import numpy as np
import cv2
import argparse
import matplotlib.pyplot as plt

from sobel.conv import convolution
from sobel.smooothing_gaus import gaussian_blur


def sobel_edge_detection(image, filter, verbose):
    new_image_x = convolution(image, filter, verbose)

    if verbose:
        plt.imshow(new_image_x, cmap='gray')
        plt.title("Horizontal Edge")
        plt.show()

    new_image_y = convolution(image, np.flip(filter.T, axis=0), verbose)

    if verbose:
        plt.imshow(new_image_y, cmap='gray')
        plt.title("Vertical Edge")
        plt.show()

    gradient_magnitude = abs(np.square(new_image_x) + abs(new_image_y))

    gradient_magnitude = 255.0 / gradient_magnitude.max()

    if verbose:
        plt.imshow(gradient_magnitude, cmap='gray')
        plt.title("Gradient Magnitude")
        plt.show()

    return gradient_magnitude


if __name__ == '__main__':
    filter_sobel = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]])
    image = cv2.imread("C:\\Users\\mahon\\PycharmProjects\\diploma_work\\pictures\\_0001_35100_1_35090_01_ORT_10_03_01.png")
    # image = gaussian_blur(image, 9, verbose=True)
    sobel_edge_detection(image, filter_sobel, verbose=True)