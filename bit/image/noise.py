# -*- coding:utf-8 -*-
"""
Methods to simulate(add) different noise models on images
"""

import numpy as np


def gaussian(img, stdev=3.):
    """
    Generate zero-mean additive gaussian noise
    """
    noiz = np.random.normal(0., stdev, img.shape)
    noisy_img = (1. * img) + noiz
    return noisy_img


def poisson(img):
    """
    Add poisson noise to image array
    """
    img_nzy = np.random.poisson(img).astype(float);
    return img_nzy;


def salt_n_pepper(img, perc=10):
    """
    Generate salt-and-pepper noise in an image. 
    """
    # Create a flat copy of the image
    flat = img.ravel().copy
    # The total number of pixels
    total = len(flat)
    # The number of pixels we need to modify
    nmod = int(total * (perc/100.))
    # Random indices to modify
    indices = np.random.random_integers(0, total-1, (nmod,))
    # Set the first half of the pixels to 0
    flat[indices[:nmod/2]] = 0
    # Set the second half of the pixels to 255
    flat[indices[nmod/2:]] = 255
    return flat.reshape(img.shape)
