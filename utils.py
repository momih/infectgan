from __future__ import division
from dataset import Xrays, Masked
import numpy as np
import math
import imageio
import time
import os


def get_stddev(x, k_h, k_w):
    return 1/math.sqrt(k_w*k_h*x.get_shape()[-1])


def read_data(data):
    dir_data = './data/' + data
    return Xrays(directory=dir_data)

def read_masked(data):
    dir_data = './data/' + data
    return Masked(directory=dir_data)

def squash_image_to_range(image, image_value_range=(-1, 1)):
    image = image.astype(np.float32) * (image_value_range[-1] - image_value_range[0]) / 255.0 + image_value_range[0]
    return image


def save_images(images, size, image_path):
    n, w, h = images.shape[:-1]
    images = images.reshape((n, w, h))
    images = (images + 1.0) / 2.0
    img = np.zeros((h * size[0], w * size[1]))
    for idx, image in enumerate(images):
        i = idx % size[1]
        j = idx // size[1]
        img[j*h:j*h+h, i*w:i*w+w] = image
        
    return imageio.imsave(image_path, img)


def merge(images, size):
    h, w = images.shape[1:]
    img = np.zeros((h * size[0], w * size[1]))
    for idx, image in enumerate(images):
        i = idx % size[1]
        j = idx // size[1]
        img[j*h:j*h+h, i*w:i*w+w] = image
    return img


def imsave(images, size, path):
    return imageio.imsave(path, merge(images, size))


def inverse_transform(images):
    return (images + 1.0) / 2.0


def save_stats(filename, image_name, labels):
    current_time = time.strftime("%d/%m/%Y - %H:%M:%S")
    text = "\n\nFile {} created at {} \n" \
           "The files used are the following combinations - \n".format(image_name, current_time)
    str_files = '\n'.join(["{}".format(x) for x in labels])
    ending_line = "\n\n==========================================================================\n\n"
    with open(filename, 'a') as f:
        f.write(text + str_files + ending_line)
