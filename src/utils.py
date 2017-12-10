"""
Some useful utilities
"""
import logging
import json
from os import path, makedirs
import numpy as np
from PIL.Image import open as open_img, LANCZOS
from PIL.ImageOps import equalize

IMAGE_WIDTH = 256
IMAGE_HEIGHT = 256

MODE = 'gpu'

CWD = path.dirname(path.realpath(__file__))
DATA_PATH = path.join(CWD, 'data')
CAFFE_PATH = '$CAFFE_ROOT'


def get_genre_labels(transofrm=False, file=path.join(CWD, 'labels.json')):
    """
    Returns genre, label dictionary or null and exception
    """
    try:
        labels = json.load(open(file))
        if not transofrm:
            return labels
        genre_label = {}
        for label in labels:
            for genre in label['addition']:
                genre_label[genre] = {
                    'label': label['label'],
                    'genre': label['genre']
                }
        return genre_label
    except (IOError, ValueError):
        return {}


def get_logger(file):
    """Returns logger object"""
    log_path = path.join(CWD, 'log')
    try_makedirs(log_path)
    logging.basicConfig(
        format=u'%(levelname)-8s [%(asctime)s] %(message)s',
        level=logging.INFO,
        filename=path.join(log_path, file))
    return logging


def transform_img(image_path, width=IMAGE_WIDTH, height=IMAGE_HEIGHT):
    """Returns resized image"""
    image = open_img(image_path).resize((width, height), resample=LANCZOS)
    image = equalize(image)
    # image.save('out.jpg')
    return image


def augment_img(image):
    """Generates images"""
    images = [image, np.fliplr(image)]
    return images


def try_makedirs(name):
    """Makes path if it doesn't exist"""
    if not path.exists(name):
        makedirs(name)
