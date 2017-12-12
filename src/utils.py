"""
Some useful utilities
"""
import logging
import json
from os import path, makedirs
import numpy as np
from PIL.Image import open as open_img, fromarray, LANCZOS
from PIL.ImageOps import equalize
from imgaug import augmenters as iaa

IMAGE_WIDTH = 256
IMAGE_HEIGHT = 256

MODE = 'gpu'

CWD = path.dirname(path.realpath(__file__))
DATA_PATH = path.join(CWD, 'data')
CAFFE_MODELS_PATH = path.join(CWD, 'caffe_models')
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
    return transform(open_img(image_path), width, height)


def augment_img(image_path, amount, width=IMAGE_WIDTH, height=IMAGE_HEIGHT):
    """Returns an array of `amount` generated images"""
    if amount <= 1:
        return [transform(open_img(image_path), width, height)]
    img = np.array(open_img(image_path))
    # The array has shape (amount, width, height, 3)
    # and dtype uint8.
    images = np.array([img for _ in range(amount)], dtype=np.uint8)

    seq = iaa.Sequential(
        [
            iaa.Fliplr(0.5),  # horizontal flips
            iaa.Crop(percent=(0, 0.1)),  # random crops
            # Small gaussian blur with random sigma between 0 and 0.5.
            # But we only blur about 50% of all images.
            iaa.Sometimes(0.5, iaa.GaussianBlur(sigma=(0, 0.5))),
            # Strengthen or weaken the contrast in each image.
            iaa.ContrastNormalization((0.75, 1.5)),
            # Add gaussian noise.
            # For 50% of all images, we sample the noise once per pixel.
            # For the other 50% of all images, we sample the noise per pixel AND
            # channel. This can change the color (not only brightness) of the
            # pixels.
            iaa.AdditiveGaussianNoise(
                loc=0, scale=(0.0, 0.05 * 255), per_channel=0.5),
            # Make some images brighter and some darker.
            # In 20% of all cases, we sample the multiplier once per channel,
            # which can end up changing the color of the images.
            iaa.Multiply((0.8, 1.2), per_channel=0.2),
            # Apply affine transformations to each image.
            # Scale/zoom them, translate/move them, rotate them and shear them.
            iaa.Affine(
                scale={"x": (0.8, 1.2),
                       "y": (0.8, 1.2)},
                translate_percent={"x": (-0.2, 0.2),
                                   "y": (-0.2, 0.2)},
                rotate=(-25, 25),
                shear=(-8, 8))
        ],
        random_order=True)  # apply augmenters in random order

    images_aug = seq.augment_images(images)
    return [transform(fromarray(img), width, height) for img in images_aug]


def try_makedirs(name):
    """Makes path if it doesn't exist"""
    if not path.exists(name):
        makedirs(name)


def transform(image, width, height):
    """Returns resized and eqialized copy of the image"""
    img = image.resize((width, height), LANCZOS)
    return equalize(img)
