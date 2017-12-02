"""
Some useful utilities
"""
import logging
from os import path, makedirs
from PIL import Image

IMAGE_WIDTH = 256
IMAGE_HEIGHT = 256

CWD = path.dirname(path.realpath(__file__))
CAFFE_PATH = '/home/ubuntu/caffe/'
DATA_PATH = path.join(CWD, 'data')


def get_logger(file):
    """Returns logger object"""
    log_path = path.join(CWD, 'log')
    try_makedirs(log_path)
    logging.basicConfig(
        format=u'%(levelname)-8s [%(asctime)s] %(message)s',
        level=logging.INFO,
        filename=path.join(log_path, file))
    return logging


def transform_img(image, width=IMAGE_WIDTH, height=IMAGE_HEIGHT):
    """Returns resized image"""
    return image.resize((width, height), resample=Image.BICUBIC)


def try_makedirs(name):
    """Makes path if it doesn't exist"""
    if not path.exists(name):
        makedirs(name)
