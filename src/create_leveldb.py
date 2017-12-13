"""
This script divides the training images into 2 sets and
stores them in leveldb databases for training and validation.

Usage: python create_leveldb.py
"""

from random import shuffle, random
from os import path
from shutil import rmtree
from glob import glob
import pandas as pd
import numpy as np
import leveldb
from caffe.proto import caffe_pb2
from utils import IMAGE_WIDTH, IMAGE_HEIGHT, CWD, DATA_PATH
from utils import get_logger, augment_img, try_makedirs, get_genre_labels

# approximate number of images per class
IMAGES_PER_CLASS = 30000


def make_datum(image, label):
    """
    Makes datum where image is in numpy.ndarray format.
    RGB changed to BGR
    """
    return caffe_pb2.Datum(
        channels=3,
        width=IMAGE_WIDTH,
        height=IMAGE_HEIGHT,
        label=label,
        data=np.rollaxis(np.array(image), 2).tostring())


def get_percentage(curr, total):
    """Returns progress"""
    percentage = int((curr / float(total)) * 100)
    if percentage < 10:
        return '[   ' + str(percentage) + '% ] '
    if percentage < 100:
        return '[  ' + str(percentage) + '% ] '
    return '[ ' + str(percentage) + '% ] '


def generate_images(image_path, number_of_imgs):
    """Generates some images"""
    amount = IMAGES_PER_CLASS / float(number_of_imgs)
    if amount <= 1:
        return augment_img(image_path, 1)
    if random() > 1 - (amount - int(amount)):
        amount += 1
    return augment_img(image_path, int(amount))


def main():
    """Main function"""

    logger = get_logger(path.splitext(path.basename(__file__))[0] + '.log')
    logger.info('Greetings from ' + path.basename(__file__))

    validation_ratio = 6
    leveldb_data_path = path.join(CWD, 'input')
    train_db_path = path.join(leveldb_data_path, 'train_leveldb')
    validation_db_path = path.join(leveldb_data_path, 'validation_leveldb')

    try_makedirs(leveldb_data_path)

    if path.exists(train_db_path):
        logger.info('Removing ' + train_db_path)
        rmtree(train_db_path)
    if path.exists(validation_db_path):
        logger.info('Removing ' + validation_db_path)
        rmtree(validation_db_path)

    all_data_info = pd.read_csv(path.join(DATA_PATH, 'all_data_info.csv'))
    # Creating label, genre data frame
    # genres = pd.DataFrame({'genre': all_data_info['genre'].dropna().unique()})
    genres = pd.DataFrame(columns=['label', 'genre', 'amount'])
    genre_label = get_genre_labels()
    for i, data in enumerate(genre_label):
        amount = all_data_info[all_data_info['genre'].isin(data['addition'])]
        genres.loc[i] = [data['label'], data['genre'], len(amount)]
    genres.to_csv(path.join(DATA_PATH, 'genre_labels.csv'), index=False)

    logger.info('Creating train_db_path and validation_db_path')
    train_db = leveldb.LevelDB(train_db_path)
    validation_db = leveldb.LevelDB(validation_db_path)

    train_images = [img for img in glob(path.join(DATA_PATH, 'train', '*.jpg'))]
    shuffle(train_images)
    null_genre = 0
    null_label = 0
    generated_imgs = 0
    genre_label = get_genre_labels(True)
    for in_idx, img_path in enumerate(train_images):
        # getting painting genre
        genre = all_data_info[all_data_info['new_filename'] == path.basename(
            img_path)]['genre'].dropna()
        # some paintings don't have a genre. Checking it
        if len(genre) < 1:
            null_genre += 1
            continue
        if genre.values[0] not in genre_label:
            # No label. It's strange, but let's go on...
            null_label += 1
            logger.critical(str(genre.values[0]) + ' has no label!')
            continue
        label = genre_label[genre.values[0]]['label']
        imgs = generate_images(
            img_path, genres[genres['label'] == int(label)]['amount'].values[0])
        for i, img in enumerate(imgs):
            datum = make_datum(img, int(label))
            # with open('datum', 'w') as file:
            #     file.write(datum.SerializeToString())
            if (in_idx + generated_imgs + i) % validation_ratio != 0:
                train_db.Put('{:0>5d}'.format(in_idx),
                             datum.SerializeToString())
            else:
                validation_db.Put('{:0>5d}'.format(in_idx),
                                  datum.SerializeToString())
        generated_imgs += len(imgs)
        logger.debug('{:0>5d}'.format(in_idx) + ':' + img_path + ' (+ ' +
                     str(len(imgs)) + ' augmented)')

        # printing progress and file name
        print(
            get_percentage(in_idx, len(train_images)) + str(label) + ' ' +
            path.basename(img_path))

    logger.info('Genre is null: ' + str(null_genre))
    logger.info('Label is null: ' + str(null_label))
    logger.info('Finished processing all images')


if __name__ == '__main__':
    main()
