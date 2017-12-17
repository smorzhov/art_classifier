"""
This script makes predictions using the pretrained model and
generates a submission file.

Usage: python make_predictions.py [-h]
"""

import argparse
from glob import glob
from os import path, system
import pandas as pd
import numpy as np
import progressbar
from utils import CWD, DATA_PATH, get_logger, MODE, CAFFE_MODELS_PATH
from utils import get_genre_labels, transform_img


def get_model_data(mean_file, model_arc, model_weights, device=0):
    """
    Reading mean image, caffe model and its weights
    """
    import caffe
    from caffe.proto import caffe_pb2

    if MODE == 'gpu':
        caffe.set_mode_gpu()
        caffe.set_device(device)
    else:
        caffe.set_mode_cpu()

    # Read mean image
    mean_blob = caffe_pb2.BlobProto()
    with open(mean_file) as file:
        mean_blob.ParseFromString(file.read())
    mean_array = np.asarray(
        mean_blob.data, dtype=np.float).reshape(
            (mean_blob.channels, mean_blob.height, mean_blob.width))

    # Read model architecture and trained model's weights
    net = caffe.Net(model_arc, model_weights, caffe.TEST)
    # net = caffe.Net(model_arc, caffe.TEST, weights=model_weights)

    # Define image transformers
    transformer = caffe.io.Transformer({'data': net.blobs['data'].data.shape})
    transformer.set_mean('data', mean_array)
    transformer.set_transpose('data', (2, 0, 1))
    return net, transformer


def predict(net, transformer, img_path):
    """
    Making predictions
    """
    logger = get_logger(path.splitext(path.basename(__file__))[0] + '.log')

    # Reading image paths
    test_images = [img_path]
    if path.isdir(img_path):
        test_images = [img for img in glob(path.join(img_path, '*.jpg'))]
    test_ids = []
    predictions = []
    # Making predictions
    for img_path in test_images:
        img = transform_img(img_path)

        net.blobs['data'].data[...] = transformer.preprocess(
            'data', np.array(img))
        out = net.forward()
        pred_probas = out['prob']

        test_ids.append(path.basename(img_path))
        predictions = predictions + [pred_probas.argmax()]

        logger.info(img_path)
        logger.info(pred_probas.argmax())
    return test_ids, predictions


def make_submission_file(submission_model_path, test_ids, predictions):
    """
    Making submission file
    """
    with open(submission_model_path, 'w') as file:
        file.write('img,label\n')
        for test_id, prediction in zip(test_ids, predictions):
            file.write(str(test_id) + "," + str(prediction) + "\n")
    file.close()


def analyze_predictions(submission_model_path, csv_result):
    """It analyze predictions and shows result"""
    all_data_info = pd.read_csv(path.join(DATA_PATH, 'all_data_info.csv'))
    submission_model = pd.read_csv(submission_model_path)

    genre_label = get_genre_labels(True)
    null_genre = 0
    null_label = 0
    exact_labels = []
    errors = 0
    progress = progressbar.ProgressBar(
        maxval=len(submission_model),
        widgets=[progressbar.Bar('=', '[', ']'), ' ',
                 progressbar.Percentage()])
    progress.start()
    for i, row in submission_model.iterrows():
        progress.update(i + 1)
        genre = all_data_info[all_data_info['new_filename'] == row['img']][
            'genre'].dropna()
        # some paintings don't have a genre. Checking it
        if len(genre) < 1:
            null_genre += 1
            exact_labels.append(row['label'])
            continue
        if genre.values[0] not in genre_label:
            # No label. It's strange, but let's go on...
            null_label += 1
            exact_labels.append(row['label'])
            continue
        exact_labels.append(int(genre_label[genre.values[0]]['label']))
        if row['label'] != exact_labels[-1]:
            errors += 1
    progress.finish()

    submission_model['exact_label'] = exact_labels
    submission_model.to_csv(csv_result, index=False)

    # plot_confusion_matrix(submission_model)

    print('Genre is null: ' + str(null_genre))
    print('Label is null: ' + str(null_label))
    print('Errors: ' + str(errors) + ' (out of ' + str(len(submission_model)) +
          ')')
    print('Accuracy: ' + str(1 - (errors / float(len(submission_model)))))


def init_argparse():
    """Initializes argparse"""
    parser = argparse.ArgumentParser(description='Make predictions')
    parser.add_argument(
        '-m',
        '--mean',
        nargs='?',
        help='mean file path (mean.binaryproto)',
        default=path.join(CWD, 'input', 'mean.binaryproto'),
        type=str)
    parser.add_argument(
        '-a',
        '--architecture',
        nargs='?',
        help='prototxt file path with mode architecture',
        default=path.join(CAFFE_MODELS_PATH, 'caffe_model_1',
                          'caffenet_deploy_1.prototxt'),
        type=str)
    parser.add_argument(
        '-d',
        '--device',
        nargs='?',
        help='Number of GPU device',
        default=0,
        type=int)
    parser.add_argument(
        '-w',
        '--weights',
        nargs='?',
        help='weight file path',
        default=path.join(CAFFE_MODELS_PATH, 'caffe_model_1',
                          'caffe_model_1_iter_40000.caffemodel'),
        type=str)
    parser.add_argument(
        '-f',
        '--file',
        nargs='?',
        default=path.join(DATA_PATH, 'test'),
        help=
        'Path to image or directory. In case of directory only jpg images will be read)',
        type=str)
    return parser


def label_to_class_name(label):
    """Transforms label to class name"""
    try:
        genre_label = pd.read_csv(path.join(DATA_PATH, 'genre_labels.csv'))
        return genre_label[genre_label['label'] == int(label)]['genre'].values[
            0]
    except IOError:
        return label


def main():
    """Main function"""
    submission_model_path = path.join(CAFFE_MODELS_PATH, 'caffe_model_1',
                                      'submission_model_1.csv')
    parser = init_argparse()
    args = parser.parse_args()

    net, transformer = get_model_data(args.mean, args.architecture,
                                      args.weights)
    test_ids, predictions = predict(net, transformer, args.file)
    if len(predictions) == 1:
        print(test_ids[0], label_to_class_name(predictions[0]))
        return
    make_submission_file(submission_model_path, test_ids, predictions)
    predictions_path = path.join(path.dirname(args.weights), 'predictions.csv')
    print('Analizing predictions')
    analyze_predictions(submission_model_path, predictions_path)
    system('python plot_confusion_matrix.py ' + predictions_path)


if __name__ == '__main__':
    main()
