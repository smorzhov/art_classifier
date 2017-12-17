"""
Plots submission matrix

Usage python plot_confusion_matrix.py predictions_path
"""
import sys
from os import path
import itertools
import matplotlib
import pandas as pd
import numpy as np
# generates images without having a window appear
matplotlib.use('Agg', warn=False, force=True)
import matplotlib.pylab as plt
from sklearn.metrics import confusion_matrix
from utils import DATA_PATH


def plot(cm,
         classes,
         normalize=False,
         title='Confusion matrix',
         cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]

    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(
            j,
            i,
            format(cm[i, j], fmt),
            horizontalalignment="center",
            color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')


submission_model = pd.read_csv(sys.argv[1])
labels = pd.read_csv(path.join(DATA_PATH, 'genre_labels.csv'))
# Compute confusion matrix
cnf_matrix = confusion_matrix(submission_model['exact_label'].values,
                              submission_model['label'].values)
np.set_printoptions(precision=2)
# Plot normalized confusion matrix
plt.figure()
plot(
    cnf_matrix,
    classes=labels['genre'].values,
    normalize=True,
    title='Normalized confusion matrix')
# Saving learning curve
plt.savefig(path.join(path.dirname(sys.argv[1]), 'confusuin_matrix'))
