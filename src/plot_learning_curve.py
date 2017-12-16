"""
This script generates learning curves for caffe models

Usage: python plot_learning_curve.py model_train.log ./caffe_model_learning_curve.png
"""

import sys
import subprocess
from os import system, path, chdir, remove
import pandas as pd
import matplotlib
# generates images without having a window appear
matplotlib.use('Agg')
import matplotlib.pylab as plt
from utils import CAFFE_PATH, CWD


def generate_logs(model_log_path, must_clear=True):
    """
    Returns training and test logs
    """

    def clear():
        """Removes generated files"""
        # remove(train_log_path)
        # remove(test_log_path)

    # Get directory where the model logs is saved, and move to it
    model_log_dir_path = path.dirname(model_log_path)
    chdir(model_log_dir_path)
    # Parsing training/validation logs
    command = 'python ' + CAFFE_PATH + '/tools/extra/parse_log.py ' + model_log_path + ' ' + CWD
    process = subprocess.Popen(command, shell=True, stdout=subprocess.PIPE)
    process.wait()
    # Read training and test logs
    train_log_path = model_log_path + '.train'
    test_log_path = model_log_path + '.test'
    train_log = pd.read_csv(train_log_path)
    test_log = pd.read_csv(test_log_path)
    if must_clear:
        clear()
    return train_log, test_log


def make_learning_curves(learning_curve_path, train_log, test_log):
    """
    Making learning curve
    """
    plt.style.use('ggplot')
    _, ax1 = plt.subplots()

    # Plotting training and test losses
    train_loss, = ax1.plot(
        train_log['NumIters'], train_log['loss'], color='red', alpha=.5)
    test_loss, = ax1.plot(
        test_log['NumIters'], test_log['loss'], linewidth=2, color='green')
    ax1.set_ylim(ymin=0, ymax=1)
    ax1.set_xlabel('Iterations', fontsize=15)
    ax1.set_ylabel('Loss', fontsize=15)
    ax1.tick_params(labelsize=15)
    # Plotting test accuracy
    ax2 = ax1.twinx()
    test_accuracy, = ax2.plot(
        test_log['NumIters'], test_log['accuracy'], linewidth=2, color='blue')
    ax2.set_ylim(ymin=0, ymax=1)
    ax2.set_ylabel('Accuracy', fontsize=15)
    ax2.tick_params(labelsize=15)
    # Adding legend
    plt.legend(
        [train_loss, test_loss, test_accuracy],
        ['Training Loss', 'Test Loss', 'Test Accuracy'],
        bbox_to_anchor=(1, 0.8))
    plt.title('Training Curve', fontsize=18)
    # Saving learning curve
    plt.savefig(learning_curve_path)


def main():
    """Main function"""
    train_log, test_log = generate_logs(sys.argv[1])
    make_learning_curves(sys.argv[2], train_log, test_log)


if __name__ == '__main__':
    main()