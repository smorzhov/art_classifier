# Art classifier

## Prerequisites

You will need the following things properly installed on your computer.

* [Python](https://www.python.org/)

## Installation

* `git clone https://github.com/smorzhov/art_classifier.git`
* `cd art_classifier`
* `pip install -r requirements.txt`

## Running

1. First of all, you need data to train. You can download it [here](https://drive.google.com/file/d/1uSz9xfYQD3VSN17wlxdGZ6yDpO5uWz6A/view?usp=sharing). Also, you can download train and test data as tgz archives (smaller size) with the following commands:
```bash
wget --no-check-certificate "https://onedrive.live.com/download?cid=9B1DCE6B8AAEBBAB&resid=9B1DCE6B8AAEBBAB%211094&authkey=ALTTp6IUBu8v4v4" -O test.tgz;wget --no-check-certificate "https://onedrive.live.com/download?cid=9B1DCE6B8AAEBBAB&resid=9B1DCE6B8AAEBBAB%211095&authkey=ACicffxzKxa9D1U" -O train.tgz;
```
2. At the top of the `art_classifier` folder unrar archive with train and test data.
    ```bash
    unrar x data.rar` or `unp data.rar
    ```
3. Create leveldb data
    ```bash
    python create_leveldb.py
    ```
4. Generate the mean image of training data
    ```bash
    /home/lebedev/caffe/build/tools/compute_image_mean -backend=leveldb /home/ivanovskii/workspace/art_classifier/input/train_leveldb /home/ivanovskii/workspace/art_classifier/input/mean.binaryproto
    ```
5. Model training
    ```bash
    /home/lebedev/caffe/build/tools/caffe train --solver /home/ivanovskii/workspace/art_classifier/caffe_models/caffe_model_1/solver_1.prototxt 2>&1 | tee /home/ivanovskii/workspace/art_classifier/caffe_models/caffe_model_1/model_1_train.log
    ```
6. Plotting the learning curves
    ```bash
    python /home/ivanovskii/workspace/art_classifier/plot_learning_curve.py /home/ivanovskii/workspace/art_classifier/caffe_models/caffe_model_1/model_1_train.log /home/ivanovskii/workspace/art_classifier/caffe_models/caffe_model_1/caffe_model_1_learning_curve.png
    ```
7. Prediction on new data
    ```bash
    python make_predictions.py
    ```
    For more details how to use this script you can run
    ```bash
    python make_predictions.py -h
    ```

Optionally you can print the model architecture by executing the command below. The model architecture image will be stored under `~/art_classifier/caffe_models/caffe_model_1/caffe_model_1.png` 

```bash
python /home/lebedev/caffe/python/draw_net.py /home/ivanovskii/workspace/art_classifier/caffe_models/caffe_model_1/caffenet_train_val_1.prototxt /home/ivanovskii/workspace/art_classifier/caffe_models/caffe_model_1/caffe_model_1.png
``` 
