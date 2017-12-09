# Art classifier

## Prerequisites

You will need the following things properly installed on your computer.

* [Python](https://www.python.org/)

## Installation

* `git clone https://github.com/smorzhov/art_classifier.git`

## Running

1. First of all, you need the data to train. You can download it [here](https://drive.google.com/file/d/1uSz9xfYQD3VSN17wlxdGZ6yDpO5uWz6A/view?usp=sharing). Also, you can download train and test data as tgz archives (smaller size) with the following commands:
    ```bash
    wget --no-check-certificate "https://onedrive.live.com/download?cid=9B1DCE6B8AAEBBAB&resid=9B1DCE6B8AAEBBAB%211094&authkey=ALTTp6IUBu8v4v4" -O test.tgz;wget --no-check-certificate "https://onedrive.live.com/download?cid=9B1DCE6B8AAEBBAB&resid=9B1DCE6B8AAEBBAB%211095&authkey=ACicffxzKxa9D1U" -O train.tgz;
    ```
2. At the top of the `~art_classifier/src` folder unrar archive with train and test data
    ```bash
    unrar x data.rar
    ```
    Or at the top of `~art_classifier/src/data`
    ```bash
    tar -xvf train.tgz;tar -xvf test.tgz;
    ```
3. If you are planning to use nvidia-docker, you need to building nvidia-docker image first. Otherwise, you can skip this step
    ```bash
    nvidia-docker build -t sm_caffe_keras:gpu .
    ```
    Run container
    ```bash
    nvidia-docker run -v $PWD/src:/art_classifier -dt --name art sm_caffe_keras:gpu /bin/bash
    ```
5. Create leveldb data

    Docker
    ```bash
    nvidia-docker exec art python create_leveldb.py
    ```
    Native
    ```bash
    python create_leveldb.py
    ```
6. Generate the mean image of training data

    Docker
    ```bash
    nvidia-docker exec art compute_image_mean -backend=leveldb input/train_leveldb input/mean.binaryproto
    ```
    Native
    ```bash
    /home/lebedev/caffe/build/tools/compute_image_mean -backend=leveldb /home/ivanovskii/workspace/art_classifier/src/input/train_leveldb /home/ivanovskii/workspace/art_classifier/src/input/mean.binaryproto
    ```
7. Model training

    Docker
    ```bash
    nvidia-docker exec art caffe train --solver caffe_models/caffe_model_1/solver_1.prototxt 2>&1 | tee caffe_models/caffe_model_1/model_1_train.log
    ```
    Native
    ```bash
    /home/lebedev/caffe/build/tools/caffe train --solver /home/ivanovskii/workspace/art_classifier/src/caffe_models/caffe_model_1/solver_1.prototxt 2>&1 | tee /home/ivanovskii/workspace/art_classifier/src/caffe_models/caffe_model_1/model_1_train.log
    ```
8. Plotting the learning 

    Docker
    ```bash
    nvidia-docker exec art python plot_learning_curve.py caffe_models/caffe_model_1/model_1_train.log caffe_models/caffe_model_1/caffe_model_1_learning_curve.png
    ```
    Native
    ```bash
    python /home/ivanovskii/workspace/art_classifier/src/plot_learning_curve.py /home/ivanovskii/workspace/art_classifier/src/caffe_models/caffe_model_1/model_1_train.log /home/ivanovskii/workspace/art_classifier/src/caffe_models/caffe_model_1/caffe_model_1_learning_curve.png
    ```
9. Prediction on new data

    Docker
    ```bash
    nvidia-docker exec art python make_predictions.py [-h]
    ```
    Native
    ```bash
    python make_predictions.py [-h]
    ```

Optionally you can print the model architecture by executing the command below. The model architecture image will be stored under `~/art_classifier/caffe_models/caffe_model_1/caffe_model_1.png` 

Docker
```bash
nvidia-docker exec art python /opt/caffe/python/draw_net.py /art_classifier/caffe_models/caffe_model_1/caffenet_train_val_1.prototxt /art_classifier/caffe_models/caffe_model_1/caffe_model_1.png
``` 
Native
```bash
python /home/lebedev/caffe/python/draw_net.py /home/ivanovskii/workspace/art_classifier/src/caffe_models/caffe_model_1/caffenet_train_val_1.prototxt /home/ivanovskii/workspace/art_classifier/src/caffe_models/caffe_model_1/caffe_model_1.png
```
