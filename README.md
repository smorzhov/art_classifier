# Art classifier

## Description

It classifies paintings into 6 genres:
* abstract
* genre painting
* landscape
* portrait
* history painting
* still life.

Train and test data was taken from [Painter by number](https://www.kaggle.com/c/painter-by-numbers) Kaggle competition.

First of all, it is worth noting that original data set contains 42 genres. Some of them includes very few paintings (less than 20) and some of them, such as portrait, has more than 10000. That is why I decided to merge some genres into one. For example, I consider self-portrait is just a portrait; cityscape, cloudscape or marina is just a landscape. In addition, I decided not to consider some genres. Original data contains "sketch and study" genre, which can include both sketches of portraits, landscapes, religious paintings and still lifes. For more details about merged and discarded genres, please, see [labels.json](/src/labels.json). That is why, data preprocessing stage becomes a little bit tricky.

I do not have enough time for the fine tuning of the convolutional neural network (CNN) used while training. That is why, my result, I suppose, is not perfect, despite even the fact that the training data may have been labeled badly or the problem to determine painting genres is difficult.

I suppose VGG_19_prelu CNN model (was take from the [VGG_19_layers_Network](https://github.com/n3011/VGG_19_layers_Network)) being quite promising and it may be tuned much better.

I got approximately 70% accuracy during the testing phase. The learning curve and the confusion matrix are as follows:

![Training curve](/imgs/vgg19_learning_curve.png)

Obviously, the learning curve is unstable in the tail and hence, CNN parameters must be carefully tuned to solve this issue.

![Confusion matrix](/imgs/confusion_matrix.png)

For training VGG_19_prelu model I used a cluster of 8 NVIDIA Tesla V100 SMX2 GPUs. Approximately 12 Gb of video memory were used on each GPU. The training process took about 6 hours.

Caffe_model_1 has not so complex architecture as VGG_19_prelu and may be trained on one GPU in several hours' time. Using this simpler model, I cannot get above 60% accuracy.

If you were interested in this problem and you improved my result, please inform me about it. Feel free to communicate with me via [email](mailto:smorzhov@gmail.com) or [LinkedIn](https://www.linkedin.com/in/smorzhov/).

## Prerequisites

You will need the following things properly installed on your computer.

* [Python](https://www.python.org/)
* [Docker](https://docs.docker.com/engine/installation/linux/ubuntulinux/) (v17 or higher)
* [Nvidia-Docker](https://github.com/NVIDIA/nvidia-docker)

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
    nvidia-docker build -t sm_caffe:gpu .
    ```
    Run container
    ```bash
    nvidia-docker run -v $PWD/src:/art_classifier -dt --name art sm_caffe:gpu /bin/bash
    ```
5. Create lmdb data
    ```bash
    nvidia-docker exec art python create_lmdb.py
    ```
6. Generate the mean image of training data
    ```bash
    nvidia-docker exec art compute_image_mean -backend=lmdb input/train_lmdb input/mean.binaryproto
    ```
7. Model training for caffenet
    ```bash
    nvidia-docker exec art caffe train --solver caffe_models/caffe_model_1/solver_1.prototxt 2>&1 | tee model_train.log
    ```
    for VGG_19_prelu
    ```bash
    nvidia-docker exec art caffe train --solver caffe_models/VGG_19_prelu/VGG_19_prelu_solver.prototxt --gpu=all 2>&1 | tee model_train.log
    ```
8. Plotting the learning (firstly, you need to copy model_train.log into `./src` directory)
    ```bash
    nvidia-docker exec art python plot_learning_curve.py /art_classifier/model_train.log /art_classifier/model_1_learning_curve.png
    ```
9. Prediction on new data

    ```bash
    nvidia-docker exec art python make_predictions.py [-h]
    ```
    for VGG_19_prelu
    ```bash
    nvidia-docker exec art python make_predictions.py -a caffe_models/VGG_19_prelu/VGG_19_prelu_deploy.prototxt -w caffe_models/VGG_19_prelu/ -w caffe_models/VGG_19_prelu/VGG_19_prelu_iter_80000.caffemodel
    ```

Optionally you can print the model architecture by executing the command below. The model architecture image will be stored under `~/art_classifier/caffe_models/caffe_model_1/caffe_model_1.png` 
```bash
nvidia-docker exec art python /opt/caffe/python/draw_net.py /art_classifier/caffe_models/caffe_model_1/caffenet_train_val_1.prototxt /art_classifier/caffe_models/caffe_model_1/caffe_model_1.png
``` 

## Acknowledgments

I would like to thank [Leonid Ivanovsky](https://www.linkedin.com/in/leonid-ivanovsky-2b64ba127/) and [Yaroslavl State University](http://uniyar.ac.ru/) for their help and provision of the infrastructure.
