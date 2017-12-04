# This is the docker image available. I am using cpu version here. If needed there is gpu version available.
FROM bvlc/caffe:cpu

# Copy the file into docker
COPY requirements.txt requirements.txt

# Run the copied file
RUN pip install -r requirements.txt && \
    rm requirements.txt

# create a folder called art_classifier
# and copy some files (folders) into that folder
ADD . /art_classifier

WORKDIR /art_classifier

# Create volumes
VOLUME ["/caffe_models", "/data", "create_leveldb.py",
    "make_predictions.py", "plot_learning_curve.py", "utils.py"]
