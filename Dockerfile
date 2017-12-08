# This is the docker image available. I am using cpu version here. If needed there is gpu version available.
FROM bvlc/caffe:cpu

# Copy the file into docker
COPY requirements.txt requirements.txt

# Run the copied file
RUN pip install -r requirements.txt && \
    rm requirements.txt && \
    apt-get update && \
    apt-get install -y graphviz

WORKDIR /art_classifier

VOLUME ["/art_classifier"]