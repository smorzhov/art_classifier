FROM bvlc/caffe:gpu as caffe

# Copy the file into docker
COPY requirements.txt requirements.txt

# Run the copied file
RUN pip install -r requirements.txt && \
    rm requirements.txt && \
    apt-get update && \
    apt-get install -y graphviz && \ 
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*

FROM gw000/keras:2.1.1-py2-tf-gpu

WORKDIR /art_classifier

VOLUME ["/art_classifier"]