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

WORKDIR /art_classifier

VOLUME ["/art_classifier"]