FROM  nvcr.io/nvidia/caffe:17.10

# Copy the file into docker
COPY requirements.txt requirements.txt

# Run the copied file
RUN pip install -r requirements.txt && \
    pip install git+https://github.com/aleju/imgaug && \
    apt-get update && \
    apt-get install -y graphviz && \ 
    apt-get clean && \
    rm requirements.txt && \
    rm -rf /var/lib/apt/lists/*

WORKDIR /art_classifier

VOLUME ["/art_classifier"]