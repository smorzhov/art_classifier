FROM bvlc/caffe:gpu

# Copy the file into docker
COPY requirements.txt requirements.txt

# Run the copied file
RUN pip install -r requirements.txt && \
    rm requirements.txt && \
    apt-get update && \
    apt-get install -y graphviz

WORKDIR /art_classifier

VOLUME ["/art_classifier"]