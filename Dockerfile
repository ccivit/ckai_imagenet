FROM nvidia/cuda:10.1-base

RUN apt-get update
# RUN apt-get install --allow-downgrades --allow-remove-essential --allow-change-held-packages -y libcudnn7
RUN apt-get install -y python3-dev python3-pip python3-nose python3-numpy python3-scipy apt-utils
RUN python3 -m pip install --upgrade pip
RUN python3 -m pip install tensorflow Pillow PyYAML

WORKDIR /imagenet

CMD ["/bin/bash"]
