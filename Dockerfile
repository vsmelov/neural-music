# use with nvidia-docker
FROM nvidia/cuda:7.5-cudnn5-runtime-ubuntu14.04
MAINTAINER Smelov Vladimir <vladimirfol@gmail.com>

RUN apt-get update
RUN apt-get -y install wget

# solving this problem - http://askubuntu.com/questions/831386/gpgkeys-key-f60f4b3d7fa2af80-not-found-on-keyserver
RUN wget -qO - http://developer.download.nvidia.com/compute/cuda/repos/ubuntu1404/x86_64/7fa2af80.pub | sudo apt-key add -
RUN apt-get update

RUN apt-get -y install python-pip python-dev

#RUN apt-get -y install lame libmp3lame0  # mp3 processing
RUN apt-get -y install sox libsox-fmt-all  # the Swiss Army knife of sound processing programs and additional libs for different audio-formats
#RUN pip install sox  # Python wrapper around sox

RUN apt-get -y install python-numpy python-scipy
#RUN pip install Keras

# tensorflow for python3.4
# RUN export TF_BINARY_URL=https://storage.googleapis.com/tensorflow/linux/gpu/tensorflow-0.10.0-cp34-cp34m-linux_x86_64.whl
# RUN sudo pip3 install --upgrade $TF_BINARY_URL

# tensorflow for python2.7
RUN pip install --upgrade https://storage.googleapis.com/tensorflow/linux/gpu/tensorflow-0.11.0rc0-cp27-none-linux_x86_64.whl

# Theano (may be useful for Theano backend in future)
# ubuntu14.04 for python2 - http://deeplearning.net/software/theano/install_ubuntu.html#install-ubuntu
RUN apt-get -y install python-numpy python-scipy python-dev python-pip python-nose g++ libopenblas-dev git
RUN pip install Theano

# for save and load weights
RUN apt-get -y install libhdf5-dev
RUN pip install h5py
RUN pip install stft
RUN apt-get install -y git

# https://developer.nvidia.com/rdp/cudnn-download
# RUN wget https://developer.nvidia.com/compute/machine-learning/cudnn/secure/v5.1/prod/7.5/libcudnn5_5.1.3-1+cuda7.5_amd64-deb
# RUN dpkg -i libcudnn5_5.1.3-1+cuda7.5_amd64-deb
# RUN apt-get install -f

# to use Keras visualize tools
RUN pip install pyparsing==1.5.7
RUN pip install pydot==1.0.28

# old unsupported useful stuff
# RUN pip install git+https://github.com/wuaalb/keras_extensions

RUN pip install cython  # use C-lib in python
RUN apt-get install -y libyaml-dev

RUN git clone https://github.com/fchollet/keras.git
# WARN: keras bug!
# TODO: wait until fix, and remove this
# https://github.com/fchollet/keras/commit/99bd066f38ac9603a5c00b2eab57f6d15412ddc2
RUN sed -i "116s/.*/            if K.backend() == 'tensorflow':/" /keras/keras/layers/wrappers.py
RUN cd /keras && python setup.py install

# helps keras to find libcudnn
RUN ln -s /usr/lib/x86_64-linux-gnu/libcudnn.so.5 /usr/lib/x86_64-linux-gnu/libcudnn.so
RUN export LD_LIBRARY_PATH=/usr/lib/x86_64-linux-gnu:$LD_LIBRARY_PATH
