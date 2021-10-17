# https://hub.docker.com/r/cwaffles/openpose
FROM nvidia/cuda:11.4.2-cudnn8-devel-ubuntu20.04

#get deps
RUN apt update && apt upgrade -y
RUN DEBIAN_FRONTEND=noninteractive apt-get install software-properties-common -y
RUN add-apt-repository universe
RUN add-apt-repository multiverse
RUN add-apt-repository restricted
RUN apt update
RUN apt-get install -y \
python3-dev python3-pip git g++ wget make libprotobuf-dev protobuf-compiler libopencv-dev \
libgoogle-glog-dev libboost-all-dev libhdf5-dev libatlas-base-dev


#for python api
RUN pip3 install tensorflow==2.4.0 opencv-python

#replace cmake as old version has CUDA variable bugs
RUN wget https://github.com/Kitware/CMake/releases/download/v3.16.0/cmake-3.16.0-Linux-x86_64.tar.gz && \
tar xzf cmake-3.16.0-Linux-x86_64.tar.gz -C /opt && \
rm cmake-3.16.0-Linux-x86_64.tar.gz
ENV PATH="/opt/cmake-3.16.0-Linux-x86_64/bin:${PATH}"

#get openpose
WORKDIR /openpose
RUN git clone https://github.com/CMU-Perceptual-Computing-Lab/openpose.git .

#build it
WORKDIR /openpose/build
RUN cmake -DBUILD_PYTHON=ON .. && make -j `nproc`
WORKDIR /openpose/build/python/openpose
RUN make install

RUN cp ./pyopenpose.cpython-38-x86_64-linux-gnu.so /usr/local/lib/python3.8/dist-packages
WORKDIR /usr/local/lib/python3.8/dist-packages
RUN ln -s pyopenpose.cpython-38-x86_64-linux-gnu.so pyopenpose
ENV LD_LIBRARY_PATH=/openpose/build/python/openpose
WORKDIR /openpose