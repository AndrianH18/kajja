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

