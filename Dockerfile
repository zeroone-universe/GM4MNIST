# 베이스 이미지로 python:3.8을 사용합니다.
FROM pytorch/pytorch:2.0.0-cuda11.7-cudnn8-runtime

RUN apt-get update && apt-get upgrade -y
COPY requirements.txt .

RUN pip install --upgrade pip

RUN pip install -r requirements.txt
RUN rm requirements.txt

RUN apt update 
RUN apt install -y tmux htop ncdu 
RUN apt clean 
RUN apt autoremove 
RUN rm -rf /var/lib/apt/lists/* /tmp/* 

WORKDIR /root/GM4MNIST
# mount this repository path to /root/GM4MNIST when running container, like
# docker run -itd -v /home/GM4MNIST:/root/GM4MNIST --runtime nvidia -e NVIDIA_VISIBLE_DEVICES=0,1,2,3 gm4mnist:latest