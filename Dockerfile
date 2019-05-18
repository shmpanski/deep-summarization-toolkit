FROM pytorch/pytorch:1.1.0-cuda10.0-cudnn7.5-runtime

ENV LANG C.UTF-8
ENV LC_ALL C.UTF-8

COPY requirements.txt /deep-summarization-toolkit/
WORKDIR /deep-summarization-toolkit

RUN pip install -r requirements.txt
RUN python -m nltk.downloader punkt

COPY . /deep-summarization-toolkit/