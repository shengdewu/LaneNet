FROM dl.nvidia/cuda:11.1-cudnn8-devel-torch1.10.0

COPY lanenet /home/lanenet/lanenet
copy util /home/lanenet/util
COPY whl /home/lanenet/whl
COPY train.py /home/lanenet/
COPY config /home/lanenet/config

RUN pip3 install /home/lanenet/whl/engine_frame-*.whl

WORKDIR /home/lanenet
ENTRYPOINT ["python3", "train.py"]