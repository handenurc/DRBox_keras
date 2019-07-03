# Use a local image "candela-base" containing all the requirements
FROM nvidia/cuda:9.0-cudnn7-runtime

RUN apt-get update && apt-get install -y software-properties-common && apt-add-repository -y ppa:ubuntugis/ppa
RUN apt-get update && apt-get install -y gdal-bin libgdal-dev python3-gdal build-essential python3-pip && gdal-config --version

ENV CPLUS_INCLUDE_PATH=/usr/include/gdal
ENV C_INCLUDE_PATH=/usr/include/gdal

RUN apt-get update && apt-get install -y python3-dev gcc \
    && rm -rf /var/lib/apt/lists/*


# remove above


COPY requirements.txt .

#RUN pip install -r requirements.txt
RUN pip3 install --upgrade -r requirements.txt



RUN pip3 install tqdm opencv-python

USER root
RUN apt-get update
RUN apt-get install -y python3-tk x11-xserver-utils

# Run app.py when the container launches
WORKDIR /home
CMD ["python3", "Training_DRBox.py"]
