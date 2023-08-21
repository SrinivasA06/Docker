# start by pulling the python image
FROM ubuntu:latest

# copy the requirements file into the image
# COPY ./requirements.txt /app/requirements.txt

# switch working directory
WORKDIR /app

# install the dependencies and packages in the requirements file
RUN apt update
RUN apt-get install ffmpeg libsm6 libxext6  -y
RUN apt install python3 -y
RUN apt install --upgrade python3-pip -y
RUN pip3 install scipy
RUN pip3 install --upgrade opencv-python==4.6.0.66
RUN pip3 install --upgrade opencv-contrib-python==4.6.0.66
RUN pip3 install --upgrade tensorflow-gpu

# copy every content from the local file to the image
COPY . /app


CMD [ "python3", "object_tracking.py" ]