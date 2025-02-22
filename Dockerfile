FROM ubuntu:21.04
ARG DEBIAN_FRONTEND=noninteractive
RUN apt-get update -y && apt-get install wget -y
RUN apt-get update -y && apt-get install -y python3 python3-pip python3-dev git && apt-get autoclean -y
#RUN apt-get update && apt-get install sqlite3 libsqlite3-dev -y
#RUN ln -s $(which pip3) /usr/bin/pip
RUN pip install --upgrade pip

#RUN make /app
COPY ./ /graphmb/
#COPY ./data/strong100/ /graphmb/data/strong100/
WORKDIR /graphmb
RUN pip install .
#CMD python /app/app.py