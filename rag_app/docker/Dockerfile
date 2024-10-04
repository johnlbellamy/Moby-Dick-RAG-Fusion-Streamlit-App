FROM ubuntu:22.04
LABEL authors="johnbellamy"

USER root
RUN mkdir /app
WORKDIR /app

ADD tools /app/tools
ADD utils /app/utils
ADD requirements /app/requirements
ADD docker/app.py /app/app.py
ADD docker/query.py /app/query.py

RUN apt-get update && \
    apt-get install -y software-properties-common && \
    add-apt-repository -y ppa:deadsnakes/ppa && \
    DEBIAN_FRONTEND=noninteractive TZ=Etc/UTC apt-get install -y python3.11 python3-pip

RUN pip install --no-cache-dir  -r requirements &&\
    pip install fastapi[standard]

CMD ["fastapi", "run", "app.py", "--port", "5600"]