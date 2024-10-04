FROM ubuntu:22.04
LABEL authors="johnbellamy"

USER root
RUN mkdir /app
WORKDIR /app

ADD requirements /app/requirements
ADD app.py /app/app.py


RUN apt-get update && \
    apt-get install -y software-properties-common && \
    add-apt-repository -y ppa:deadsnakes/ppa && \
    DEBIAN_FRONTEND=noninteractive TZ=Etc/UTC apt-get install -y python3.11 python3-pip

RUN pip install --no-cache-dir  -r requirements

CMD ["streamlit", "run", "app.py"]