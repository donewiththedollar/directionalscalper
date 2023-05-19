FROM python:3.11

COPY ./ /code/

WORKDIR /code

RUN pip3 install -r requirements.txt
