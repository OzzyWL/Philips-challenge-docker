FROM python:3.6-stretch

## Put everything in current folder
WORKDIR /src/object_detection_ozzy

## For installing python dependencies
COPY requirements.txt /src
## All files needed
COPY detect_object.py /src
COPY validation-images/ /src
COPY weights.data-00000-of-00002 /src
COPY weights.data-00001-of-00002 /src
COPY weights.index /src

RUN ls -la /src/* 
RUN pip install --no-cache-dir -r /src/requirements.txt

## Using unbuffered output with -u
CMD [ "python3", "-u" ,"/src/detect_object.py" ]
