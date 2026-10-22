FROM python:3.12
WORKDIR /opt/software
COPY . .
RUN  pip install --no-cache-dir .
WORKDIR /data
ENTRYPOINT [ "disvis" ]
