FROM python:2.7.18
WORKDIR /opt/software
COPY . .
RUN  pip install numpy pyfftw && pip install  .
ENTRYPOINT [ "disvis" ]
