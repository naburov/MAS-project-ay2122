FROM tensorflow/tensorflow:2.4.0-gpu

RUN apt-get --yes install wget
RUN apt-key del A4B469963BF863CC
RUN apt-key adv --fetch-keys https://developer.download.nvidia.com/compute/cuda/repos/ubuntu1804/x86_64/3bf863cc.pub

RUN apt-get --yes -qq update --allow-insecure-repositories \
 && apt-get --yes -qq upgrade \
 && apt-get --yes -qq install \
                      bzip2 \
                      cmake \
                      cpio \
                      curl \
                      g++ \
                      gcc \
                      gfortran \
                      git \
                      gosu \
                      libblas-dev \
                      liblapack-dev \
                      virtualenv \
                      wget \
                      zlib1g-dev

RUN mkdir -p /openmpi/downloads
RUN wget -O  /openmpi/downloads/openmpi-4.1.4.tar.gz https://www.open-mpi.org/software/ompi/v4.1/downloads/openmpi-4.1.4.tar.gz
RUN tar  -xvzf /openmpi/downloads/openmpi-4.1.4.tar.gz -C /openmpi/downloads

WORKDIR "/openmpi/downloads/openmpi-4.1.4"
RUN ./configure --prefix="/openmpi"
RUN make -j2
RUN make install

ENV CONDA_DIR /opt/conda
COPY Anaconda3-5.3.1-Linux-x86_64.sh "~/Anaconda3-5.3.1-Linux-x86_64.sh"
RUN /bin/bash "~/Anaconda3-5.3.1-Linux-x86_64.sh" -b -p /opt/conda

# Put conda in path so we can use conda activate
ENV PATH=$CONDA_DIR/bin:$PATH
ENV PATH=/openmpi/bin:$PATH
ENV PATH=/openmpi/lib:$PATH

RUN mkdir -p /app
RUN mkdir -p /app/mas-project-burov-ay2122
RUN mkdir -p /app/mas-project-burov-ay2122/Trainers

COPY environment.yml "/app/mas-project-burov-ay2122"


WORKDIR "/app/mas-project-burov-ay2122"
RUN conda env create -f environment.yml
RUN pip install --upgrade pip
RUN pip install mpi4py

#RUN pip install -r rnts.txt


