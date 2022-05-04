FROM continuumio/miniconda3

WORKDIR /recon

RUN conda install -c conda-forge numpy
RUN conda install -c conda-forge cupy
RUN conda install -c conda-forge cudnn
RUN conda install -c conda-forge nccl
RUN conda install -c conda-forge numba
RUN conda install -c conda-forge pywavelets
RUN conda install -c conda-forge tqdm
RUN conda install -c conda-forge scipy
RUN pip install git+https://github.com/mikgroup/sigpy.git@master

COPY rovir.py ./
COPY main.py ./

ENTRYPOINT ["python3", "main.py"]
