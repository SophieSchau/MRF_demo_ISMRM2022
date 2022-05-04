FROM continuumio/miniconda3

WORKDIR /recon

RUN conda install -c conda-forge numpy
RUN conda install pytorch torchvision torchaudio cudatoolkit=11.3 -c pytorch
RUN conda install -c conda-forge cudnn nccl
RUN conda install -c conda-forge nipype

COPY networks.py ./
COPY params.py ./
COPY preprocess.py ./
COPY synthesize.py ./
COPY main.py ./

ENTRYPOINT ["python3", "main.py"]
