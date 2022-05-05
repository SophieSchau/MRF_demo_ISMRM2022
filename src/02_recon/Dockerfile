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
RUN conda install -c conda-forge sympy
RUN pip install git+https://github.com/mikgroup/sigpy.git@master

# SMILR
RUN pip3 install torch==1.9.1+cu111 torchvision==0.10.1+cu111 torchaudio==0.9.1 -f https://download.pytorch.org/whl/torch_stable.html
RUN pip3 install pytorch-lightning==1.5.0
RUN conda install -c conda-forge setuptools==59.5.0

RUN mkdir smilr
COPY smilr/* smilr/

COPY load_data.py ./
COPY autofov.py ./
COPY mrf.py ./
COPY llr.py ./
COPY optalg.py ./
COPY optpoly.py ./
COPY nn.py ./
COPY main.py ./

ENTRYPOINT ["python3", "main.py"]
