FROM python-sdk

LABEL MAINTAINER="ssi@mit.edu"

USER root

# Python packages.
RUN curl "https://repo.anaconda.com/archive/Anaconda3-2021.05-Linux-x86_64.sh" -o "install.sh" && chmod +x install.sh && ./install.sh -b && eval "$(/root/anaconda3/bin/conda shell.bash hook)" && eval /root/anaconda3/bin/conda init
RUN /root/anaconda3/bin/conda install -c conda-forge pydicom
RUN /root/anaconda3/bin/conda install -y -c frankong sigpy

# Paths.
ENV LD_LIBRARY_PATH=${LD_LIBRARY_PATH}:/root/anaconda3/lib/
ENV PATH=/root/anaconda3/bin/:${PATH}

# Copy.
COPY GERecon.so /
COPY entrypoint.sh /

# Main file.
COPY scan_archive_io.py /

# Entrypoint.
ENTRYPOINT ["/entrypoint.sh"]
