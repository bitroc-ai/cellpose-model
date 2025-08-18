# Use CUDA-enabled PyTorch base image
FROM pytorch/pytorch:2.1.0-cuda11.8-cudnn8-runtime

# Set working directory
WORKDIR /app

# Accept proxy arguments
ARG http_proxy
ARG https_proxy
ARG no_proxy

# Configure timezone and package installation
ENV DEBIAN_FRONTEND=noninteractive
ENV TZ=UTC
RUN ln -snf /usr/share/zoneinfo/$TZ /etc/localtime && echo $TZ > /etc/timezone

# Install system dependencies
RUN apt-get update && apt-get install -y \
    git \
    wget \
    ffmpeg \
    libsm6 \
    libxext6 \
    libfontconfig1 \
    libxrender1 \
    libgl1-mesa-glx \
    && rm -rf /var/lib/apt/lists/*

# Install Python dependencies
RUN pip install --no-cache-dir \
    cellpose \
    opencv-python \
    scikit-image \
    matplotlib \
    tqdm \
    natsort

# Create directories for data and models
RUN mkdir -p /data /models

# Copy training script
COPY train_cellpose.py /app/

# Set environment variables for GPU
ENV CUDA_VISIBLE_DEVICES=0
ENV PYTHONUNBUFFERED=1

# Default command to run training
CMD ["python", "train_cellpose.py"]
