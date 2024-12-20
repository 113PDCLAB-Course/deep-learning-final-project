FROM tensorflow/tensorflow:latest-gpu

ENV NVIDIA_DISABLE_REQUIRE=1

WORKDIR /app

RUN apt-get update && apt-get install -y \
    git \
    wget \
    unzip \
    && rm -rf /var/lib/apt/lists/*

RUN pip install 'tensorflow[and-cuda]' tf_keras
RUN python3 -c "import tensorflow as tf; print(tf.config.list_physical_devices('GPU'))"

# Download packages
COPY requirements.txt requirements.txt
RUN pip install -r requirements.txt