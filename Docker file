FROM python:3.10-slim

# Install system packages needed by OpenCV and pillow
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender1 \
    wget \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Copy requirements and install Python deps (minus tensorflow)
COPY requirements.txt /app/requirements.txt
RUN pip install --upgrade pip
RUN pip install -r /app/requirements.txt

# Install TensorFlow (specific version compatible with python 3.10)
RUN pip install tensorflow==2.15.0

# Copy app files
COPY . /app

# Expose the port Render will provide via $PORT
ENV PORT 10000
EXPOSE 10000

# Use gunicorn to serve the app. Bind to 0.0.0.0:$PORT
CMD exec gunicorn --bind 0.0.0.0:$PORT app:app
