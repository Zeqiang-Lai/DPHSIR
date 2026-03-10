FROM pytorch/pytorch:1.11.0-cuda11.3-cudnn8-runtime

# Install system dependencies
ENV DEBIAN_FRONTEND=noninteractive
RUN apt-get update && apt-get install -y \
    libgl1-mesa-glx \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# Set working directory
WORKDIR /workspace

# Copy requirements and install
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy the rest of the repository
COPY . /workspace/

# Install the local dphsir package
RUN pip install -e .

# Set the default command to show help for run_inference.py
CMD ["python", "run_inference.py", "--help"]
