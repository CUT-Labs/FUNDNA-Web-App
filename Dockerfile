# Use an official Python image from the Docker Hub
FROM python:3.11-slim

# Set the working directory inside the container
WORKDIR /app

# Install PostgreSQL dependencies, Clang, and other necessary build tools
RUN apt-get update && apt-get install -y \
    build-essential \
    libpq-dev \
    git \
    clang \
    wget \
    unzip \
    && rm -rf /var/lib/apt/lists/*

# Set environment variables to use Clang as the default compiler
ENV CC=gcc
ENV CXX=g++

# Copy the requirements files into the container
COPY requirements.txt /app/
COPY piperine_requirements.txt /app/

# Install Python dependencies
RUN pip install --upgrade pip && \
    pip install -r requirements.txt

# NUPACK Installation (manual process due to external dependency)
RUN echo "Installing NUPACK" && \
    wget https://jacksonhuse.com/wp-content/uploads/2024/10/nupack3.0.6.tar && \
    tar -xvf nupack3.0.6.tar && \
    cd nupack3.0.6 && \
    make clean && \
    make || true && \
    export NUPACKHOME="/app/nupack3.0.6" && \
    cd /app

# Set nupackhome variable
RUN export NUPACKHOME="/app/nupack3.0.6"

# Install Piperine-specific dependencies and uninstall conflicting ones (numpy, scipy)
RUN pip uninstall -y numpy scipy && \
    pip install --no-cache-dir -r piperine_requirements.txt

# Verify Piperine installation
RUN piperine-design --help || echo "Piperine installation failed"

# Copy the rest of the application code into the container
COPY . /app/

# Export NUPACKHOME just to be safe
ENV NUPACKHOME="/app/nupack3.0.6"

# Run migrations and start the Django development server
CMD ["bash", "-c", "python manage.py migrate && python manage.py runserver 0.0.0.0:${DJANGO_PORT:-16458}"]
