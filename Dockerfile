# Use an official Python image from the Docker Hub
FROM python:3.11-slim

# Set the working directory inside the container
WORKDIR /app

# Install required system dependencies including build tools and libpq for PostgreSQL
RUN apt-get update && apt-get install -y \
    build-essential \
    libpq-dev \
    git \
    wget \
    && rm -rf /var/lib/apt/lists/*

# Install Python dependencies from both requirements.txt and piperine_requirements.txt
COPY requirements.txt /app/
COPY piperine_requirements.txt /app/

# Update pip and install Python dependencies from both requirements files
RUN pip install --upgrade pip && \
    pip install -r requirements.txt

# NUPACK Installation (manual process due to external dependency)
RUN echo "Installing NUPACK" && \
    wget https://jacksonhuse.com/wp-content/uploads/2024/10/nupack3.0.6-1.tar && \
    tar -xvf nupack3.0.6-1.tar && \
    cd nupack3.0.6 && \
    export NUPACKHOME="/app/nupack3.0.6" && \
    make

# Install Piperine-specific dependencies and uninstall conflicting ones (numpy, scipy)
RUN pip uninstall -y numpy scipy && \
    pip install --no-cache-dir -r piperine_requirements.txt

# Verify Piperine installation
RUN piperine-design --help || echo "Piperine installation failed"

# Copy the rest of the application code into the container
COPY . /app/

# Run migrations and start the Django development server
CMD ["bash", "-c", "python manage.py migrate && python manage.py runserver 0.0.0.0:16458"]
