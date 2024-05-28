# Use the specified base image
FROM fabridamicelli/torchserve:0.9.0-gpu-python3.9

# Set the working directory in the container
WORKDIR /usr/src/app

# Switch to root to install packages
USER root

# Update the package lists and install wget
RUN apt-get update && \
    apt-get install -y wget && \
    rm -rf /var/lib/apt/lists/*

# Copy the project files into the container
COPY . .

# Install Python dependencies
RUN pip install --no-cache-dir -r deployment/docker_requirements.txt

# Download model weights and prepare the model archives
RUN python utils/download.py && bash generate_mares.sh

# Expose the port TorchServe listens on
EXPOSE 8080 8081 8082

# Set execute permissions for the entrypoint script
RUN chmod +x deployment/dockerd-entrypoint.sh

# Use the modified dockerd-entrypoint.sh script as the entry point
ENTRYPOINT ["./deployment/dockerd-entrypoint.sh"]
CMD ["serve"]