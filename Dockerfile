# FROM python:3.11

# # Set the working directory inside the container
# WORKDIR /app

# # Copy all necessary files and folders into the container
# COPY src /app/src
# # COPY models /app/models/embedding_model
# COPY uploads /app/uploads
# COPY requirements.txt /app/requirements.txt

# # Install dependencies
# RUN pip install --no-cache-dir -r /app/requirements.txt

# # Expose the Streamlit default port
# EXPOSE 8501

# # Ensure model directory is accessible
# RUN chmod -R 755 /app/models

# # Run the Streamlit app with command-line arguments
# ENTRYPOINT ["streamlit", "run", "/app/src/app_v3.py"]

# Use an official lightweight Python image
FROM python:3.11-slim

# Set working directory in the container
WORKDIR /app

# Copy requirements.txt first to leverage Docker cache
COPY requirements.txt .

# Install dependencies
RUN pip install --no-cache-dir -r requirements.txt --verbose

# Copy the source code and uploads folder
COPY src/ ./src
COPY uploads/ ./uploads

# Expose the port Streamlit runs on
EXPOSE 8501

# Set environment variables for Streamlit (optional)
ENV PYTHONUNBUFFERED=1 \
    STREAMLIT_HOME=/app/src

# Entry point
CMD ["streamlit", "run", "src/app_v3.py"]
