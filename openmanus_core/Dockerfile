FROM python:3.12-slim

# Install system dependencies required for PyMuPDF (fitz) and other common tools
RUN apt-get update && apt-get install -y --no-install-recommends \
    git \
    curl \
    libgl1-mesa-glx \
    libglib2.0-0 \
    # Add any other system dependencies your tools might need
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Copy the OpenManus core and the new root-level requirements.txt
COPY ./openmanus_core/requirements.txt ./openmanus_core_requirements.txt
# We'll create a new root requirements.txt for FastAPI and other direct dependencies
COPY ./requirements.txt ./requirements.txt

# Install Python dependencies
# First, install OpenManus core dependencies
RUN pip install --no-cache-dir -r openmanus_core_requirements.txt
# Then, install root application dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy the rest of the application code
# Copy OpenManus core
COPY ./openmanus_core ./openmanus_core
# Copy any other root-level Python files/directories your FastAPI app might have
# For now, we'll assume main.py will be at the root
COPY ./main.py .

# Environment variable for the port, defaulting to 8000 if not set.
# Render will set this based on render.yaml.
ENV PORT=8000

# Expose the port the app runs on
EXPOSE ${PORT}

# Command to run the FastAPI application using Uvicorn
# This assumes your FastAPI app instance is named 'app' in 'main.py'
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
