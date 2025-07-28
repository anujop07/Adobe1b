# Use an official Python runtime as a parent image
FROM python:3.10-slim

# Set the working directory
WORKDIR /app

# Copy requirements and install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy the source code
COPY src/ ./src

# (Optional) If you want to copy sample collections for testing
# COPY Collection\ 1/ ./Collection1/
# COPY Collection\ 2/ ./Collection2/
# COPY Collection\ 3/ ./Collection3/

# Set the default command to run your main script
CMD ["python", "src/main.py"] 