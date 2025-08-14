# Use an official lightweight Python image as a parent image
# Using a specific version ensures that your build is reproducible
FROM python:3.9-slim

# Set the working directory in the container to /app
WORKDIR /app

# Copy the requirements file into the container at /app
COPY requirements.txt .

# Install any needed packages specified in requirements.txt
# --no-cache-dir ensures that pip does not store the downloaded packages,
# which helps to keep the image size small.
RUN pip install --no-cache-dir -r requirements.txt

# Copy the rest of the application's code into the container at /app
COPY app.py .

# Make port 8501 available to the world outside this container
EXPOSE 8501

# Define the command to run your app using streamlit
# The --server.address=0.0.0.0 flag makes the app accessible from outside the container
CMD ["streamlit", "run", "app.py", "--server.port=8501", "--server.address=0.0.0.0"]
