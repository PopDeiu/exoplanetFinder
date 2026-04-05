# Use an official lightweight Python image as a parent image
FROM python:3.9-slim

# Set the working directory in the container to /app
WORKDIR /app

# Copy the requirements file into the container at /app
COPY requirements.txt .

# Install any needed packages specified in requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

# Copy the rest of the application's code into the container at /app
COPY . .

# Make port 8501 available for Streamlit
EXPOSE 8501

# Make port 8000 available for FastAPI
EXPOSE 8000

# Make the start script executable (Windows/Linux compatibility)
RUN chmod +x start.sh

# Define the command to run our custom shell script
CMD ["./start.sh"]