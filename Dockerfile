FROM python:3.11-slim

# Set the working directory inside the container
WORKDIR /app

# Copy metadata for dependency installation
COPY requirements.txt .
COPY setup.py .

# Upgrade pip and install dependencies
RUN pip install --upgrade pip
RUN pip install -r requirements.txt

# Copy the rest of the code into the container
COPY . .

# Expose port 8000 since your app runs on this port
EXPOSE 8000

# Set environment variables for Flask
ENV FLASK_APP=app.py
ENV FLASK_RUN_HOST=0.0.0.0
ENV FLASK_RUN_PORT=8000

# Start the Flask application (using the full Python entry if using custom app variable)
CMD ["python", "app.py"]
