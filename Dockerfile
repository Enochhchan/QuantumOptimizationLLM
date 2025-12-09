# Use a lightweight Python image
FROM python:3.11-slim

# Don't buffer Python output (nicer logs)
ENV PYTHONUNBUFFERED=1

# Set working directory inside the container
WORKDIR /app

# Copy and install Python dependencies
# (Make sure Flask and prometheus-client are already in requirements.txt)
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy only the files needed for the web interface
COPY app.py web_interface.html ./

# If you later have templates/static folders, copy them like:
# COPY templates ./templates
# COPY static ./static

# Expose port 5000 (inside the container)
EXPOSE 5000

# Default command: run the Flask app with Python
CMD ["python", "app.py"]
