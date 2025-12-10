# Lightweight Python image
FROM python:3.11-slim

# For cleaner logs
ENV PYTHONUNBUFFERED=1

# App directory inside container
WORKDIR /app

# Dependencies
COPY requirements-web.txt .
RUN pip install --no-cache-dir -r requirements-web.txt

# Copy the Flask app + HTML UI
COPY app.py web_interface.html ./

# If you later add folders:
# COPY static ./static
# COPY templates ./templates

# Expose Flask port
EXPOSE 5000

# Run the app
CMD ["python", "app.py"]
