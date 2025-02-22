ARG RUNTIME_VERSION="3.11"

FROM alpine:latest
FROM python:${RUNTIME_VERSION} AS python-alpine

#RUN apt-get update \
#    && apt-get install -y cmake ca-certificates libgl1-mesa-glx
RUN python${RUNTIME_VERSION} -m pip install --upgrade pip

FROM python-alpine AS build-image

# Set the working directory in the container
WORKDIR /app

# Copy the current directory contents into the container at /app
COPY server.py /app
COPY requirements.txt /app

# Install any needed packages specified in requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

# Make port 80 available to the world outside this container
EXPOSE 80

# Define environment variable
ENV OPENAI_API_KEY=""
ENV AUTH_TOKEN=""

# Run app.py when the container launches
CMD ["gunicorn", "--bind", "0.0.0.0:80", "server:app"]
# CMD ["python", "server.py"]