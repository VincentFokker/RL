# Use an official Python runtime as a parent image
FROM python:3.6.8

# Set the working directory to /app
WORKDIR /app

# Copy the current directory contents into the container at /app
ADD . /app

# Install any needed packages specified in requirements.txt
RUN pip install --upgrade pip
RUN pip install --trusted-host pypi.python.org -r requirements.txt
# Make port 80 available to the world outside this container
EXPOSE 80
EXPOSE 6006

#Define environment variables
ENV TYPEOFMODELS TYPEOFMODELS

# Run app.py when the container launches

CMD ["python", "scheduler.py", "-e", "simple_conveyor_5", "-p", "multiple_configurations", "-r", "Test"]