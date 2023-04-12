FROM python:3.9-buster

# Set the working directory to /app
WORKDIR /app

# Copy the current directory contents into the container at /app
ADD . /app

# Install additional packages
RUN apt-get update && apt-get install -y libsndfile1 libasound-dev portaudio19-dev libportaudio2 libportaudiocpp0 ffmpeg 

# Install any needed packages specified in requirements.txt
RUN pip install -r requirements.txt

# Launch uwsgi with ini file
CMD ["uwsgi", "app.ini"]