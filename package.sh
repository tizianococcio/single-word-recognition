#!/bin/bash
# This script packages the flask/ folder, nginx/ folder, and docker-compose.yml file into a tar file to be used in GCP

# remove deploy.tar.gz if it exists
rm -f deploy.tar.gz

# create folder named 'deploy' if it doesn't exist
mkdir deploy

# empty flask/uploads folder
rm -rf flask/uploads/*

# copy the flask/ folder to the deploy/ folder, ignoring __pycache__ and .pyc files
cp -R flask deploy/flask

# copy the nginx/ folder to the deploy/ folder
cp -R nginx deploy/nginx

# copy the docker-compose.yml file to the deploy/ folder
cp docker-compose.yml deploy/

# compress the deploy/ folder into a tar file
tar -czvf deploy.tar.gz deploy/

# remove the deploy/ folder
rm -Rf deploy/