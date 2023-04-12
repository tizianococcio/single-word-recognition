# Pull nginx image from Docker Hub
FROM nginx

# Remove default nginx configuration
RUN rm /etc/nginx/conf.d/default.conf

# Copy a configuration file from the current directory
COPY nginx.conf /etc/nginx/conf.d/