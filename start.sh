#!/bin/sh
# Navigate to the project's subdirectory
cd src
# Start Gunicorn with your application
gunicorn app:app
