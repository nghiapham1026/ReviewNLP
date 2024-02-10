#!/bin/sh
chmod +x start.sh
cd src
gunicorn app:app
