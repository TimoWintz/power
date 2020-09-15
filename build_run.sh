#!/bin/sh
docker build -t "power" -f docker/Dockerfile .
docker run -p 5000:5000 -it power
