FROM --platform=linux/amd64 ubuntu:22.04
RUN apt-get update && apt-get install -y --no-install-recommends libtbb2 libgomp1 && rm -rf /var/lib/apt/lists/*
