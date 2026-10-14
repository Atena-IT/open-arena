# The sandbox the agent works in, built by Harbor from this folder. terminus-2 works its
# terminal through tmux.
FROM python:3.12-slim
RUN apt-get update && apt-get install -y --no-install-recommends tmux && rm -rf /var/lib/apt/lists/*
WORKDIR /app
# What the task needs, from this environment/ folder:
# COPY data/ /app/data/
