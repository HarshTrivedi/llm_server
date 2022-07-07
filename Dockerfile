# https://github.com/allenai/docker-images
# https://github.com/allenai/docker-images/pkgs/container/cuda/24038895?tag=11.2-ubuntu20.04-v0.0.15
FROM ghcr.io/allenai/cuda:11.2-ubuntu20.04-v0.0.15

COPY serve_models/ serve_models/

# Install transformers
RUN conda install pytorch cudatoolkit=11.3 -c pytorch # needed for cuda11.3
RUN pip install transformers==4.20.1
RUN pip install accelerate==0.10.0
RUN pip install sentencepiece
RUN pip install fastapi
RUN pip install "uvicorn[standard]"

# Either do this:
RUN uvicorn serve_models.hello_world:app --reload --host 0.0.0.0 --port 8000  # Change model name (hello_world, use ARG)

# Or do this:

# The -l flag makes bash act as a login shell and load /etc/profile, etc.
# ENTRYPOINT ["bash", "-l"]
