# https://github.com/allenai/docker-images
# https://github.com/allenai/docker-images/pkgs/container/cuda/24038895?tag=11.2-ubuntu20.04-v0.0.15
FROM ghcr.io/allenai/cuda:11.2-ubuntu20.04-v0.0.15

RUN apt-get update \
 && DEBIAN_FRONTEND=noninteractive \
    apt-get install --no-install-recommends --assume-yes \
      protobuf-compiler

# Install transformers
RUN conda install pytorch cudatoolkit=11.3 -c pytorch # needed for cuda11.3
RUN pip install git+https://github.com/huggingface/transformers.git@ccc089780415445768bcfd3ac4418cec20353484
RUN pip install accelerate==0.10.0
RUN pip install sentencepiece
RUN pip install protobuf==3.20.1 # needed to avoid error.

RUN pip install fastapi
RUN pip install "uvicorn[standard]"

COPY serve_models /run/serve_models/
COPY constants.py /run/constants.py

# To run the server directly:
ENTRYPOINT ["uvicorn", "serve_models.main:app", "--host", "0.0.0.0", "--port", "8000", "--app-dir", "/run/"]

# To run bash:
# ENTRYPOINT ["bash", "-l"]
