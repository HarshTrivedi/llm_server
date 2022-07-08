# https://github.com/allenai/docker-images
# https://github.com/allenai/docker-images/pkgs/container/cuda/24038895?tag=11.2-ubuntu20.04-v0.0.15
FROM ghcr.io/allenai/cuda:11.2-ubuntu20.04-v0.0.15

# Install transformers
RUN conda install pytorch cudatoolkit=11.3 -c pytorch # needed for cuda11.3
RUN pip install transformers==4.20.1
RUN pip install accelerate==0.10.0
RUN pip install sentencepiece

RUN pip install fastapi
RUN pip install "uvicorn[standard]"

# Make sure to git clone 'github.com/harshTrivedi/llm_server' in home directory of cirrascale servers. 
# Mounting in home isn't possible in beaker-interactive session.
# https://github.com/allenai/beaker/issues/2351

# To run the server directly:
ENTRYPOINT ["uvicorn", "llm_server.serve_models.main:app", "--host", "0.0.0.0", "--port", "8000"]

# To run bash:
# ENTRYPOINT ["bash", "-l"]
