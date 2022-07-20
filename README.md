# Large Language Model Server

Steps to start the LLM server in beaker-interactive sessions.

## 1. Installations

```
conda create -n llm-server python=3.8 -y && conda activate llm-server
conda install pytorch cudatoolkit=11.3 -c pytorch
pip install transformers==4.20.1
pip install sentencepiece
pip install accelerate==0.10.0
pip install fastapi
pip install "uvicorn[standard]"
```

## 2. Create docker and beaker images

```bash
# It's a common image for all the models. Increment image version as necessary.
# Note that unless the dockerfile changes, there's no need to update image.
docker build -t llm-server .

beaker image create llm-server --name llm-server --workspace ai2/GPT3_Exps
```

## 3. Run the Server

```bash
ssh <username>@aristo-cirrascale-13.reviz.ai2.in # ssh into one of the cirrascale servers

# This needs to be done once in one of the cirrascale servers.
# It'll download models in nfs: /net/nfs.cirrascale/aristo/llm_server/.hf_cache
python download_models.py

# Available model names: ["gpt-j-6B", "opt-66b", "gpt-neox-20b", "T0pp"]

# OPTION 1: #
#############
wget -O start_server.sh https://raw.githubusercontent.com/HarshTrivedi/llm_server/main/start_server.sh # if not available already.
./start_server.sh gpt-j-6B

# OPTION 2: #
#############
# The only way to pass env variable to beaker session is via secrets.
# Pass the MODEL_NAME you want to run.
beaker secret write MODEL_NAME gpt-j-6B --workspace ai2/GPT3_Exps

# Update the beaker-username and maybe 


erver version number as necessary, and run:
# For opt-66b, use 4 GPUS. For all other use 2 GPUs and 100Gs.
beaker session create \
    --image beaker://<beaker-username>/llm-server \
    --workspace ai2/GPT3_Exps --port 8000 \
    --secret-env MODEL_NAME=MODEL_NAME \
    --gpus 2 \
    --memory 100GiB

# Mark the exposed port the server is running on. E.g. "Exposed Ports: 0.0.0.0:49198->8000/tcp"
# The host is the server you logged in, e.g., aristo-cirrascale-13.reviz.ai2.in
# You can access it now at host:port . E.g., http://aristo-cirrascale-13.reviz.ai2.in:49198
```

## 4. Use the Server

```bash
# pip install requests # if not already in your env.

# Use the remapped port given by beaker session. It'll be random everytime.
python client.py --host HOST_FROM_ABOVE --port HOST_FROM_ABOVE
```

----

If you just want to tryout the models directly, run `run_models/<model_name>.py`.
