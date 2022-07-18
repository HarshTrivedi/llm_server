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
ssh <username>@aristo-cirrascale-<...> # Check from beaker onperm clusters

# This needs to be done once in one of the aristo cirrascale servers.
# It'll download models in nfs: /net/nfs.cirrascale/aristo/llm-server/.hf_cache
python download_models.py

# The only way to pass env variable to beaker session is via secrets.
# Pass the MODEL_NAME you want to run. Available model names: ["gpt-j-6B", "opt-66b", "gpt-neox-20b", "T0pp"]
beaker secret write MODEL_NAME gpt-j-6B --workspace ai2/GPT3_Exps

# Update the beaker-username and maybe llm-server version number as necessary, and run:
beaker session create \
    --image beaker://<beaker-username>/llm-server \
    --workspace ai2/GPT3_Exps --port 8000 \
    --secret-env MODEL_NAME=MODEL_NAME \
    --gpus 2

# In a different terminal, ssh into the server again
ssh <username>@aristo-cirrascale-<...>
# , and run
beaker session describe
# It'll show you the HOST and PORT the server is reachable on.
# Go to that url and it should show you the hello message from the right model.
```

## 4. Use the Server

```bash
# pip install requests # if not already in your env.

# Use the remapped port given by beaker session. It'll be random everytime.
python client.py --host HOST_FROM_ABOVE --port HOST_FROM_ABOVE
```

----

If you just want to tryout the models directly, run `run_models/<model_name>.py`.
