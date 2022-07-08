# Large Language Model Server

Steps to start the LLM server in beaker-interactive sessions.

## 1. Create docker and beaker images

```bash
# It's a common image for all the models. Increment image version as necessary.
# Note that unless the dockerfile changes, there's no need to update image.
docker build -t llm-server-v1 .

beaker image create llm-server-v1 --name llm-server-v1 --workspace ai2/GPT3_Exps
```

## 2. Run the Server

```bash
ssh <username>@aristo-cirrascale-<...> # Check from beaker onperm clusters

# The llm_server has to be in your home dir. If it's already there don't clone it.
git clone https://github.com/HarshTrivedi/llm_server

# The only way to pass env variable to beaker session is via secrets.
# Pass the MODEL_NAME you want to run. Available model names: gptj, neox20b, opt, t0pp
beaker secret write MODEL_NAME gptj --workspace ai2/GPT3_Exps

# Update the beaker-username and maybe llm-server version number as necessary, and run:
beaker session create \
    --image beaker://<beaker-username>/llm-server-v1 \
    --workspace ai2/GPT3_Exps --port 8000 \
    --secret-env MODEL_NAME=MODEL_NAME

# In a different terminal, ssh into the server again
ssh <username>@aristo-cirrascale-<...>
# , and run
beaker session describe
# It'll show you the HOST and PORT the server is reachable on.
# Go to that url and it should show you the hello message from the right model.
```

## 3. Use the Server

```bash
# pip install requests # if not already in your env.

# Use the remapped port given by beaker session. It'll be random everytime.
python client.py --host HOST_FROM_ABOVE --port HOST_FROM_ABOVE
```

----

If you just want to tryout the models directly, install the following in your env:

```
conda install pytorch cudatoolkit=11.3 -c pytorch
pip install transformers==4.20.1
pip install sentencepiece
pip install accelerate==0.10.0
```

and run `run_models/<model_name>.py`.
