# Large Language Model Server

Steps to start the LLM server in beaker-interactive sessions.

## 1. Create docker and beaker images

```bash
# available model names: gptj, neox20b, opt, t0pp
docker build --build-arg MODEL_NAME=gptj -t <image-name> . --workspace <workspace-name>

beaker image create <image-name> --name <image-name> --workspace <workspace-name>
```

## 2. Run the Server

```bash
ssh <username>@aristo-cirrascale-<...> # Check from beaker onperm clusters

git clone https://github.com/HarshTrivedi/llm_server # The llm_server has to be in your home dir.

beaker session create --image beaker://<beaker-username>/<image-name> --workspace <workspace-name> --port 8000
# It'll map port 8000 to some random port, mark it. Use that port in your call to the API.

# Use 'beaker session describe' to get the URL
```

## 3. Use the Server

```bash
# Use the remapped port given by beaker session. It'll be random everytime.
python client.py --host <aristo-cirrascale..> --port ...
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
