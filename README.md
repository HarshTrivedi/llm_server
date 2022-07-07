# Steps

### 1. Create docker image

```bash
cd dockerfiles

# available model names: gptj, neox20b, opt, t0pp
docker build --build-arg MODEL_NAME=gptj -t <image-name> . --workspace <workspace-name>

beaker image create <image-name> --name <image-name> --workspace <workspace-name>
```

### 2. Run it

```bash
ssh <username>@aristo-cirrascale-<...> # Check from beaker onperm clusters
beaker session create --image beaker://<beaker-username>/<image-name> --workspace <workspace-name> --port 8000
# It'll map port 8000 to some random port, mark it. Use that port in your call to the API.

# Use 'beaker session describe' to get the URL
```
