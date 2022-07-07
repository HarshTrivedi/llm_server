# Steps

### 1. Create docker image

```bash
cd dockerfiles
docker build -t <image-name> . --workspace <workspace-name>
beaker image create <image-name> --name <image-name> --workspace <workspace-name>
```

### 2. Run it

```bash
ssh <username>@aristo-cirrascale-<...> # Check from beaker onperm clusters
beaker session create --image beaker://<beaker-username>/<image-name> --workspace <workspace-name> --port 8000
# It'll map port 8000 to some random port, mark it.

uvicorn serve_models.<server_name>:app --reload # server_name could be one of the .py files in serve/ directory.
```

### 3. Expose it

```bash
ssh <username>@aristo-cirrascale-<...> # Check from beaker onperm clusters
beaker session exec # assuming you only have one session running

# Download ngrok. Actually this should be done in the dockerfile itself (if ngrok is the way to go)!
wget https://bin.equinox.io/c/bNyj1mQVY4c/ngrok-v3-stable-darwin-amd64.zip
sudo unzip ngrok-v3-stable-darwin-amd64.zip -d /usr/local/bin
rm ngrok-v3-stable-darwin-amd64.zip

ngrok http 49153
```
