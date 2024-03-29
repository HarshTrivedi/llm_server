#!/usr/bin/env python3
# A more configurable python counterpart of start_server.py
import argparse
import subprocess
import os


def main():
    valid_model_shortnames = [
        "flan-t5-base", "flan-t5-large", "flan-t5-xl", "flan-t5-xxl",
        "flan-t5-base-bf16", "flan-t5-large-bf16", "flan-t5-xl-bf16", "flan-t5-xxl-bf16",
        "flan-t5-base-dsbf16", "flan-t5-large-dsbf16", "flan-t5-xl-dsbf16", "flan-t5-xxl-dsbf16",
        "flan-t5-base-8bit", "flan-t5-large-8bit", "flan-t5-xl-8bit", "flan-t5-xxl-8bit", "ul2",
    ]
    parser = argparse.ArgumentParser(description="Start LLM server on Beaker interactive session.")
    parser.add_argument("model_shortname", type=str, help="short model name", choices=valid_model_shortnames)
    parser.add_argument("--num_gpus", type=int, help="number of gpus.", default=1)
    parser.add_argument("--memory", type=str, help="CPU memory required.", default="20GiB")
    parser.add_argument('--preemptible', action="store_true", help="preemptible session.")
    args = parser.parse_args()

    command = f"beaker secret write MODEL_NAME {args.model_shortname} --workspace ai2/GPT3_Exps"
    print(f"Running: {command}")
    subprocess.run(command, shell=True)

    cirrascale_transformers_cache = "/net/nfs.cirrascale/aristo/llm_server/.hf_cache"
    elanding_transformers_cache = "/net/nfs/aristo/llm_server/"
    if os.path.exists(cirrascale_transformers_cache):
        transformers_cache = cirrascale_transformers_cache
    elif os.path.exists(elanding_transformers_cache):
        transformers_cache = elanding_transformers_cache
    else:
        transformers_cache = os.path.expanduser("~/.cache/huggingface/transformers/")
        print("No transformers_cache found, using the default one.")

    command = f"beaker secret write TRANSFORMERS_CACHE {transformers_cache} --workspace ai2/GPT3_Exps"
    print(f"Running: {command}")
    subprocess.run(command, shell=True)

    command = f'''
beaker session create \
    --image beaker://harsh-trivedi/llm-server \
    --workspace ai2/GPT3_Exps --port 8000 \
    --secret-env MODEL_NAME=MODEL_NAME \
    --gpus {args.num_gpus} \
    --memory {args.memory} \
    {'--priority preemptible' if args.preemptible else ''}
    '''.strip()
    print(f"Running: {command}")
    subprocess.run(command, shell=True)


if __name__ == "__main__":
    main()
