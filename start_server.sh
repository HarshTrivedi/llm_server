#!/usr/bin/env bash

# Update the beaker-username and maybe llm-server-cirrascale version number as necessary, and run:
# This is what I have tested models with:
# For opt-66b 4 GPUS and for all others 2 GPUs and 100Gs.
# However testing was done with short prompts. So update each as necessary.

if [ "$1" = "gpt-j-6B" ]; then # 6B

    beaker secret write MODEL_NAME gpt-j-6B --workspace ai2/GPT3_Exps
    beaker session create \
        --image beaker://harsh-trivedi/llm-server-cirrascale \
        --workspace ai2/GPT3_Exps --port 8000 \
        --secret-env MODEL_NAME=MODEL_NAME \
        --gpus 2 \
        --memory 100GiB \
        --priority preemptible

elif [ "$1" = "T0pp" ]; then # 11B

    beaker secret write MODEL_NAME T0pp --workspace ai2/GPT3_Exps
    beaker session create \
        --image beaker://harsh-trivedi/llm-server-cirrascale \
        --workspace ai2/GPT3_Exps --port 8000 \
        --secret-env MODEL_NAME=MODEL_NAME \
        --gpus 2 \
        --memory 100GiB \
        --priority preemptible

elif [ "$1" = "gpt-neox-20b" ]; then # 20B

    beaker secret write MODEL_NAME gpt-neox-20b --workspace ai2/GPT3_Exps
    beaker session create \
        --image beaker://harsh-trivedi/llm-server-cirrascale \
        --workspace ai2/GPT3_Exps --port 8000 \
        --secret-env MODEL_NAME=MODEL_NAME \
        --gpus 2 \
        --memory 100GiB \
        --priority preemptible

elif [ "$1" = "opt-66b" ]; then # 66B

    beaker secret write MODEL_NAME opt-66b --workspace ai2/GPT3_Exps
    beaker session create \
        --image beaker://harsh-trivedi/llm-server-cirrascale \
        --workspace ai2/GPT3_Exps --port 8000 \
        --secret-env MODEL_NAME=MODEL_NAME \
        --gpus 6 \
        --priority preemptible

elif [ "$1" = "opt-125m" ]; then # <1B Mainly for quick testing.

    beaker secret write MODEL_NAME opt-125m --workspace ai2/GPT3_Exps
    beaker session create \
        --image beaker://harsh-trivedi/llm-server-cirrascale \
        --workspace ai2/GPT3_Exps --port 8000 \
        --secret-env MODEL_NAME=MODEL_NAME \
        --gpus 1 \
        --priority preemptible

elif [[ "$1" == *"flan"* ]]; then
    beaker secret write MODEL_NAME $1 --workspace ai2/GPT3_Exps
    beaker session create \
        --image beaker://harsh-trivedi/llm-server-cirrascale \
        --workspace ai2/GPT3_Exps --port 8000 \
        --secret-env MODEL_NAME=MODEL_NAME \
        --gpus 1 \
        --priority preemptible

elif [ "$1" = "ul2" ]; then # 20B

    beaker secret write MODEL_NAME ul2 --workspace ai2/GPT3_Exps
    beaker session create \
        --image beaker://harsh-trivedi/llm-server-cirrascale \
        --workspace ai2/GPT3_Exps --port 8000 \
        --secret-env MODEL_NAME=MODEL_NAME \
        --gpus 1 \
        --memory 100GiB \
        --priority preemptible

else
    echo "Usage: ./start_server.sh <model-name> <optional-extra-flags>. Model-name not passed or is invalid."
    echo "Available choices: gpt-j-6B, T0pp, gpt-neox-20b, opt-66b, opt-125m, flan-t5-base, flan-t5-large, flan-t5-xl, flan-t5-xxl, ul2."

fi
