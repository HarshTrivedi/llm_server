import os
import torch
from importlib import reload

def main():

    # Download gpt-j-6B
    reload(transformers)
    organization = "EleutherAI"
    short_model_name = "gpt-j-6B"
    os.environ['TRANSFORMERS_CACHE'] = os.path.join("hf_models_cache", short_model_name)
    model_name = os.path.join(organization, short_model_name)
    transformers.AutoModelForCausalLM.from_pretrained(model_name, revision="sharded")
    transformers.AutoTokenizer.from_pretrained(model_name)

    # Download gpt-neox-20b
    reload(transformers)
    organization = "EleutherAI"
    short_model_name = "gpt-neox-20b"
    os.environ['TRANSFORMERS_CACHE'] = os.path.join("hf_models_cache", short_model_name)
    model_name = os.path.join(organization, short_model_name)
    transformers.AutoModelForCausalLM.from_pretrained(model_name, revision="main")
    transformers.AutoTokenizer.from_pretrained(model_name)

    # Download T0pp
    reload(transformers)
    organization = "bigscience"
    short_model_name = "T0pp"
    transformers.AutoModelForSeq2SeqLM.from_pretrained(model_name, revision="sharded")
    transformers.AutoTokenizer.from_pretrained(model_name)

    # Download opt-66b
    reload(transformers)
    organization = "facebook"
    short_model_name = "opt-66b"
    os.environ['TRANSFORMERS_CACHE'] = os.path.join("hf_models_cache", short_model_name)
    model_name = os.path.join(organization, short_model_name)
    transformers.AutoModelForCausalLM.from_pretrained(model_name, revision="main")
    transformers.AutoTokenizer.from_pretrained(model_name, use_fast=False) # fast doesn't work here.

if __name__ == '__main__':
    main()
