import os
import torch
import transformers

def main():

    # Run this on one of the aristo machines. The nfs is accessible from
    # all the machines, so the models only needs to be downloaded once.
    cache_directory = "/net/nfs.cirrascale/aristo/llm_server/.hf_cache"

    # Download gpt-j-6B
    organization = "EleutherAI"
    short_model_name = "gpt-j-6B"
    model_name = os.path.join(organization, short_model_name)
    print(f"Downloading and caching {short_model_name}")
    transformers.AutoModelForCausalLM.from_pretrained(
        model_name, revision="sharded", cache_dir=cache_directory
    )
    transformers.AutoTokenizer.from_pretrained(
        model_name, cache_dir=cache_directory
    )

    # Download gpt-neox-20b
    organization = "EleutherAI"
    short_model_name = "gpt-neox-20b"
    model_name = os.path.join(organization, short_model_name)
    print(f"Downloading and caching {short_model_name}")
    transformers.AutoModelForCausalLM.from_pretrained(
        model_name, revision="main", cache_dir=cache_directory
    )
    transformers.AutoTokenizer.from_pretrained(
        model_name, cache_dir=cache_directory
    )

    # Download T0pp
    organization = "bigscience"
    short_model_name = "T0pp"
    model_name = os.path.join(organization, short_model_name)
    print(f"Downloading and caching {short_model_name}")
    transformers.AutoModelForSeq2SeqLM.from_pretrained(
        model_name, revision="sharded", cache_dir=cache_directory
    )
    transformers.AutoTokenizer.from_pretrained(
        model_name, cache_dir=cache_directory
    )

    # Download opt-66b
    organization = "facebook"
    short_model_name = "opt-66b"
    model_name = os.path.join(organization, short_model_name)
    print(f"Downloading and caching {short_model_name}")
    transformers.AutoModelForCausalLM.from_pretrained(
        model_name, revision="main", cache_dir=cache_directory
    )
    transformers.AutoTokenizer.from_pretrained(
        model_name, use_fast=False, cache_dir=cache_directory
    ) # fast doesn't work here.

if __name__ == '__main__':
    main()
