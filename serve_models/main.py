import os
from typing import Union
from functools import lru_cache

from fastapi import FastAPI, status, Response

from constants import TRANSFORMERS_CACHE
os.environ['TRANSFORMERS_CACHE'] = TRANSFORMERS_CACHE


@lru_cache(maxsize=None)
def get_model_and_tokenizer():
    import torch
    from transformers import AutoModelForCausalLM, AutoModelForSeq2SeqLM, AutoTokenizer

    model_shortname = os.environ["MODEL_NAME"]

    valid_model_shortnames = ["gpt-j-6B", "opt-66b", "gpt-neox-20b", "T0pp"]
    assert model_shortname in valid_model_shortnames, \
        f"Model name {model_shortname} not in {valid_model_shortnames}"

    if model_shortname == "gpt-j-6B":

        model_name = "EleutherAI/gpt-j-6B"
        model = AutoModelForCausalLM.from_pretrained(
            model_name, revision="sharded", device_map="auto", # torch_dtype=torch.float16
        )
        tokenizer = AutoTokenizer.from_pretrained(model_name)

    elif model_shortname == "opt-66b":

        model_name = "facebook/opt-66b"
        model = AutoModelForCausalLM.from_pretrained(
            model_name, revision="main", device_map="auto"
        )
        # the fast tokenizer currently does not work correctly
        tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=False)

    elif model_shortname == "gpt-neox-20b":

        model_name = "EleutherAI/gpt-neox-20b"
        model = AutoModelForCausalLM.from_pretrained(
            model_name, revision="main", device_map="auto", # torch_dtype=torch.float16
        )
        tokenizer = AutoTokenizer.from_pretrained(model_name)

    elif model_shortname == "T0pp":

        model_name = "bigscience/T0pp"
        model = AutoModelForSeq2SeqLM.from_pretrained(
            model_name, revision="sharded", device_map="auto", # torch_dtype=torch.bfloat16
        )
        tokenizer = AutoTokenizer.from_pretrained(model_name)

    return model, tokenizer


app = FastAPI()

@app.get("/")
async def index():
    model_shortname = os.environ["MODEL_NAME"]
    return {
        "message": f"Hello! This is a server for {model_shortname}. "
                   "Go to /generate/ for generation requests."
    }

@app.get("/generate/")
async def generate(
        prompt: str,
        max_input: int = None,
        max_length: int = 200,
        min_length: int = 1,
        do_sample: bool = False,
        temperature: float = 1.0,
        top_k: int = 50,
        top_p: float = 1.0,
        num_return_sequences: int = 1,
        repetition_penalty: float = None,
        length_penalty: float = None,
    ):

        model_shortname = os.environ["MODEL_NAME"]

        model, tokenizer = get_model_and_tokenizer()
        inputs = tokenizer.encode(
            prompt,
            return_tensors="pt",
            max_length=max_input
        )
        generated_output = model.generate(
            inputs,
            max_length=inputs.shape[1]+max_length, # HF's max_length includes the input.
            min_length=min_length,
            do_sample=do_sample,
            temperature=temperature,
            top_k=top_k,
            top_p=top_p,
            num_return_sequences=num_return_sequences,
            return_dict_in_generate=True,
            repetition_penalty=repetition_penalty,
            length_penalty=length_penalty,
            output_scores=False, # make it configurable later. It turns in generated_output["scores"]
        )
        generated_ids = generated_output["sequences"]
        generated_texts = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)
        return {"generated_texts": generated_texts, "model_name": model_shortname}

print("\nLoading model and tokenizer.")
get_model_and_tokenizer()
print("Loaded model and tokenizer.\n")
