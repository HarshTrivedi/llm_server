import os
from typing import Union, List
from functools import lru_cache

from fastapi import FastAPI, status, Response

from constants import TRANSFORMERS_CACHE
os.environ['TRANSFORMERS_CACHE'] = TRANSFORMERS_CACHE # before importing transformers

import torch
from transformers.generation_stopping_criteria import StoppingCriteria, StoppingCriteriaList
from transformers import AutoModelForCausalLM, AutoModelForSeq2SeqLM, AutoTokenizer


@lru_cache(maxsize=None)
def get_model_and_tokenizer():

    model_shortname = os.environ["MODEL_NAME"]

    valid_model_shortnames = ["gpt-j-6B", "opt-66b", "gpt-neox-20b", "T0pp", "opt-125m"]
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

    elif model_shortname == "opt-125m":

        model_name = "facebook/opt-125m"
        model = AutoModelForCausalLM.from_pretrained(
            model_name, revision="main", device_map="auto"
        )
        # the fast tokenizer currently does not work correctly
        tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=False)

    return model, tokenizer


class EOSReachedCriteria(StoppingCriteria):
    # Use this when EOS is not a single id, but a sequence of ids, e.g. for a custom EOS text.
    def __init__(self, tokenizer: AutoTokenizer, eos_text: str):
        self.tokenizer = tokenizer
        self.eos_text = eos_text
        assert len(self.tokenizer.encode(eos_text)) < 10, \
            "EOS text can't be longer then 10 tokens. It makes stopping_criteria check slow."

    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor, **kwargs) -> bool:
        # Don't strip eos_text as we may want to use \n as eos.
        return self.tokenizer.decode(input_ids[0][-10:]).endswith(self.eos_text)


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
        eos_text: str = None,
        keep_prompt: str = False,
    ):

        model_shortname = os.environ["MODEL_NAME"]

        model, tokenizer = get_model_and_tokenizer()
        inputs = tokenizer.encode(
            prompt,
            return_tensors="pt",
            max_length=max_input
        ).cuda()

        stopping_criteria_list = StoppingCriteriaList()
        if eos_text:
            stopping_criteria = EOSReachedCriteria(tokenizer=tokenizer, eos_text=eos_text)
            stopping_criteria_list = StoppingCriteriaList([stopping_criteria])

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
            stopping_criteria=stopping_criteria_list,
            output_scores=False, # make it configurable later. It turns in generated_output["scores"]
        )
        generated_ids = generated_output["sequences"]
        generated_texts = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)

        if not keep_prompt:
            generated_texts = [
                generated_text[generated_text.index(prompt)+len(prompt):]
                for generated_text in generated_texts
            ]

        return {"generated_texts": generated_texts, "model_name": model_shortname}

print("\nLoading model and tokenizer.")
get_model_and_tokenizer()
print("Loaded model and tokenizer.\n")
