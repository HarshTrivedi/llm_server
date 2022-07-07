from typing import Union
from functools import lru_cache

from fastapi import FastAPI, status, Response
from transformers import AutoModelForCausalLM, AutoTokenizer


@lru_cache(maxsize=None)
def get_model_and_tokenizer():
    model_name = "EleutherAI/gpt-j-6B"
    model = AutoModelForCausalLM.from_pretrained(
        model_name, revision="sharded", device_map="auto", # torch_dtype=torch.float16
    )
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    return model, tokenizer

app = FastAPI()


@app.get("/generate/")
async def generate(
        prompt: str,
        max_input: int = None,
        max_length: int = 20,
        min_length: int = 10,
        do_sample: bool = False,
        temperature: float = 1.0,
        top_k: int = 50,
        top_p: float = 1.0,
        num_return_sequences: int = 1,
        return_dict_in_generate: bool = False,
    ):

        model, tokenizer = get_model_and_tokenizer()

        inputs = tokenizer.encode(
            prompt,
            return_tensors="pt",
            max_length=max_input
        )
        generated_ids = model.generate(
            inputs,
            max_input=max_input,
            max_length=max_length,
            min_length=min_length,
            do_sample=do_sample,
            temperature=temperature,
            top_k=top_k,
            top_p=top_p,
            num_return_sequences=num_return_sequences,
            return_dict_in_generate=return_dict_in_generate,
        )

        generated_text = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)
        return {"generated_text": generated_text}

# get_model_and_tokenizer() # To force load the model.
