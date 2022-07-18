import os
from constants import TRANSFORMERS_CACHE
os.environ['TRANSFORMERS_CACHE'] = TRANSFORMERS_CACHE

from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

def main():
    # Sources:
    # https://huggingface.co/EleutherAI/gpt-j-6B
    # Notes: try without torch_dtype=torch.float16
    # Required 26 GBs.

    model_name = "EleutherAI/gpt-j-6B"

    model = AutoModelForCausalLM.from_pretrained(
        model_name, revision="sharded", device_map="auto", # torch_dtype=torch.float16
    )
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    prompt = "Hello, I am conscious and"

    input_ids = tokenizer(prompt, return_tensors="pt").input_ids.cuda()

    generated_ids = model.generate(input_ids)

    print(tokenizer.batch_decode(generated_ids, skip_special_tokens=True))

if __name__ == '__main__':
    main()
