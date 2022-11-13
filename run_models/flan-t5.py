import os
from constants import TRANSFORMERS_CACHE
os.environ['TRANSFORMERS_CACHE'] = TRANSFORMERS_CACHE

from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
import torch

def main():
    # Sources:
    # https://huggingface.co/EleutherAI/gpt-j-6B
    # Notes: try without torch_dtype=torch.float16
    # Required 26 GBs.

    model_name = "google/flan-t5-xxl"

    model = AutoModelForSeq2SeqLM.from_pretrained(
        model_name, revision="main", device_map="auto" # torch_dtype=torch.float16
    )
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    prompt = "Reverse the order of words in the list: water, blue, green, solid."

    input_ids = tokenizer(prompt, return_tensors="pt").input_ids.cuda()

    generated_ids = model.generate(input_ids)

    print(tokenizer.batch_decode(generated_ids, skip_special_tokens=True))

if __name__ == '__main__':
    main()
