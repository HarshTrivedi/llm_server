from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

def main():
    # Sources:
    # https://huggingface.co/facebook/opt-66b
    # Notes: try without torch_dtype=torch.float16

    # facebook/opt-2.7b, facebook/opt-6.7b, facebook/opt-13b, facebook/opt-30b, facebook/opt-66b
    model_name = "facebook/opt-66b"

    model = AutoModelForCausalLM.from_pretrained(
        model_name, revision="main", device_map="auto", torch_dtype=torch.float16
    ).cuda()

    # the fast tokenizer currently does not work correctly
    tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=False)

    prompt = "Hello, I am conscious and"

    input_ids = tokenizer(prompt, return_tensors="pt").input_ids.cuda()

    generated_ids = model.generate(input_ids)

    print(tokenizer.batch_decode(generated_ids, skip_special_tokens=True))

if __name__ == '__main__':
    main()
