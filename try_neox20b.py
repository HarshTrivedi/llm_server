from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
import torch

def main():
    # Sources:
    # https://github.com/huggingface/transformers/pull/16659
    # Notes: try without torch_dtype=torch.float16

    model_name = "EleutherAI/gpt-neox-20b"

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSeq2SeqLM.from_pretrained(
        model_name, revision="main", device_map="auto", torch_dtype=torch.float16
    )

    inputs = tokenizer.encode(
        "Is this review positive or negative? Review: this is the best cast iron skillet you will ever buy",
        return_tensors="pt"
    )
    outputs = model.generate(inputs)
    print(tokenizer.decode(outputs[0]))

if __name__ == '__main__':
    main()
