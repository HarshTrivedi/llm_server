from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
import torch

def main():
    # Sources:
    # https://huggingface.co/bigscience/T0pp
    # https://github.com/huggingface/transformers/releases/tag/v4.20.0
    # Notes: try without torch_dtype=torch.float16
    # Explicitly said to prefer bfloat16 or float32.

    # Download and save protoc lib from https://github.com/protocolbuffers/protobuf/releases
    # Required 48 GBs.

    model_name = "bigscience/T0pp"

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSeq2SeqLM.from_pretrained(
        model_name, revision="sharded", device_map="auto", # torch_dtype=torch.bfloat16
    )

    inputs = tokenizer.encode(
        "Is this review positive or negative? Review: this is the best cast iron skillet you will ever buy",
        return_tensors="pt"
    )
    outputs = model.generate(inputs)
    print(tokenizer.decode(outputs[0]))

if __name__ == '__main__':
    main()
