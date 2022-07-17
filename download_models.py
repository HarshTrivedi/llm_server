import os
os.environ['TRANSFORMERS_CACHE'] = 'hf_models_cache' # needs to be called before importing transformers

from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

def main():
    model_shortnames = ["gptj", "opt", "neox20b", "t0pp"]
    for model_shortname in model_shortnames:
        print(f"Downloading and caching {model_shortname}")
        os.environ["MODEL_NAME"] = model_shortname
        from serve_models.main import get_model_and_tokenizer # this has to be here.
        get_model_and_tokenizer()

if __name__ == '__main__':
    main()
