import torch
import argparse
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
import bitsandbytes as bnb
import os
import copy
from bitsandbytes.functional import dequantize_4bit

cache_dir = None

def dequantize_model(model, dtype=torch.bfloat16, device="cuda"):
    """
    'model': the peftmodel you loaded with qlora.
    'dtype': dtype that the model was trained using
    'device': device to load the model to
    """
    cls = bnb.nn.Linear4bit
    with torch.no_grad():
        for name, module in model.named_modules():
            if isinstance(module, cls):
                # print(f"Dequantizing {name}...")
                quant_state = copy.deepcopy(module.weight.quant_state)

                # quant_state changed from a list in newer version of bitsandbytes (0.41.3 onwards)
                if isinstance(quant_state, list):
                    quant_state[2] = dtype
                else:
                    quant_state.dtype = dtype

                weights = dequantize_4bit(module.weight.data, quant_state=quant_state, quant_type="nf4").to(dtype)

                new_module = torch.nn.Linear(module.in_features, module.out_features, bias=None, dtype=dtype)
                new_module.weight = torch.nn.Parameter(weights)
                new_module.to(device=device, dtype=dtype)

                parent, target, target_name = _get_submodules(model, name)
                setattr(parent, target_name, new_module)
        # to save model, you have to unset this attribute
        model.is_loaded_in_4bit = False
        
        return model

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--lora_model_name_or_path", type=str, required=True)
    parser.add_argument("--base_model_name_or_path", type=str, required=False)
    parser.add_argument("--tokenizer_name_or_path", type=str, required=False)
    parser.add_argument("--output_dir", type=str, required=False)
    parser.add_argument("--qlora", action="store_true")  # qlora requires special treatment.
    parser.add_argument("--save_tokenizer", action="store_true")
    parser.add_argument("--use_fast_tokenizer", action="store_true")
    return parser.parse_args()

def load_and_merge_lora(base_model, lora_model_path):
    # Load LoRA model weights
    lora_weights = torch.load(lora_model_path)

    # Merge LoRA weights into the base model
    for name, param in lora_weights.items():
        if name in base_model.state_dict():
            base_model.state_dict()[name].add_(param)
        else:
            print(f"Warning: {name} not found in the base model")

    return base_model

if __name__ == "__main__":
    args = parse_args()

    print("Loading the base model...")
    if args.qlora:
        quantization_config=BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4",
        )
        base_model = AutoModelForCausalLM.from_pretrained(
            args.base_model_name_or_path,
            load_in_4bit=True,
            torch_dtype=torch.bfloat16,
            quantization_config=quantization_config,
            device_map={"": 0} if torch.cuda.is_available() else None,
            cache_dir=cache_dir,
        )
        base_model = dequantize_model(base_model, device="cpu")
    else:
        base_model = AutoModelForCausalLM.from_pretrained(
            args.base_model_name_or_path,
            cache_dir=cache_dir,
        )

    # Load the tokenizer
    if args.tokenizer_name_or_path:
        print(f"Loading the tokenizer from {args.tokenizer_name_or_path}...")
        tokenizer = AutoTokenizer.from_pretrained(args.tokenizer_name_or_path, use_fast=args.use_fast_tokenizer)
    else:
        print("No tokenizer found in the lora model folder. Using the tokenizer in the base model folder...")
        tokenizer = AutoTokenizer.from_pretrained(args.base_model_name_or_path, use_fast=args.use_fast_tokenizer)

    embedding_size = base_model.get_input_embeddings().weight.shape[0]

    if len(tokenizer) > embedding_size:
        print(f"The vocabulary the tokenizer contains {len(tokenizer) - embedding_size} more tokens than the base model.")
        print("Resizing the token embeddings of the merged model...")
        base_model.resize_token_embeddings(len(tokenizer))

    print("Loading and merging the lora model weights...")
    merged_model = load_and_merge_lora(base_model, args.lora_model_name_or_path)

    output_dir = args.output_dir if args.output_dir else args.lora_model_name_or_path
    os.makedirs(output_dir, exist_ok=True)
    
    print(f"Saving merged model to {output_dir}...")
    merged_model.save_pretrained(output_dir)

    if args.save_tokenizer:
        print(f"Saving the tokenizer to {output_dir}...")
        tokenizer.save_pretrained(output_dir)
