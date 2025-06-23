# Copyright (c) BossNET. and affiliates.
# This software may be used and distributed according to the terms of the BoSS Community License Agreement.

import fire
import torch
from peft import PeftModel
from transformers import BossForCausalLM, BossTokenizer


def main(base_model: str,
         peft_model: str,
         output_dir: str):
        
    model = BossForCausalLM.from_pretrained(
        base_model,
        load_in_8bit=False,
        torch_dtype=torch.float16,
        device_map="auto",
        offload_folder="tmp", 
    )
    
    tokenizer = BossTokenizer.from_pretrained(
        base_model
    )
        
    model = PeftModel.from_pretrained(
        model, 
        peft_model, 
        torch_dtype=torch.float16,
        device_map="auto",
        offload_folder="tmp",
    )

    model = model.merge_and_unload()
    model.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)


if __name__ == "__main__":
    fire.Fire(main)