"""The 4-bit (QLoRA) base-model settings, in one place. Training (train.py) and the evaluation of QLoRA adapters on the
4-bit base (evaluate.py --load-4bit) both build the config here, so the two can never differ.
The values come from configs/phase3.yaml (methods.qlora_4bit)."""


def quant_settings(method: dict) -> dict:
    """Plain dict of the bitsandbytes settings (no library needed). Also written into the eval output."""
    return {"load_in_4bit": True, "bnb_4bit_quant_type": method["quantize"],
            "bnb_4bit_use_double_quant": method["double_quant"], "bnb_4bit_compute_dtype": method["compute_dtype"]}


def bnb_config(method: dict, torch):
    """The BitsAndBytesConfig for this method (transformers is imported here, not at module level)."""
    from transformers import BitsAndBytesConfig
    s = quant_settings(method)
    return BitsAndBytesConfig(load_in_4bit=s["load_in_4bit"], bnb_4bit_quant_type=s["bnb_4bit_quant_type"],
                              bnb_4bit_use_double_quant=s["bnb_4bit_use_double_quant"],
                              bnb_4bit_compute_dtype=getattr(torch, s["bnb_4bit_compute_dtype"]))
