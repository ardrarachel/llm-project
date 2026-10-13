import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


def load_backbone(name, device, fp16=True):
    """Load tokenizer + frozen causal LM. Returns tokenizer, model, eos_ids, end_id, pad_id."""
    tok = AutoTokenizer.from_pretrained(name)
    dtype = torch.float16 if (fp16 and device.type == "cuda") else torch.float32
    try:                                   # newer transformers use `dtype`, older ones `torch_dtype`
        model = AutoModelForCausalLM.from_pretrained(name, dtype=dtype)
    except TypeError:
        model = AutoModelForCausalLM.from_pretrained(name, torch_dtype=dtype)
    model = model.to(device)
    model.eval()

    eos = model.generation_config.eos_token_id
    eos_ids = set(eos if isinstance(eos, (list, tuple)) else [eos]) if eos is not None else set()
    if tok.eos_token_id is not None:
        eos_ids.add(tok.eos_token_id)
    eot = tok.convert_tokens_to_ids("<|eot_id|>")
    if eot is not None and eot != tok.unk_token_id:
        eos_ids.add(eot)
        end_id = eot
    else:
        end_id = tok.eos_token_id
    pad_id = tok.pad_token_id if tok.pad_token_id is not None else tok.eos_token_id
    tok.pad_token_id = pad_id
    return tok, model, sorted(i for i in eos_ids if i is not None), end_id, pad_id
