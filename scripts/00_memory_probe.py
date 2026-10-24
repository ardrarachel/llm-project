"""
Memory probe: run ONE worst-case training step (full-length prompt + response for every response in the
group) with the chosen config, and report peak GPU memory. Takes ~2-3 minutes, needs the model but NOT the
dataset. Run it before generating data, so you know whether a config fits the GPU.

    python scripts/00_memory_probe.py --config configs/config_3b.yaml
"""
import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch

from src.latent_model.loading import load_backbone
from src.latent_model.model import LatentCVAE
from src.utils import get_device, hf_login_if_possible, load_config


def probe(cfg):
    device = get_device()
    hf_login_if_possible()
    mcfg, tcfg = cfg["model"], cfg["training"]
    M = len(cfg["generation_data"]["styles"])
    Lp, Lr = mcfg["max_prompt_tokens"], mcfg["max_response_tokens"]
    tok, backbone, eos_ids, end_id, pad_id = load_backbone(mcfg["backbone"], device, fp16=tcfg["fp16"])
    use_amp = bool(tcfg["fp16"] and device.type == "cuda")
    model = LatentCVAE(backbone, mcfg["latent_dim"], mcfg["num_query_tokens"], mcfg["encoder_layers"],
                       mcfg["adapter_hidden"], use_amp=use_amp,
                       adapter_norm=mcfg.get("adapter_norm", False)).to(device)
    model.train()
    params = model.trainable_parameters()
    n_train = sum(p.numel() for p in params) / 1e6
    opt = torch.optim.AdamW(params, lr=1e-4)
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)

    V = backbone.config.vocab_size
    lo = max(3, min(1000, V // 4))
    ids = torch.randint(lo, min(V, 20000), (M, Lp + Lr), device=device)
    mask = torch.ones_like(ids)
    labels = ids.clone(); labels[:, :Lp] = -100
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats()
    out = model.compute_loss(ids, mask, ids, mask, labels, beta=1.0)
    scaler.scale(out["loss"]).backward()
    scaler.step(opt); scaler.update(); opt.zero_grad(set_to_none=True)

    print(f"\nbackbone: {mcfg['backbone']} | trainable parameters: {n_train:.0f}M | group size M={M} | tokens per row: {Lp + Lr}")
    if device.type == "cuda":
        peak = torch.cuda.max_memory_allocated() / 1e9
        total = torch.cuda.get_device_properties(0).total_memory / 1e9
        verdict = "FITS" if peak < 0.90 * total else "TOO TIGHT - reduce max_prompt_tokens, top_k, styles (M) or encoder_layers"
        print(f"peak GPU memory: {peak:.1f} GB of {total:.1f} GB  ->  {verdict}")
        print("(real batches are usually shorter than this worst case, so there is some headroom)")
    else:
        print("No GPU found, so memory was not measured. Run this on the Kaggle GPU.")
    return out


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/config.yaml")
    probe(load_config(ap.parse_args().config))
