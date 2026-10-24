"""
Step 5 (diagnostic). Does the trained latent z actually carry information, or is it being ignored?

Three checks on the validation responses (run after 03_train.py, takes a few minutes):
  1. NLL per response token under different z:
        none      = plain frozen Llama, no latent        (the baseline)
        zero      = z = 0
        prior     = z ~ N(0, I)                          (what generation uses)
        swapped   = posterior mean of ANOTHER response to the same question
        posterior = this response's own posterior mean   (cheating: it has seen the answer)
     A working latent gives:  posterior < swapped <= prior ~ zero ~ none, with a clear gap.
  2. Style probe: can the style (plain / technical / ...) be predicted from the posterior mean?
     Accuracy well above chance means z encodes the style.
  3. Spread of the posterior mean across responses (the 'active units' idea).

Usage:  python scripts/05_diagnose_latent.py [--config configs/config.yaml]
"""
import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import torch
import torch.nn.functional as F

from src.latent_model.batching import make_group
from src.latent_model.loading import load_backbone
from src.latent_model.model import LatentCVAE
from src.utils import get_device, hf_login_if_possible, load_config, load_jsonl, set_seed


def nll_sum(model, dec_ids, dec_mask, labels, z):
    """Summed NLL over response tokens and the token count, for a given z (None = no latent at all)."""
    with model.amp():
        if z is None:
            hidden = model.backbone.model(input_ids=dec_ids, attention_mask=dec_mask).last_hidden_state
        else:
            hidden = model.decode_hidden(dec_ids, dec_mask, z)
        tgt = labels[:, 1:]
        sel = tgt != -100
        logits = model.backbone.lm_head(hidden[:, :-1][sel])
    return F.cross_entropy(logits.float(), tgt[sel], reduction="sum").item(), int(sel.sum())


@torch.no_grad()
def collect_mu(model, tok, items, cfg, end_id, pad_id, device):
    mcfg = cfg["model"]
    mus, styles, groups = [], [], []
    for gi, it in enumerate(items):
        g = make_group(tok, it, mcfg["max_prompt_tokens"], mcfg["max_response_tokens"], end_id, pad_id, device)
        mu, _ = model.encode(g[0], g[1])
        mus.append(mu.float().cpu().numpy())
        styles += list(it["styles"])
        groups += [gi] * len(it["styles"])
    return np.concatenate(mus, 0), np.array(styles), np.array(groups)


def main(config_path="configs/config.yaml", ckpt_name="phase1_latent.pt", max_val=40, max_train=150):
    cfg = load_config(config_path)
    set_seed(cfg["seed"])
    device = get_device()
    hf_login_if_possible()
    mcfg = cfg["model"]
    tok, backbone, eos_ids, end_id, pad_id = load_backbone(mcfg["backbone"], device, fp16=cfg["training"]["fp16"])
    use_amp = bool(cfg["training"]["fp16"] and device.type == "cuda")
    model = LatentCVAE(backbone, mcfg["latent_dim"], mcfg["num_query_tokens"], mcfg["encoder_layers"],
                       mcfg["adapter_hidden"], use_amp=use_amp,
                       adapter_norm=mcfg.get("adapter_norm", False)).to(device)
    ck = torch.load(os.path.join(cfg["paths"]["output_dir"], ckpt_name), map_location="cpu")
    model.load_trainable(ck["model"])
    model.eval()

    pdir = cfg["paths"]["processed_dir"]
    val = load_jsonl(os.path.join(pdir, "val_responses.jsonl"))[:max_val]
    train = load_jsonl(os.path.join(pdir, "train_responses.jsonl"))[:max_train]
    if not val:
        raise RuntimeError("No validation responses found. Run steps 1-3 first.")

    # ---------------- 1. NLL under different z
    tot = {k: [0.0, 0] for k in ["none", "zero", "prior", "swapped", "posterior"]}
    with torch.no_grad():
        for it in val:
            g = make_group(tok, it, mcfg["max_prompt_tokens"], mcfg["max_response_tokens"], end_id, pad_id, device)
            enc_ids, enc_mask, dec_ids, dec_mask, labels = g
            mu, _ = model.encode(enc_ids, enc_mask)
            zs = {"none": None, "zero": torch.zeros_like(mu), "prior": torch.randn_like(mu),
                  "swapped": torch.roll(mu, 1, dims=0), "posterior": mu}
            for name, z in zs.items():
                s, n = nll_sum(model, dec_ids, dec_mask, labels, z)
                tot[name][0] += s
                tot[name][1] += n
    nll = {k: v[0] / max(v[1], 1) for k, v in tot.items()}

    # ---------------- 2/3. style probe and spread of mu
    mu_val, sty_val, grp_val = collect_mu(model, tok, val, cfg, end_id, pad_id, device)
    mu_tr, sty_tr, _ = collect_mu(model, tok, train, cfg, end_id, pad_id, device)
    classes = sorted(str(c) for c in (set(sty_tr) | set(sty_val)))
    chance = 1.0 / len(classes)
    acc = None
    if len(set(sty_tr)) >= 2 and len(mu_val) >= 2:
        mean, std = mu_tr.mean(0), mu_tr.std(0) + 1e-6
        cent = {c: ((mu_tr[sty_tr == c] - mean) / std).mean(0) for c in set(sty_tr)}
        pred = []
        for x in (mu_val - mean) / std:
            pred.append(min(cent, key=lambda c: float(((x - cent[c]) ** 2).sum())))
        acc = float(np.mean(np.array(pred) == sty_val))
    std_total = float(mu_val.std(0).mean())
    within = [mu_val[grp_val == g].std(0).mean() for g in set(grp_val) if (grp_val == g).sum() > 1]
    std_within = float(np.mean(within)) if within else float("nan")
    active = int((mu_val.var(0) > 0.01).sum())

    # ---------------- report
    print("\n=== 1. NLL per response token (lower = z helps the decoder) ===")
    for k in ["none", "zero", "prior", "swapped", "posterior"]:
        print(f"  z = {k:<10} {nll[k]:.4f}")
    gain = nll["none"] - nll["posterior"]
    print(f"  gain from the response's own z vs no latent: {gain:.4f} nats/token")
    print(f"  gain over another response's z (swapped):    {nll['swapped'] - nll['posterior']:.4f} nats/token")
    print("\n=== 2. Style probe (is the style readable from z?) ===")
    print(f"  classes: {classes}, chance = {chance:.2f}, accuracy = {acc if acc is None else round(acc, 2)}")
    print("\n=== 3. Spread of the posterior mean ===")
    print(f"  std of mu across all responses (mean over dims): {std_total:.4f}")
    print(f"  std of mu within one question's responses:        {std_within:.4f}")
    print(f"  active units (variance > 0.01): {active} of {model.D}")

    works = gain > 0.03 and acc is not None and acc > chance + 0.15
    print("\n=== Verdict (rule of thumb) ===")
    if works:
        print("  z carries information: the latent is being used. Next: look at eval_samples.json for visible variation.")
    else:
        print("  z carries little or no information (partial posterior collapse). Typical fixes, in this order:")
        print("   1. check the training answers really differ (self-BLEU among the 4 answers of a question);")
        print("   2. lower kl_weight to 0.2-0.5 and train longer (epochs 15-20, more questions);")
        print("   3. then retrain and run this script again.")
    return {"nll": nll, "style_acc": acc, "chance": chance, "active_units": active}


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/config.yaml")
    main(ap.parse_args().config)
