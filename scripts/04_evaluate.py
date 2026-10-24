"""
Step 4. Phase 1 evaluation on the TEST split (contracts never seen in training).

For every test question we draw k answers in two ways with the SAME sampler settings:
    baseline : plain Llama RAG, k samples with temperature/top-p
    latent   : Llama + latent module, k samples each with its own z ~ N(0, I)
and compare diversity (distinct-n, self-BLEU) and grounding (words supported by the retrieved
excerpts, and recall of the gold clause words).

Outputs: outputs/eval_summary.json, outputs/eval_samples.json (read these answers yourself!)
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import torch
from tqdm import tqdm

from src.evaluation.metrics import distinct_n, gold_recall, grounding, pairwise_length_spread, self_bleu
from src.latent_model.loading import load_backbone
from src.latent_model.model import LatentCVAE
from src.prompts import build_prompt_ids
from src.utils import get_device, hf_login_if_possible, load_config, load_json, save_json, set_seed


def decode(tok, gen, eos_ids):
    texts = []
    for row in gen.tolist():
        cut = next((i for i, t in enumerate(row) if t in eos_ids), len(row))
        texts.append(tok.decode(row[:cut], skip_special_tokens=True).strip())
    return texts


def score(texts, context_text, gold_text):
    return {
        "distinct_1": distinct_n(texts, 1), "distinct_2": distinct_n(texts, 2), "self_bleu": self_bleu(texts),
        "grounding": float(np.mean([grounding(t, context_text) for t in texts])),
        "gold_recall": float(np.mean([gold_recall(t, gold_text) for t in texts])),
        "length_std": pairwise_length_spread(texts),
        "avg_words": float(np.mean([len(t.split()) for t in texts])),
    }


def main(config_path="configs/config.yaml", ckpt_name="phase1_latent.pt"):
    cfg = load_config(config_path)
    set_seed(cfg["seed"])
    device = get_device()
    hf_login_if_possible()
    ecfg, mcfg = cfg["eval"], cfg["model"]

    tok, backbone, eos_ids, end_id, pad_id = load_backbone(mcfg["backbone"], device, fp16=cfg["training"]["fp16"])
    use_amp = bool(cfg["training"]["fp16"] and device.type == "cuda")
    model = LatentCVAE(backbone, mcfg["latent_dim"], mcfg["num_query_tokens"], mcfg["encoder_layers"],
                       mcfg["adapter_hidden"], use_amp=use_amp,
                       adapter_norm=mcfg.get("adapter_norm", False),
                       adapter_gain_init=mcfg.get("adapter_gain_init", 1.0),
                       feature_norm=mcfg.get("feature_norm", False),
                       mu_bound=mcfg.get("mu_bound", 0.0)).to(device)
    ck = torch.load(os.path.join(cfg["paths"]["output_dir"], ckpt_name), map_location="cpu")
    model.load_trainable(ck["model"])
    model.eval()

    test = load_json(os.path.join(cfg["paths"]["processed_dir"], "test.json"))
    rows, samples = {"baseline": [], "latent": []}, []
    for it in tqdm(test):
        chunks = [r["text"] for r in it["retrieved"]]
        prompt = build_prompt_ids(tok, chunks, it["query"], mcfg["max_prompt_tokens"])
        context_text = " ".join(chunks) + " " + it["query"]
        gold_text = " ".join(g.get("text", "") or g.get("answer", "") for g in it["gold"])
        kw = dict(num_samples=ecfg["num_samples"], max_new_tokens=ecfg["max_new_tokens"],
                  temperature=ecfg["temperature"], top_p=ecfg["top_p"], eos_ids=eos_ids, pad_id=pad_id)
        base = decode(tok, model.generate(prompt, use_latent=False, **kw), set(eos_ids))
        lat = decode(tok, model.generate(prompt, use_latent=True, **kw), set(eos_ids))
        rows["baseline"].append(score(base, context_text, gold_text))
        rows["latent"].append(score(lat, context_text, gold_text))
        samples.append({"qid": it["qid"], "query": it["query"], "baseline": base, "latent": lat})

    summary = {name: {k: float(np.mean([r[k] for r in rs])) for k in rs[0]} for name, rs in rows.items()}
    save_json(summary, os.path.join(cfg["paths"]["output_dir"], "eval_summary.json"))
    save_json(samples, os.path.join(cfg["paths"]["output_dir"], "eval_samples.json"))

    print("\n=== Phase 1 results (test split, mean over questions) ===")
    print(f"{'metric':<12}{'baseline':>12}{'latent':>12}   (distinct-n up = more diverse, self-BLEU down = more diverse)")
    for k in summary["baseline"]:
        print(f"{k:<12}{summary['baseline'][k]:>12.3f}{summary['latent'][k]:>12.3f}")


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/config.yaml")
    a = ap.parse_args()
    main(a.config)
