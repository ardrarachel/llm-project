"""
Step 2. Build the self-supervised training data, as in the paper: the BACKBONE writes M responses per
instruction x = (retrieved excerpts + question). Style hints (plain / technical / concise / cautious)
are used only here, to make the M responses genuinely different. The training prompt later has no hint.

Output: data/processed/{train,val}_responses.jsonl  (resumable: finished questions are skipped)
        each row: qid, query, chunks[list of text], responses[list], styles[list]
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
from tqdm import tqdm

from src.latent_model.loading import load_backbone
from src.prompts import build_prompt_text
from src.utils import append_jsonl, get_device, hf_login_if_possible, load_config, load_json, load_jsonl, set_seed


def main(config_path="configs/config.yaml"):
    cfg = load_config(config_path)
    set_seed(cfg["seed"])
    gcfg = cfg["generation_data"]
    device = get_device()
    hf_login_if_possible()

    tok, model, eos_ids, end_id, pad_id = load_backbone(cfg["model"]["backbone"], device, fp16=True)
    tok.padding_side = "left"

    for split in ["train", "val"]:
        items = load_json(os.path.join(cfg["paths"]["processed_dir"], f"{split}.json"))
        out_path = os.path.join(cfg["paths"]["processed_dir"], f"{split}_responses.jsonl")
        done = {r["qid"] for r in load_jsonl(out_path)}
        todo = [it for it in items if it["qid"] not in done]
        print(f"[{split}] {len(items)} questions, {len(done)} already done, {len(todo)} to generate")

        per_batch = max(1, gcfg["batch_size"] // len(gcfg["styles"]))   # questions per batch
        for i in tqdm(range(0, len(todo), per_batch)):
            batch = todo[i:i + per_batch]
            prompts, owners = [], []
            for it in batch:
                chunks = [r["text"] for r in it["retrieved"]]
                for style in gcfg["styles"]:
                    prompts.append(build_prompt_text(tok, chunks, it["query"], style=style))
                    owners.append((it["qid"], style))
            enc = tok(prompts, return_tensors="pt", padding=True, add_special_tokens=False).to(device)
            with torch.no_grad():
                out = model.generate(**enc, do_sample=True, temperature=gcfg["temperature"], top_p=gcfg["top_p"],
                                     max_new_tokens=gcfg["max_new_tokens"], pad_token_id=pad_id,
                                     eos_token_id=eos_ids)
            texts = tok.batch_decode(out[:, enc.input_ids.size(1):], skip_special_tokens=True)
            by_q = {}
            for (qid, style), text in zip(owners, texts):
                text = text.strip()
                if len(text.split()) >= gcfg["min_words"]:
                    by_q.setdefault(qid, []).append((style, text))
            for it in batch:
                pairs = by_q.get(it["qid"], [])
                if len(pairs) < 2:          # a group needs at least 2 responses
                    continue
                append_jsonl(out_path, {
                    "qid": it["qid"], "query": it["query"],
                    "chunks": [r["text"] for r in it["retrieved"]],
                    "styles": [p[0] for p in pairs], "responses": [p[1] for p in pairs],
                })
        print(f"[{split}] saved to {out_path}")


if __name__ == "__main__":
    main()
