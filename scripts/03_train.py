"""
Step 3. Train the latent encoder, latent adapter and G-Skip with the DG-ELBo objective.
The Llama backbone stays frozen (as in the paper).

Usage:
    python scripts/03_train.py            # train from scratch
    python scripts/03_train.py --resume   # continue from outputs/ckpt_last.pt after a Kaggle cutoff
Outputs: outputs/phase1_latent.pt (best validation loss), outputs/ckpt_last.pt, outputs/train_log.json
"""
import argparse
import os

os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")   # fewer fragmentation OOMs
import random
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch

from src.latent_model.batching import make_group
from src.latent_model.loading import load_backbone
from src.latent_model.model import LatentCVAE
from src.utils import ensure_dir, get_device, hf_login_if_possible, load_config, load_jsonl, save_json, set_seed


@torch.no_grad()
def evaluate(model, tok, items, cfg, end_id, pad_id, device, beta):
    model.eval()
    tot = {"loss": 0.0, "recon_per_token": 0.0, "kl_group": 0.0, "kl_instance": 0.0}
    mus = []
    for it in items:
        g = make_group(tok, it, cfg["model"]["max_prompt_tokens"], cfg["model"]["max_response_tokens"],
                       end_id, pad_id, device)
        out = model.compute_loss(*g, beta=beta)
        for k in tot:
            tot[k] += float(out[k])
        mus.append(out["mu"].float().cpu())
    n = max(len(items), 1)
    res = {k: v / n for k, v in tot.items()}
    mu = torch.cat(mus, 0) if mus else torch.zeros(1, model.D)
    res["active_units"] = int((mu.var(0) > 0.01).sum()) if mu.size(0) > 1 else 0
    model.train()
    return res


def plot_curves(log, out_dir):
    """Save outputs/training_curves.png (reconstruction, KL, validation loss). Skipped if matplotlib is missing."""
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        return
    tr = [e for e in log if "loss" in e]
    vl = [e for e in log if e.get("val")]
    if not tr:
        return
    fig, ax = plt.subplots(1, 3, figsize=(15, 4))
    ax[0].plot([e["step"] for e in tr], [e["recon_per_token"] for e in tr])
    ax[0].set_title("Reconstruction loss per token")
    ax[1].plot([e["step"] for e in tr], [e["kl_instance"] for e in tr], label="KL (per response)")
    ax[1].plot([e["step"] for e in tr], [e["kl_group"] for e in tr], label="KL (group)")
    ax[1].legend(); ax[1].set_title("KL (should not stay near 0)")
    if vl:
        ax[2].plot([e["epoch"] for e in vl], [e["val"]["loss"] for e in vl], marker="o")
    ax[2].set_title("Validation loss")
    for a, name in zip(ax, ["step", "step", "epoch"]):
        a.set_xlabel(name)
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "training_curves.png"), dpi=150)
    plt.close(fig)


def train(cfg, resume=False):
    set_seed(cfg["seed"])
    device = get_device()
    tcfg = cfg["training"]
    out_dir = ensure_dir(cfg["paths"]["output_dir"])
    hf_login_if_possible()

    tok, backbone, eos_ids, end_id, pad_id = load_backbone(cfg["model"]["backbone"], device, fp16=tcfg["fp16"])
    use_amp = bool(tcfg["fp16"] and device.type == "cuda")
    mcfg = cfg["model"]
    model = LatentCVAE(backbone, mcfg["latent_dim"], mcfg["num_query_tokens"], mcfg["encoder_layers"],
                       mcfg["adapter_hidden"], use_amp=use_amp,
                       adapter_norm=mcfg.get("adapter_norm", False),
                       adapter_gain_init=mcfg.get("adapter_gain_init", 1.0),
                       feature_norm=mcfg.get("feature_norm", False),
                       mu_bound=mcfg.get("mu_bound", 0.0)).to(device)
    model.train()
    params = model.trainable_parameters()
    print(f"Trainable parameters: {sum(p.numel() for p in params) / 1e6:.1f}M "
          f"(backbone frozen: {sum(p.numel() for p in backbone.parameters()) / 1e6:.0f}M)")

    # Optional higher learning rate for the decoder-side latent path (adapter + G-Skip FFN), so the decoder
    # learns to use z sooner. latent_lr_mult = 1.0 reproduces the earlier behaviour.
    mult = float(tcfg.get("latent_lr_mult", 1.0))
    is_latent = lambda n: n.startswith("adapter") or n.startswith("skip")
    enc_params = [p for n, p in model.named_parameters() if p.requires_grad and not is_latent(n)]
    lat_params = [p for n, p in model.named_parameters() if p.requires_grad and is_latent(n)]
    print(f"learning rates: encoder side {tcfg['lr']:g} ({sum(p.numel() for p in enc_params) / 1e6:.0f}M params), "
          f"latent path {tcfg['lr'] * mult:g} ({sum(p.numel() for p in lat_params) / 1e6:.0f}M params)")
    opt = torch.optim.AdamW([{"params": enc_params, "lr": tcfg["lr"]}, {"params": lat_params, "lr": tcfg["lr"] * mult}],
                            weight_decay=tcfg["weight_decay"])
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)

    pdir = cfg["paths"]["processed_dir"]
    train_items = load_jsonl(os.path.join(pdir, "train_responses.jsonl"))
    val_items = load_jsonl(os.path.join(pdir, "val_responses.jsonl"))
    print(f"Training groups: {len(train_items)}, validation groups: {len(val_items)}")
    if not train_items:
        raise RuntimeError("No training data. Run scripts/02_generate_training_data.py first.")

    start_epoch, step, best_val, log = 0, 0, float("inf"), []
    last_path = os.path.join(out_dir, "ckpt_last.pt")
    if resume and os.path.exists(last_path):
        ck = torch.load(last_path, map_location="cpu")
        model.load_trainable(ck["model"])
        opt.load_state_dict(ck["opt"])
        scaler.load_state_dict(ck["scaler"])
        start_epoch, step, best_val, log = ck["epoch"], ck["step"], ck["best_val"], ck["log"]
        print(f"Resumed from epoch {start_epoch}, step {step}")

    wandb = None
    if tcfg.get("use_wandb"):
        import wandb as _wandb
        _wandb.init(project="legal-rag-latent", config=cfg)
        wandb = _wandb

    accum = tcfg["grad_accum"]
    kl_abort = float(tcfg.get("kl_abort", 500))        # stop early if KL stays above this for 3 log intervals
    bad_kl, bad_loss = 0, 0
    rng = random.Random(cfg["seed"])
    for epoch in range(start_epoch, tcfg["epochs"]):
        order = list(range(len(train_items)))
        rng.shuffle(order)
        run = {"loss": 0.0, "recon_per_token": 0.0, "kl_group": 0.0, "kl_instance": 0.0, "n": 0}
        t0 = time.time()
        for i, idx in enumerate(order):
            beta = tcfg["kl_weight"] * min(1.0, (step + 1) / max(1, tcfg["kl_warmup_steps"]))
            g = make_group(tok, train_items[idx], mcfg["max_prompt_tokens"], mcfg["max_response_tokens"],
                           end_id, pad_id, device)
            out = model.compute_loss(*g, beta=beta)
            if not torch.isfinite(out["loss"]):
                bad_loss += 1
                print("Non-finite loss, skipping this group")
                del out, g                      # free the graph, otherwise the next group runs out of memory
                opt.zero_grad(set_to_none=True)
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                if bad_loss >= 8:
                    raise RuntimeError("Training diverged (8 non-finite losses in a row). Lower training.lr, "
                                       "turn on model.feature_norm and lower model.adapter_gain_init, then retrain.")
                continue
            bad_loss = 0
            scaler.scale(out["loss"] / accum).backward()
            for k in ["loss", "recon_per_token", "kl_group", "kl_instance"]:
                run[k] += float(out[k].detach())
            run["n"] += 1

            if (i + 1) % accum == 0 or i == len(order) - 1:
                scaler.unscale_(opt)
                torch.nn.utils.clip_grad_norm_(params, tcfg["grad_clip"])
                scaler.step(opt)
                scaler.update()
                opt.zero_grad(set_to_none=True)
                step += 1
                if step % tcfg["log_every"] == 0 and run["n"]:
                    avg = {k: run[k] / run["n"] for k in ["loss", "recon_per_token", "kl_group", "kl_instance"]}
                    mem = f" | peak GPU {torch.cuda.max_memory_allocated() / 1e9:.1f}GB" if device.type == "cuda" else ""
                    print(f"ep {epoch + 1} step {step} | loss {avg['loss']:.3f} | recon/token {avg['recon_per_token']:.3f} "
                          f"| KL(group) {avg['kl_group']:.2f} | KL(inst) {avg['kl_instance']:.2f} | beta {beta:.2f} "
                          f"| {time.time() - t0:.0f}s{mem}")
                    log.append({"epoch": epoch + 1, "step": step, "beta": beta, **avg})
                    bad_kl = bad_kl + 1 if avg["kl_instance"] > kl_abort else 0
                    if bad_kl >= 3:
                        plot_curves(log, out_dir)
                        raise RuntimeError(f"KL exploded (KL(inst) {avg['kl_instance']:.0f} for 3 log intervals in a row). "
                                           "Stopping early. Lower training.lr, enable model.feature_norm, lower model.adapter_gain_init.")
                    if wandb:
                        wandb.log({"step": step, "beta": beta, **avg})
                    run = {"loss": 0.0, "recon_per_token": 0.0, "kl_group": 0.0, "kl_instance": 0.0, "n": 0}

        val = evaluate(model, tok, val_items, cfg, end_id, pad_id, device, tcfg["kl_weight"]) if val_items else {}
        print(f"== epoch {epoch + 1} validation: {val}")
        log.append({"epoch": epoch + 1, "step": step, "val": val})
        if val and val["loss"] < best_val:
            best_val = val["loss"]
            torch.save({"model": model.trainable_state(), "cfg": cfg, "val": val}, os.path.join(out_dir, "phase1_latent.pt"))
            print("   saved best checkpoint")
        elif not val:
            torch.save({"model": model.trainable_state(), "cfg": cfg, "val": {}}, os.path.join(out_dir, "phase1_latent.pt"))
        torch.save({"model": model.trainable_state(), "opt": opt.state_dict(), "scaler": scaler.state_dict(),
                    "epoch": epoch + 1, "step": step, "best_val": best_val, "log": log}, last_path)
        save_json(log, os.path.join(out_dir, "train_log.json"))
    plot_curves(log, out_dir)
    print("Training finished.")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/config.yaml")
    ap.add_argument("--resume", action="store_true")
    args = ap.parse_args()
    train(load_config(args.config), resume=args.resume)
