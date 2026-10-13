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
import random
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import matplotlib.pyplot as plt
import torch

from src.latent_model.batching import make_group
from src.latent_model.loading import load_backbone
from src.latent_model.model import LatentCVAE
from src.utils import ensure_dir, get_device, hf_login_if_possible, load_config, load_jsonl, save_json, set_seed


def plot_and_save_metrics(log, output_dir="llm-project", filename="traingraph.png"):
    """Generates a clean 4-panel report plot from training logs and saves it."""
    os.makedirs(output_dir, exist_ok=True)
    save_path = os.path.join(output_dir, filename)

    train_steps = [entry for entry in log if "recon_per_token" in entry]
    val_steps = [entry for entry in log if "val" in entry and entry["val"]]

    if not train_steps:
        return

    steps = [e["step"] for e in train_steps]
    losses = [e["loss"] for e in train_steps]
    recon = [e["recon_per_token"] for e in train_steps]
    kl_grp = [e["kl_group"] for e in train_steps]
    kl_inst = [e["kl_instance"] for e in train_steps]
    betas = [e["beta"] for e in train_steps]

    fig, axes = plt.subplots(2, 2, figsize=(12, 8), dpi=300)
    plt.subplots_adjust(hspace=0.3, wspace=0.25)

    # 1. Total Loss (Train + Val)
    axes[0, 0].plot(steps, losses, label="Train DG-ELBo Loss", color="#1f77b4", linewidth=1.5)
    if val_steps:
        val_s = [e["step"] for e in val_steps]
        val_l = [e["val"]["loss"] for e in val_steps]
        axes[0, 0].plot(val_s, val_l, label="Val Loss", color="#d62728", marker="o", linestyle="--", linewidth=1.5)
    axes[0, 0].set_title("DG-ELBo Loss", fontsize=11, fontweight="bold")
    axes[0, 0].set_xlabel("Steps")
    axes[0, 0].set_ylabel("Loss")
    axes[0, 0].grid(True, linestyle=":", alpha=0.6)
    axes[0, 0].legend()

    # 2. Reconstruction Loss
    axes[0, 1].plot(steps, recon, label="Recon / Token", color="#2ca02c", linewidth=1.5)
    axes[0, 1].set_title("Reconstruction Loss per Token", fontsize=11, fontweight="bold")
    axes[0, 1].set_xlabel("Steps")
    axes[0, 1].set_ylabel("Cross Entropy")
    axes[0, 1].grid(True, linestyle=":", alpha=0.6)
    axes[0, 1].legend()

    # 3. KL Divergence (Group vs. Instance)
    axes[1, 0].plot(steps, kl_grp, label="KL (Group)", color="#9467bd", linewidth=1.5)
    axes[1, 0].plot(steps, kl_inst, label="KL (Instance)", color="#8c564b", linestyle="--", linewidth=1.2)
    axes[1, 0].set_title("KL Divergence", fontsize=11, fontweight="bold")
    axes[1, 0].set_xlabel("Steps")
    axes[1, 0].set_ylabel("KL")
    axes[1, 0].grid(True, linestyle=":", alpha=0.6)
    axes[1, 0].legend()

    # 4. Beta Schedule & Active Latent Units
    ax_beta = axes[1, 1]
    ax_beta.plot(steps, betas, label="Beta (KL Weight)", color="#ff7f0e", linewidth=1.5)
    ax_beta.set_title("Beta Schedule & Validation Active Units", fontsize=11, fontweight="bold")
    ax_beta.set_xlabel("Steps")
    ax_beta.set_ylabel("Beta", color="#ff7f0e")
    ax_beta.tick_params(axis="y", labelcolor="#ff7f0e")
    ax_beta.grid(True, linestyle=":", alpha=0.6)

    if val_steps and "active_units" in val_steps[0]["val"]:
        ax_act = ax_beta.twinx()
        val_s = [e["step"] for e in val_steps]
        val_act = [e["val"]["active_units"] for e in val_steps]
        ax_act.plot(val_s, val_act, label="Active Units (>0.01 var)", color="#17becf", marker="s", linestyle="-.")
        ax_act.set_ylabel("Active Units", color="#17becf")
        ax_act.tick_params(axis="y", labelcolor="#17becf")

    plt.savefig(save_path, bbox_inches="tight")
    plt.close()


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
                       mcfg["adapter_hidden"], use_amp=use_amp).to(device)
    model.train()
    params = model.trainable_parameters()
    print(f"Trainable parameters: {sum(p.numel() for p in params) / 1e6:.1f}M "
          f"(backbone frozen: {sum(p.numel() for p in backbone.parameters()) / 1e6:.0f}M)")

    opt = torch.optim.AdamW(params, lr=tcfg["lr"], weight_decay=tcfg["weight_decay"])
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
                print("Non-finite loss, skipping this group")
                continue
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
                    print(f"ep {epoch + 1} step {step} | loss {avg['loss']:.3f} | recon/token {avg['recon_per_token']:.3f} "
                          f"| KL(group) {avg['kl_group']:.2f} | KL(inst) {avg['kl_instance']:.2f} | beta {beta:.2f} "
                          f"| {time.time() - t0:.0f}s")
                    log.append({"epoch": epoch + 1, "step": step, "beta": beta, **avg})
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

        # Save and update graph after every epoch
        plot_and_save_metrics(log, output_dir="llm-project", filename="traingraph.png")

    print("Training finished.")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/config.yaml")
    ap.add_argument("--resume", action="store_true")
    args = ap.parse_args()
    train(load_config(args.config), resume=args.resume)