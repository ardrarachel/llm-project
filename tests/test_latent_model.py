"""Run: python tests/test_latent_model.py   (CPU, no downloads: uses a tiny random Llama)"""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import torch
from transformers import LlamaConfig, LlamaForCausalLM
from src.latent_model.model import LatentCVAE, group_kl, instance_kl


def tiny_backbone():
    cfg = LlamaConfig(vocab_size=100, hidden_size=64, intermediate_size=128, num_hidden_layers=4,
                      num_attention_heads=4, num_key_value_heads=2, max_position_embeddings=256,
                      tie_word_embeddings=True, pad_token_id=0)
    torch.manual_seed(0)
    return LlamaForCausalLM(cfg).eval()


def make_batch(M=3, Lp=10, Lr=6, V=100):
    torch.manual_seed(1)
    prompt = torch.randint(3, V, (Lp,))
    seqs = [torch.cat([prompt, torch.randint(3, V, (Lr - (i % 2),))]) for i in range(M)]
    L = max(len(s) for s in seqs)
    enc_ids = torch.zeros(M, L, dtype=torch.long); enc_mask = torch.zeros(M, L, dtype=torch.long)
    dec_ids = torch.zeros(M, L, dtype=torch.long); dec_mask = torch.zeros(M, L, dtype=torch.long)
    labels = torch.full((M, L), -100)
    for i, s in enumerate(seqs):
        n = len(s)
        enc_ids[i, L - n:] = s; enc_mask[i, L - n:] = 1
        dec_ids[i, :n] = s; dec_mask[i, :n] = 1
        labels[i, Lp:n] = s[Lp:]
    return enc_ids, enc_mask, dec_ids, dec_mask, labels


def build(adapter_norm=False):
    return LatentCVAE(tiny_backbone(), latent_dim=8, num_query_tokens=4, encoder_layers=2, adapter_hidden=16,
                      adapter_norm=adapter_norm)


def test_loss_and_grads():
    m = build(); m.train()
    out = m.compute_loss(*make_batch(), beta=1.0)
    assert torch.isfinite(out["loss"]), out
    out["loss"].backward()
    for name in ["query", "post_head.0.weight", "adapter.0.0.weight", "adapter.3.2.weight", "skip.0.weight",
                 "encoder.layers.0.self_attn.q_proj.weight", "encoder.layers.1.mlp.up_proj.weight"]:
        p = dict(m.named_parameters())[name]
        assert p.grad is not None and p.grad.abs().sum() > 0, f"no gradient for {name}"
    assert all(p.grad is None for p in m.backbone.parameters()), "backbone must stay frozen"
    n_train = sum(p.numel() for p in m.trainable_parameters())
    print(f"ok loss/grad flow  (loss={out['loss'].item():.3f}, trainable={n_train})")


def test_gskip_forward_is_zero_but_gradient_flows():
    m = build(); m.train()
    batch = make_batch()
    torch.manual_seed(5); a = m.compute_loss(*batch)["recon_per_token"].item()
    with torch.no_grad():
        for p in m.skip.parameters():
            p.mul_(25.0)
    torch.manual_seed(5); b = m.compute_loss(*batch)["recon_per_token"].item()
    assert abs(a - b) < 1e-5, (a, b)
    torch.manual_seed(5); m.compute_loss(*batch)["loss"].backward()
    assert m.skip[0].weight.grad.abs().sum() > 0
    print("ok G-Skip: forward value unchanged, gradient present")


def test_latent_changes_output():
    m = build()
    ids = torch.randint(3, 100, (1, 12)).expand(2, -1)
    mask = torch.ones_like(ids)
    z = torch.randn(2, 8)
    with torch.no_grad():
        h = m.decode_hidden(ids, mask, z)
    assert (h[0] - h[1]).abs().max() > 1e-4, "different z must give different hidden states"
    print("ok different z -> different outputs")


def test_incremental_decoding_matches_teacher_forcing():
    m = build()
    torch.manual_seed(2)
    ids = torch.randint(3, 100, (1, 14)); z = torch.randn(1, 8)
    with torch.no_grad():
        full = m.backbone.lm_head(m.decode_hidden(ids, torch.ones_like(ids), z))
        cache = m.build_cache(z)
        attn = torch.ones(1, 1 + 8, dtype=torch.long)
        out = m.backbone.model(input_ids=ids[:, :8], attention_mask=attn, past_key_values=cache, use_cache=True)
        cache = out.past_key_values
        logits = [m.backbone.lm_head(out.last_hidden_state)]
        for t in range(8, 14):
            attn = torch.cat([attn, torch.ones(1, 1, dtype=torch.long)], 1)
            out = m.backbone.model(input_ids=ids[:, t:t + 1], attention_mask=attn, past_key_values=cache, use_cache=True)
            cache = out.past_key_values
            logits.append(m.backbone.lm_head(out.last_hidden_state))
        inc = torch.cat(logits, 1)
    assert torch.allclose(full, inc, atol=1e-4), (full - inc).abs().max()
    print("ok incremental decoding == teacher forcing")


def test_group_kl():
    M, D = 6, 8
    mu, lv = torch.zeros(M, D), torch.zeros(M, D)
    z = torch.randn(M, D)
    assert abs(group_kl(mu, lv, z).item()) < 1e-5
    mu2 = torch.randn(M, D) * 3
    z2 = mu2 + 0.1 * torch.randn(M, D)
    assert group_kl(mu2, torch.full((M, D), -4.0), z2).item() > 1.0
    assert instance_kl(mu2, torch.full((M, D), -4.0)).item() > 1.0
    print("ok group KL (zero for N(0,I), large for far-away posteriors)")


def test_generate():
    m = build()
    prompt = list(range(3, 15))
    g = m.generate(prompt, num_samples=3, max_new_tokens=9, temperature=1.0, top_p=0.9, eos_ids=[2], pad_id=0)
    assert g.shape[0] == 3 and 1 <= g.shape[1] <= 9
    b = m.generate(prompt, num_samples=3, use_latent=False, max_new_tokens=9, temperature=1.0, top_p=0.9, eos_ids=[2], pad_id=0)
    assert b.shape[0] == 3
    # the same z must give the same greedy answer; different z should usually differ
    z = torch.randn(1, 8).expand(2, -1).contiguous()
    g2 = m.generate(prompt, num_samples=2, z=z, max_new_tokens=9, temperature=0.0, top_p=1.0, eos_ids=[], pad_id=0)
    assert torch.equal(g2[0], g2[1])
    print("ok generate (latent + baseline, deterministic for fixed z)")


def test_adapter_norm():
    m = build(adapter_norm=True); m.train()
    out = m.compute_loss(*make_batch(), beta=1.0)
    out["loss"].backward()
    assert torch.isfinite(out["loss"]) and m.adapter_gain.grad is not None and m.adapter_gain.grad.abs().sum() > 0
    # latent key/value must now have a size independent of the adapter's output scale
    z = torch.randn(2, 8)
    c1 = m.build_cache(z)
    with torch.no_grad():
        for p in m.adapter.parameters():
            p.mul_(50.0)
    c2 = m.build_cache(z)
    k1 = c1.layers[0].keys if hasattr(c1, "layers") else c1.key_cache[0]
    k2 = c2.layers[0].keys if hasattr(c2, "layers") else c2.key_cache[0]
    assert abs(k1.norm().item() - k2.norm().item()) / k1.norm().item() < 0.2, (k1.norm(), k2.norm())
    # incremental decoding must still equal teacher forcing with the normalisation on
    m.eval(); torch.manual_seed(2)
    ids = torch.randint(3, 100, (1, 14)); zz = torch.randn(1, 8)
    with torch.no_grad():
        full = m.backbone.lm_head(m.decode_hidden(ids, torch.ones_like(ids), zz))
        cache = m.build_cache(zz); attn = torch.ones(1, 1 + 8, dtype=torch.long)
        o = m.backbone.model(input_ids=ids[:, :8], attention_mask=attn, past_key_values=cache, use_cache=True)
        cache, logits = o.past_key_values, [m.backbone.lm_head(o.last_hidden_state)]
        for t in range(8, 14):
            attn = torch.cat([attn, torch.ones(1, 1, dtype=torch.long)], 1)
            o = m.backbone.model(input_ids=ids[:, t:t + 1], attention_mask=attn, past_key_values=cache, use_cache=True)
            cache = o.past_key_values; logits.append(m.backbone.lm_head(o.last_hidden_state))
    assert torch.allclose(full, torch.cat(logits, 1), atol=1e-4)
    # the default (adapter_norm=False) keeps the old checkpoint layout
    assert not any("adapter_gain" in k for k in build().trainable_state())
    print("ok adapter_norm (scale-free latent K/V, gradients, decoding consistent, old checkpoints still load)")


def test_save_load():
    m = build(); sd = m.trainable_state()
    assert not any(k.startswith("backbone.") for k in sd)
    m2 = build(); m2.load_trainable(sd)
    assert torch.equal(m.query, m2.query)
    print(f"ok save/load ({len(sd)} tensors, no backbone weights)")


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_"):
            fn()
    print("ALL LATENT MODEL TESTS PASSED")
