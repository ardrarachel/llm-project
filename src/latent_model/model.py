"""
LLM-based CVAE following Zhang et al., "Latent Variable Modeling for Controllable and Diverse
Generation from LLMs" (IEEE TAI 2026), scaled down to Llama-3.2-1B.

Components (section numbers refer to the paper):
  III-A latent encoder  : embedding layer + first two transformer layers copied from the backbone,
                          a learnable 4-token latent query appended to the END of (x, y), and a
                          two-layer MLP giving mu and sigma^2 of z (Dim = 32).
  III-B latent adapter  : per-layer MLP F_l maps z to a hidden state h_{l,z}; the backbone's own
                          W_K,l / W_V,l turn it into one extra key/value that is concatenated to the
                          layer's KV cache ([K_{l,z}; K_l], [V_{l,z}; V_l]).
  III-C DG-ELBo         : reconstruction of the group of responses + a GROUP-level KL between the
                          aggregated posterior q(z|x) (mixture over the M responses) and the prior
                          N(0, I), estimated by Monte Carlo (density gap).
  III-D G-Skip          : S_latent = lm_head(FFN(z)); S_G-Skip = S_latent - sg(S_latent) is added to
                          the backbone logits. Forward value is exactly 0, gradient still flows.

The backbone is frozen. Only the encoder copy, latent query, MLP head, adapter and skip FFN train.
"""
import copy
import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import DynamicCache, LlamaModel

LOG2PI = math.log(2 * math.pi)


def group_kl(mu, logvar, z):
    """Monte-Carlo estimate of KL( q_agg(z|x) || N(0,I) ).
    q_agg is the average of the M per-response Gaussians of ONE instruction x. mu, logvar, z: (M, D)."""
    M = mu.size(0)
    var = logvar.exp()
    diff = z.unsqueeze(1) - mu.unsqueeze(0)                                   # (M_z, M_j, D)
    logq = -0.5 * ((diff ** 2) / var.unsqueeze(0) + logvar.unsqueeze(0) + LOG2PI).sum(-1)
    logq_agg = torch.logsumexp(logq, dim=1) - math.log(M)                      # (M,)
    logp = -0.5 * (z ** 2 + LOG2PI).sum(-1)
    return (logq_agg - logp).mean()


def instance_kl(mu, logvar):
    """Standard per-response KL(q(z|x,y) || N(0,I)), used only for monitoring (paper's 'KL' metric)."""
    return (0.5 * (mu ** 2 + logvar.exp() - logvar - 1)).sum(-1).mean()


def sample_top_p(logits, temperature, top_p):
    if temperature <= 0:
        return logits.argmax(-1)
    probs = torch.softmax(logits / temperature, dim=-1)
    sp, si = torch.sort(probs, descending=True, dim=-1)
    cum = sp.cumsum(-1)
    sp = sp.masked_fill(cum - sp > top_p, 0.0)
    sp = sp / sp.sum(-1, keepdim=True)
    idx = torch.multinomial(sp, 1)
    return si.gather(-1, idx).squeeze(-1)


class LatentCVAE(nn.Module):
    def __init__(self, backbone, latent_dim=32, num_query_tokens=4, encoder_layers=2,
                 adapter_hidden=256, use_amp=False):
        super().__init__()
        self.backbone = backbone
        for p in self.backbone.parameters():
            p.requires_grad_(False)
        self.backbone.eval()

        cfg = backbone.config
        d = cfg.hidden_size
        self.D, self.Q = latent_dim, num_query_tokens
        self.n_layers = cfg.num_hidden_layers
        self.kv_heads = cfg.num_key_value_heads
        self.head_dim = getattr(cfg, "head_dim", None) or d // cfg.num_attention_heads
        self.use_amp = use_amp

        # --- III-A latent encoder: first `encoder_layers` backbone layers (trainable copies) ---
        ecfg = copy.deepcopy(cfg)
        ecfg.num_hidden_layers = encoder_layers
        ecfg.vocab_size = 8                  # embeddings are shared with the backbone, so keep this tiny
        ecfg.pad_token_id = None
        if getattr(ecfg, "layer_types", None) is not None:
            ecfg.layer_types = ecfg.layer_types[:encoder_layers]
        enc = LlamaModel(ecfg)
        for i in range(encoder_layers):
            enc.layers[i].load_state_dict(backbone.model.layers[i].state_dict())
        enc.norm.load_state_dict(backbone.model.norm.state_dict())
        enc.embed_tokens.weight.requires_grad_(False)
        self.encoder = enc.float()
        self.query = nn.Parameter(torch.randn(num_query_tokens, d) * 0.02)       # N(0, 0.02) as in the paper
        self.post_head = nn.Sequential(nn.Linear(num_query_tokens * d, 512), nn.GELU(), nn.Linear(512, 2 * latent_dim))

        # --- III-B latent adapter: one small MLP per layer ---
        self.adapter = nn.ModuleList([
            nn.Sequential(nn.Linear(latent_dim, adapter_hidden), nn.GELU(), nn.Linear(adapter_hidden, d))
            for _ in range(self.n_layers)
        ])
        # --- III-D G-Skip feed-forward network ---
        self.skip = nn.Sequential(nn.Linear(latent_dim, d), nn.GELU(), nn.Linear(d, d))

    # ------------------------------------------------------------------ helpers
    def train(self, mode=True):
        super().train(mode)
        self.backbone.eval()
        return self

    def amp(self):
        return torch.autocast(device_type=self.query.device.type, dtype=torch.float16, enabled=self.use_amp)

    def trainable_parameters(self):
        return [p for p in self.parameters() if p.requires_grad]

    def trainable_state(self):
        return {k: v.detach().cpu() for k, v in self.state_dict().items()
                if not k.startswith("backbone.") and not k.startswith("encoder.embed_tokens")}

    def load_trainable(self, state):
        missing, unexpected = self.load_state_dict(state, strict=False)
        bad = [k for k in missing if not k.startswith("backbone.") and not k.startswith("encoder.embed_tokens")]
        if bad or unexpected:
            raise RuntimeError(f"Checkpoint mismatch. missing={bad[:5]} unexpected={list(unexpected)[:5]}")

    # ------------------------------------------------------------------ III-A
    def encode(self, ids, mask):
        """ids, mask: (B, L), LEFT-padded so the latent query always sits at the end. Returns mu, logvar (B, D)."""
        B = ids.size(0)
        with self.amp():
            emb = self.backbone.get_input_embeddings()(ids).float()
            q = self.query.unsqueeze(0).expand(B, -1, -1)
            x = torch.cat([emb, q], dim=1)
            m = torch.cat([mask, torch.ones(B, self.Q, dtype=mask.dtype, device=mask.device)], dim=1)
            pos = (m.cumsum(-1) - 1).clamp(min=0)
            feats = self.encoder(inputs_embeds=x, attention_mask=m, position_ids=pos, use_cache=False).last_hidden_state
            feats = feats[:, -self.Q:].reshape(B, -1)
            mu, logvar = self.post_head(feats).float().chunk(2, dim=-1)
        return mu, logvar.clamp(-8.0, 4.0)

    # ------------------------------------------------------------------ III-B
    def build_cache(self, z):
        """Latent-aware key/value caches, one extra position per layer (sits at position 0)."""
        B = z.size(0)
        cache = DynamicCache()
        for l, layer in enumerate(self.backbone.model.layers):
            h = self.adapter[l](z)
            k = layer.self_attn.k_proj(h).view(B, 1, self.kv_heads, self.head_dim).transpose(1, 2)
            v = layer.self_attn.v_proj(h).view(B, 1, self.kv_heads, self.head_dim).transpose(1, 2)
            cache.update(k, v, l)
        return cache

    def decode_hidden(self, ids, mask, z):
        B = ids.size(0)
        cache = self.build_cache(z)
        attn = torch.cat([torch.ones(B, 1, dtype=mask.dtype, device=mask.device), mask], dim=1)
        return self.backbone.model(input_ids=ids, attention_mask=attn, past_key_values=cache,
                                   use_cache=True).last_hidden_state

    # ------------------------------------------------------------------ III-C / III-D
    def compute_loss(self, enc_ids, enc_mask, dec_ids, dec_mask, labels, beta=1.0):
        """One GROUP = the M responses written for one instruction x (rows of the batch)."""
        B = dec_ids.size(0)
        mu, logvar = self.encode(enc_ids, enc_mask)
        z = mu + torch.randn_like(mu) * torch.exp(0.5 * logvar)
        with self.amp():
            hidden = self.decode_hidden(dec_ids, dec_mask, z)
            tgt = labels[:, 1:]
            sel = tgt != -100
            h_sel = hidden[:, :-1][sel]
            bidx = torch.arange(B, device=dec_ids.device).unsqueeze(1).expand_as(tgt)[sel]
            logits = self.backbone.lm_head(h_sel)
            s_latent = self.backbone.lm_head(self.skip(z))              # (B, V)
            g_skip = s_latent - s_latent.detach()                       # == 0 in value, keeps the gradient
            logits = logits + g_skip[bidx]
        nll = F.cross_entropy(logits.float(), tgt[sel], reduction="none")
        seq_nll = torch.zeros(B, device=nll.device).index_add_(0, bidx, nll)
        recon = seq_nll.mean()                                           # nats per response
        kl_g = group_kl(mu, logvar, z)
        kl_i = instance_kl(mu, logvar)
        n_tok = sel.sum().float() / B
        loss = (recon + beta * kl_g) / n_tok                              # ELBo scaled per token
        return {"loss": loss, "recon_per_token": (recon / n_tok).detach(), "kl_group": kl_g.detach(),
                "kl_instance": kl_i.detach(), "mu": mu.detach()}

    # ------------------------------------------------------------------ generation
    @torch.no_grad()
    def generate(self, prompt_ids, num_samples, z=None, use_latent=True, max_new_tokens=160,
                 temperature=1.0, top_p=0.95, eos_ids=(), pad_id=0):
        """Sample `num_samples` answers for one prompt (list of token ids).
        use_latent=True : each answer uses its own z ~ N(0, I) (prior-guided generation).
        use_latent=False: plain backbone with the same sampler (the RAG baseline)."""
        dev = self.query.device
        B = num_samples
        ids = torch.tensor([prompt_ids], device=dev).expand(B, -1).contiguous()
        eos = torch.tensor(list(eos_ids), device=dev)
        with self.amp():
            if use_latent:
                z = torch.randn(B, self.D, device=dev) if z is None else z
                cache, offset = self.build_cache(z), 1
            else:
                cache, offset = DynamicCache(), 0
            attn = torch.ones(B, offset + ids.size(1), dtype=torch.long, device=dev)
            out = self.backbone.model(input_ids=ids, attention_mask=attn, past_key_values=cache, use_cache=True)
            cache, h = out.past_key_values, out.last_hidden_state[:, -1]
            gen = []
            finished = torch.zeros(B, dtype=torch.bool, device=dev)
            for _ in range(max_new_tokens):
                nxt = sample_top_p(self.backbone.lm_head(h).float(), temperature, top_p)
                nxt = torch.where(finished, torch.full_like(nxt, pad_id), nxt)
                gen.append(nxt)
                finished = finished | torch.isin(nxt, eos)
                if bool(finished.all()):
                    break
                attn = torch.cat([attn, torch.ones(B, 1, dtype=torch.long, device=dev)], dim=1)
                out = self.backbone.model(input_ids=nxt[:, None], attention_mask=attn,
                                          past_key_values=cache, use_cache=True)
                cache, h = out.past_key_values, out.last_hidden_state[:, -1]
        return torch.stack(gen, dim=1)
