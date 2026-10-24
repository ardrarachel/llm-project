# Diverse grounded legal RAG with a latent-variable module (Phase 1)

Backbone: `meta-llama/Llama-3.2-1B-Instruct` (frozen). Data: LegalBench-RAG, CUAD subset.
Method: Zhang et al., "Latent Variable Modeling for Controllable and Diverse Generation from LLMs" (IEEE TAI 2026),
scaled down to one Kaggle T4.

## Run order (Kaggle notebook: notebooks/phase1_kaggle.ipynb)
| Step | Command | What it does | Output |
|---|---|---|---|
| 1 | `python notebooks/00_download_data.py` | download LegalBench-RAG | data/raw/ |
| 2 | `python scripts/01_prepare_data.py` | split by contract, chunk, FAISS, retrieve, retrieval precision/recall | data/processed/{train,val,test}.json |
| 3 | `python scripts/02_generate_training_data.py` | Llama writes 4 styled answers per question | data/processed/*_responses.jsonl |
| 4 | `python scripts/03_train.py` (`--resume`) | train latent encoder + adapter + G-Skip with DG-ELBo | outputs/phase1_latent.pt |
| 5 | `python scripts/04_evaluate.py` | latent sampling vs plain RAG on test contracts | outputs/eval_summary.json, eval_samples.json |
| 6 | `python scripts/05_diagnose_latent.py` | is z actually used? NLL under different z, style probe, spread of mu | printed report |

Tests (CPU, no downloads): `python tests/test_data_pipeline.py`, `python tests/test_latent_model.py`, `python tests/smoke_end_to_end.py`.

## What follows the paper
- Latent encoder: embedding + first 2 backbone layers (trainable copies), 4-token latent query appended at the END of (instruction, response), 2-layer MLP -> mu, sigma^2, Dim 32.
- Latent adapter: one MLP per layer -> hidden state -> backbone W_K / W_V -> one extra key/value per layer, concatenated to the KV cache.
- G-Skip: S_latent = lm_head(FFN(z)); S_latent - stopgrad(S_latent) is added to the logits (value 0, gradient flows).
- DG-ELBo: reconstruction of the group of M responses + group-level KL (aggregated posterior vs N(0,I)), Monte-Carlo estimate.
- Backbone frozen; prior N(0,I); at inference z ~ N(0,I).
- Training data is sampled from the backbone itself (self-supervised).

## Where we differ from the paper (state these in the viva)
- M = 4 responses per question (paper: 32) and ~300 questions, so the group-KL estimate is noisier and training is much smaller.
- The 4 responses come from 4 style hints (plain / technical / concise / cautious) used ONLY when writing the data; the training prompt has no hint.
- KL warm-up (beta 0 -> 1 over 50 steps) added as an extra anti-collapse safeguard.
- Llama-3.2-1B instead of Llama-3-8B; encoder shares the frozen embedding table instead of copying it.
- Adapter hidden size 256 and the MLP head sizes are our choices (paper does not fix them in the text we used).

## Reading the training log (posterior collapse)
- KL(group) is almost equal to KL(inst), and `active_units` is 0: the posteriors of one question's responses are identical, so z carries no response-specific information.
  The group-level KL can save at most ln(M) nats over an ordinary KL (M=4: 1.4, M=8: 2.1, paper M=32), so a small M behaves almost like a plain VAE.
- Run `scripts/05_diagnose_latent.py`. Then try `configs/config_m8.yaml` (M=8, kl_weight 0.5, longer warm-up, 15 epochs), after `scripts/00_memory_probe.py --config configs/config_m8.yaml`.

## After a collapsed run (KL ~ 1-2, active_units 0, latent answers = baseline answers)
1. `python scripts/05_diagnose_latent.py` (records the evidence: NLL with / without z, style probe).
2. Quick retrain on the SAME data, about an hour on a T4 (my estimate): `python scripts/03_train.py --config configs/config_retrain.yaml`,
   then `04_evaluate.py` and `05_diagnose_latent.py` with the same `--config`. New knobs: `training.latent_lr_mult`
   (higher learning rate for adapter + G-Skip) and `model.adapter_norm` (unit-RMS latent key/value, our addition).
   v1 of this config diverged on Kaggle (KL 300 -> 33,000, then NaN). v2 lowers `lr` to 2e-5, adds `model.feature_norm` (LayerNorm on the encoder
   features), starts the latent key/value small (`adapter_gain_init: 0.1`), and stops by itself if KL(inst) stays above `training.kl_abort` (500).
3. If still collapsed: more data per question, `configs/config_m8.yaml` (M=8; regenerates the data, several hours).

## Troubleshooting
- **CUDA out of memory** in step 3/4: lower `model.max_prompt_tokens` (e.g. 480), `retrieval.top_k` to 2, or use fewer styles (M=3). Rerun steps 2-4.
- **KL(inst) stays near 0, active_units = 0, recon/token same as baseline** = posterior collapse. Try `kl_weight: 0.5`, longer `kl_warmup_steps` (200), more epochs, more training questions.
- **Low retrieval recall**: smaller `chunk_size` (256) or higher `top_k`; rerun step 2 onward.
- **Llama download 401/403**: HF_TOKEN secret missing or the Llama-3.2-1B-Instruct access request is not approved.
- **Dropbox download fails**: download the zip in the browser, upload as a Kaggle Dataset, attach it, skip step 1.
