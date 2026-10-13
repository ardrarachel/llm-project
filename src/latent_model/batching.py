import torch

from ..prompts import build_prompt_ids


def make_group(tok, item, max_prompt_tokens, max_response_tokens, end_id, pad_id, device):
    """Turn one question (with its M responses) into one training group.

    Encoder input : prompt + response, LEFT-padded (the latent query is appended at the end by the model)
    Decoder input : prompt + response, RIGHT-padded; labels are -100 on the prompt and on padding.
    The prompt contains NO style hint, so z has to carry the variation between the M responses.
    """
    prompt = build_prompt_ids(tok, item["chunks"], item["query"], max_prompt_tokens)
    seqs, resp_lens = [], []
    for r in item["responses"]:
        rid = tok(r, add_special_tokens=False).input_ids[: max_response_tokens - 1] + [end_id]
        seqs.append(prompt + rid)
        resp_lens.append(len(rid))
    L = max(len(s) for s in seqs)
    M = len(seqs)

    enc_ids = torch.full((M, L), pad_id, dtype=torch.long)
    enc_mask = torch.zeros((M, L), dtype=torch.long)
    dec_ids = torch.full((M, L), pad_id, dtype=torch.long)
    dec_mask = torch.zeros((M, L), dtype=torch.long)
    labels = torch.full((M, L), -100, dtype=torch.long)
    for i, (s, rl) in enumerate(zip(seqs, resp_lens)):
        n = len(s)
        enc_ids[i, L - n:] = torch.tensor(s)
        enc_mask[i, L - n:] = 1
        dec_ids[i, :n] = torch.tensor(s)
        dec_mask[i, :n] = 1
        labels[i, n - rl:n] = torch.tensor(s[n - rl:])
    return [t.to(device) for t in (enc_ids, enc_mask, dec_ids, dec_mask, labels)]
