"""Run: python tests/test_data_pipeline.py   (no model downloads: synthetic contracts + hashing embedder)"""
import os, sys, json, tempfile, zlib, re
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np
from src.data import locate_dataset, load_corpus, load_tests, build_queries, split_queries
from src.retrieval.retriever import build_chunks, Retriever, retrieval_metrics


def hash_embed(texts, is_query=False, dim=4096):
    out = np.zeros((len(texts), dim), dtype="float32")
    for i, t in enumerate(texts):
        for w in re.findall(r'\w+', t.lower()):
            out[i, zlib.crc32(w.encode()) % dim] += 1.0
    out /= np.maximum(np.linalg.norm(out, axis=1, keepdims=True), 1e-9)
    return out


def make_fake_dataset(root, n_docs=20):
    os.makedirs(os.path.join(root, "corpus", "cuad"))
    os.makedirs(os.path.join(root, "benchmarks"))
    tests = []
    for d in range(n_docs):
        filler = " ".join(f"standard boilerplate sentence number {i} about generic terms." for i in range(60))
        key_sentence = f"The governing law of agreement zq{d}x is the state of zorbland{d}k and no other jurisdiction."
        text = filler[:900] + " " + key_sentence + " " + filler[900:1800]
        with open(os.path.join(root, "corpus", "cuad", f"contract_{d}.txt"), "w") as f:
            f.write(text)
        s = text.index(key_sentence)
        for j in range(3):
            tests.append({"query": f"What is the governing law of agreement zq{d}x? kind v{j}z",
                          "snippets": [{"file_path": f"cuad/contract_{d}.txt", "span": [s, s + len(key_sentence)], "answer": ""}]})
    with open(os.path.join(root, "benchmarks", "cuad.json"), "w") as f:
        json.dump({"tests": tests}, f)


def test_pipeline():
    with tempfile.TemporaryDirectory() as tmp:
        make_fake_dataset(tmp)
        corpus_root, bench = locate_dataset([tmp], "cuad")
        docs = load_corpus(corpus_root, "cuad")
        assert len(docs) == 20 and all(k.startswith("cuad/") for k in docs)
        queries = build_queries(load_tests(bench), docs)
        assert len(queries) == 60 and queries[0]["gold"][0]["text"].startswith("The governing law")
        splits = split_queries(queries, {"train": 30, "val": 6, "test": 6}, seed=1)
        docs_by_split = [set(q["gold"][0]["doc"] for q in qs) for qs in splits.values()]
        assert not (docs_by_split[0] & docs_by_split[1]) and not (docs_by_split[0] & docs_by_split[2])
        chunks = build_chunks(docs, 256, 30)
        retr = Retriever(chunks, hash_embed)
        qs = splits["test"]
        res = retr.search([q["query"] for q in qs], k=3)
        m = retrieval_metrics(qs, res)
        print("metrics:", m)
        assert m["span_hit_rate"] >= 0.8 and m["doc_hit_rate"] >= 0.8 and 0 < m["precision"] <= 1 and 0 < m["recall"] <= 1
    print("ok data + retrieval pipeline")


def test_metrics_edge_cases():
    from src.retrieval.retriever import Chunk
    q = [{"gold": [{"doc": "a", "start": 100, "end": 200}]}]
    perfect = [[(Chunk("a", 100, 200, ""), 1.0)]]
    assert retrieval_metrics(q, perfect)["precision"] == 1.0 and retrieval_metrics(q, perfect)["recall"] == 1.0
    wrong = [[(Chunk("b", 100, 200, ""), 1.0)]]
    assert retrieval_metrics(q, wrong)["recall"] == 0.0
    half = [[(Chunk("a", 150, 250, ""), 1.0)]]
    r = retrieval_metrics(q, half)
    assert abs(r["precision"] - 0.5) < 1e-9 and abs(r["recall"] - 0.5) < 1e-9
    print("ok retrieval metric edge cases")


class FakeTok:
    pad_token_id = 0
    def apply_chat_template(self, messages, tokenize=False, add_generation_prompt=True):
        return " ".join(m["content"] for m in messages) + " <assistant>"
    def __call__(self, text, add_special_tokens=False):
        class O: pass
        o = O(); o.input_ids = [3 + zlib.crc32(w.encode()) % 90 for w in text.split()]
        return o


def test_batching():
    import torch
    from src.latent_model.batching import make_group
    item = {"chunks": ["alpha beta gamma", "delta epsilon"], "query": "what is it", "responses": ["one two three", "four five"]}
    enc_ids, enc_mask, dec_ids, dec_mask, labels = make_group(FakeTok(), item, 200, 50, end_id=2, pad_id=0, device="cpu")
    assert enc_ids.shape == dec_ids.shape == labels.shape and enc_ids.size(0) == 2
    # encoder is left padded: last column is a real token for every row; decoder is right padded
    assert enc_mask[:, -1].all() and dec_mask[:, 0].all()
    # labels cover exactly the response tokens (+ end token), and nothing of the prompt
    assert (labels[0] != -100).sum() == 3 + 1 and (labels[1] != -100).sum() == 2 + 1
    first = (labels[0] != -100).nonzero()[0, 0]
    assert dec_ids[0, first:first + 4].tolist()[-1] == 2
    print("ok batching/labels")


if __name__ == "__main__":
    test_pipeline(); test_metrics_edge_cases(); test_batching()
    print("ALL DATA TESTS PASSED")
