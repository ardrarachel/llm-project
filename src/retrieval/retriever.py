"""
Chunking + dense retrieval + LegalBench-RAG style character-level precision/recall.
The embedding function is injected, so tests can run without downloading a model.
"""
from dataclasses import dataclass

import numpy as np

BGE_QUERY_PREFIX = "Represent this sentence for searching relevant passages: "


@dataclass
class Chunk:
    doc: str
    start: int
    end: int
    text: str


def chunk_text(text, doc, chunk_size=512, overlap=50):
    chunks, start = [], 0
    while start < len(text):
        end = min(start + chunk_size, len(text))
        chunks.append(Chunk(doc, start, end, text[start:end]))
        if end == len(text):
            break
        start += chunk_size - overlap
    return chunks


def build_chunks(docs, chunk_size=512, overlap=50):
    chunks = []
    for doc, text in docs.items():
        chunks.extend(chunk_text(text, doc, chunk_size, overlap))
    return chunks


def make_st_embedder(model_name, device=None, batch_size=64):
    """Returns embed(texts, is_query) -> L2-normalised float32 array (sentence-transformers)."""
    from sentence_transformers import SentenceTransformer

    model = SentenceTransformer(model_name, device=device)

    def embed(texts, is_query=False):
        if is_query and "bge" in model_name.lower():
            texts = [BGE_QUERY_PREFIX + t for t in texts]
        vecs = model.encode(texts, batch_size=batch_size, show_progress_bar=len(texts) > 500,
                            convert_to_numpy=True, normalize_embeddings=True)
        return vecs.astype("float32")

    return embed


class Retriever:
    def __init__(self, chunks, embed_fn):
        self.chunks = chunks
        self.embed_fn = embed_fn
        vecs = embed_fn([c.text for c in chunks], is_query=False)
        self.vecs = np.ascontiguousarray(vecs.astype("float32"))
        try:
            import faiss

            self.index = faiss.IndexFlatIP(self.vecs.shape[1])
            self.index.add(self.vecs)
            self._faiss = True
        except ImportError:
            self.index = None
            self._faiss = False

    def search(self, queries, k=3):
        q = np.ascontiguousarray(self.embed_fn(queries, is_query=True).astype("float32"))
        if self._faiss:
            scores, ids = self.index.search(q, k)
        else:
            sims = q @ self.vecs.T
            ids = np.argsort(-sims, axis=1)[:, :k]
            scores = np.take_along_axis(sims, ids, axis=1)
        return [[(self.chunks[j], float(s)) for j, s in zip(row_ids, row_scores)]
                for row_ids, row_scores in zip(ids, scores)]


def _merge(intervals):
    intervals = sorted(intervals)
    merged = []
    for s, e in intervals:
        if merged and s <= merged[-1][1]:
            merged[-1][1] = max(merged[-1][1], e)
        else:
            merged.append([s, e])
    return merged


def _overlap(a, b):
    """Total overlapping length between two lists of merged intervals."""
    i = j = total = 0
    while i < len(a) and j < len(b):
        lo, hi = max(a[i][0], b[j][0]), min(a[i][1], b[j][1])
        if hi > lo:
            total += hi - lo
        if a[i][1] < b[j][1]:
            i += 1
        else:
            j += 1
    return total


def retrieval_metrics(queries, results):
    """Character-level precision/recall against the gold spans (as in LegalBench-RAG)."""
    precs, recs, hits, doc_hits = [], [], 0, 0
    for q, res in zip(queries, results):
        gold = {}
        for g in q["gold"]:
            gold.setdefault(g["doc"], []).append((g["start"], g["end"]))
        ret = {}
        for chunk, _ in res:
            ret.setdefault(chunk.doc, []).append((chunk.start, chunk.end))
        gold_len = sum(e - s for ivs in gold.values() for s, e in _merge(list(ivs)))
        ret_len = sum(e - s for ivs in ret.values() for s, e in _merge(list(ivs)))
        ov = 0
        for doc, ivs in ret.items():
            if doc in gold:
                ov += _overlap(_merge(list(ivs)), _merge(list(gold[doc])))
        precs.append(ov / ret_len if ret_len else 0.0)
        recs.append(ov / gold_len if gold_len else 0.0)
        hits += int(ov > 0)
        doc_hits += int(any(d in gold for d in ret))
    n = max(len(queries), 1)
    return {
        "precision": float(np.mean(precs)) if precs else 0.0,
        "recall": float(np.mean(recs)) if recs else 0.0,
        "span_hit_rate": hits / n,
        "doc_hit_rate": doc_hits / n,
        "num_queries": len(queries),
    }
