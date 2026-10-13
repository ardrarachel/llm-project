"""
Step 1. Load LegalBench-RAG (CUAD), split by contract, build the FAISS index, retrieve the
top-k chunks for every query and measure retrieval quality against the gold spans.

Outputs (data/processed/):
    train.json, val.json, test.json   each item: qid, query, gold, retrieved[{doc,start,end,text,score}]
    retrieval_metrics.json
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.data import build_queries, load_corpus, load_tests, locate_dataset, split_queries
from src.retrieval.retriever import Retriever, build_chunks, make_st_embedder, retrieval_metrics
from src.utils import get_device, load_config, save_json, set_seed


def main(config_path="configs/config.yaml", embed_fn=None):
    cfg = load_config(config_path)
    set_seed(cfg["seed"])
    dcfg, rcfg = cfg["data"], cfg["retrieval"]
    out_dir = cfg["paths"]["processed_dir"]

    corpus_root, bench_files = locate_dataset(cfg["paths"]["data_root_candidates"], dcfg["dataset_name"])
    print("Corpus folder:", corpus_root)
    docs = load_corpus(corpus_root, dcfg["dataset_name"])
    print(f"Loaded {len(docs)} contracts, {sum(len(t) for t in docs.values()) / 1e6:.1f}M characters")

    tests = load_tests(bench_files)
    queries = build_queries(tests, docs)
    caps = {"train": dcfg["max_train_queries"], "val": dcfg["max_val_queries"], "test": dcfg["max_test_queries"]}
    splits = split_queries(queries, caps, seed=cfg["seed"])

    if dcfg["restrict_corpus_to_query_docs"]:
        used = {g["doc"] for qs in splits.values() for q in qs for g in q["gold"]}
        docs = {k: v for k, v in docs.items() if k in used}
        print(f"Index restricted to {len(docs)} contracts used by the selected queries")

    chunks = build_chunks(docs, rcfg["chunk_size"], rcfg["chunk_overlap"])
    print(f"{len(chunks)} chunks (size {rcfg['chunk_size']}, overlap {rcfg['chunk_overlap']})")

    if embed_fn is None:
        embed_fn = make_st_embedder(rcfg["embedding_model"], device=str(get_device()))
    retriever = Retriever(chunks, embed_fn)

    all_metrics = {}
    for name, qs in splits.items():
        results = retriever.search([q["query"] for q in qs], k=rcfg["top_k"])
        all_metrics[name] = retrieval_metrics(qs, results)
        items = []
        for q, res in zip(qs, results):
            items.append({
                "qid": q["qid"], "query": q["query"], "gold": q["gold"],
                "retrieved": [{"doc": c.doc, "start": c.start, "end": c.end, "text": c.text, "score": s}
                              for c, s in res],
            })
        save_json(items, os.path.join(out_dir, f"{name}.json"))
        m = all_metrics[name]
        print(f"[{name}] precision={m['precision']:.3f} recall={m['recall']:.3f} "
              f"span_hit={m['span_hit_rate']:.2f} doc_hit={m['doc_hit_rate']:.2f}")
    save_json(all_metrics, os.path.join(out_dir, "retrieval_metrics.json"))
    print("Saved to", out_dir)


if __name__ == "__main__":
    main()
