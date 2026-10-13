"""
Loading LegalBench-RAG (CUAD subset).

Expected layout (found automatically, anywhere under the candidate roots):
    .../corpus/cuad/<contract>.txt
    .../benchmarks/cuad*.json      -> {"tests": [{"query": str,
                                                   "snippets": [{"file_path": str, "span": [start, end]}]}]}
The loader is tolerant about folder names, so a slightly different Dropbox layout still works.
"""
import os
import random
from collections import defaultdict

from .utils import load_json


def locate_dataset(candidates, dataset="cuad"):
    corpus_root = None
    bench_files = []
    for root in candidates:
        if not os.path.isdir(root):
            continue
        for dirpath, dirnames, filenames in os.walk(root):
            if os.path.basename(dirpath).lower() == "corpus" and corpus_root is None:
                corpus_root = dirpath
            for fn in filenames:
                low = fn.lower()
                if low.endswith(".json") and dataset in low and "benchmark" in dirpath.lower():
                    bench_files.append(os.path.join(dirpath, fn))
    if corpus_root is None or not bench_files:
        raise FileNotFoundError(
            f"Could not find the dataset. corpus_root={corpus_root}, benchmark files={bench_files}. "
            f"Searched: {candidates}. Run notebooks/00_download_data.py or upload the data as a Kaggle dataset."
        )
    return corpus_root, bench_files


def load_corpus(corpus_root, dataset="cuad"):
    """Return {relative_path: text}. Only the dataset's sub-folder is read if it exists."""
    sub = os.path.join(corpus_root, dataset)
    walk_root = sub if os.path.isdir(sub) else corpus_root
    docs = {}
    for dirpath, _, filenames in os.walk(walk_root):
        for fn in filenames:
            if fn.lower().endswith(".txt"):
                full = os.path.join(dirpath, fn)
                rel = os.path.relpath(full, corpus_root).replace(os.sep, "/")
                with open(full, encoding="utf-8", errors="ignore") as f:
                    docs[rel] = f.read()
    if not docs:
        raise FileNotFoundError(f"No .txt files found under {walk_root}")
    return docs


def _norm(path):
    return path.replace("\\", "/").lstrip("./").lower()


def make_resolver(docs):
    exact = {_norm(k): k for k in docs}
    by_base = defaultdict(list)
    for k in docs:
        by_base[os.path.basename(k).lower()].append(k)

    def resolve(file_path):
        n = _norm(file_path)
        if n in exact:
            return exact[n]
        # suffix match (e.g. benchmark path "cuad/x.txt" vs corpus key "cuad/x.txt" in a deeper folder)
        for k_norm, k in exact.items():
            if k_norm.endswith(n) or n.endswith(k_norm):
                return k
        cands = by_base.get(os.path.basename(n), [])
        return cands[0] if cands else None

    return resolve


def load_tests(bench_files):
    """Use the benchmark file with the most test cases."""
    best, best_n = None, -1
    for p in bench_files:
        obj = load_json(p)
        tests = obj["tests"] if isinstance(obj, dict) and "tests" in obj else obj
        if len(tests) > best_n:
            best, best_n = (p, tests), len(tests)
    print(f"Using benchmark file {best[0]} ({best_n} test cases)")
    return best[1]


def build_queries(tests, docs):
    """Normalise test cases into {qid, query, gold:[{doc,start,end,answer}]} with resolved doc keys."""
    resolve = make_resolver(docs)
    out, dropped = [], 0
    for i, t in enumerate(tests):
        gold = []
        for s in t.get("snippets", []):
            key = resolve(s["file_path"])
            if key is None:
                continue
            start, end = s["span"]
            gold.append({"doc": key, "start": int(start), "end": int(end), "answer": s.get("answer", ""),
                         "text": docs[key][int(start):int(end)][:3000]})
        if not gold:
            dropped += 1
            continue
        out.append({"qid": f"q{i}", "query": t["query"], "gold": gold})
    print(f"Queries kept: {len(out)} (dropped {dropped} whose documents were not found)")
    return out


def split_queries(queries, caps, seed=42, ratios=(0.8, 0.1, 0.1)):
    """Split by DOCUMENT so no contract appears in more than one split."""
    rng = random.Random(seed)
    by_doc = defaultdict(list)
    for q in queries:
        by_doc[q["gold"][0]["doc"]].append(q)
    doc_list = sorted(by_doc)
    rng.shuffle(doc_list)
    n = len(doc_list)
    n_train = max(1, int(ratios[0] * n))
    n_val = max(1, int(ratios[1] * n))
    groups = {
        "train": doc_list[:n_train],
        "val": doc_list[n_train:n_train + n_val],
        "test": doc_list[n_train + n_val:],
    }
    splits = {}
    for name, doc_keys in groups.items():
        qs = [q for d in doc_keys for q in by_doc[d]]
        rng.shuffle(qs)
        splits[name] = qs[: caps[name]]
    for name, qs in splits.items():
        print(f"  {name}: {len(qs)} queries from {len(set(q['gold'][0]['doc'] for q in qs))} contracts")
    return splits
