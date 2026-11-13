"""
ManualMind evaluation harness.

Place next to rag.py and eval_set.json, then run:

    python evaluate.py --no-generate          # retrieval only (fast, no Ollama)
    python evaluate.py                        # retrieval + generation
    python evaluate.py --judge                # + LLM-as-judge faithfulness
    python evaluate.py --ks 1 3 4 5 8 --gen-k 4

Metrics
-------
Retrieval (answerable questions only; relevance = chunk's page is in
"relevant_pages", optionally also matching "document"):
    Hit@k, Precision@k, Recall@k (page coverage), MRR@k, nDCG@k

Generation:
    refusal_correct      refused when unanswerable / answered when answerable
    false_refusal_rate   refused although answerable
    hallucination_rate   answered although unanswerable
    fact_coverage        fraction of "must_include" strings found in answer
    citation_present     non-refusal answer contains at least one [Sn]
    citation_valid       every [Sn] cited is within 1..gen_k
    citation_precision   fraction of cited sources that sit on relevant pages
    faithful (--judge)   judge says every claim is supported by the passages
    latency_s            end-to-end generation time
"""

import argparse
import csv
import hashlib
import json
import math
import re
import statistics
import time
from pathlib import Path

import chromadb
import requests
from sentence_transformers import SentenceTransformer

from rag import LLM_MODEL, OLLAMA_URL, generate_answer

PROJECT_FOLDER = Path(__file__).resolve().parent
INPUT_FILE = PROJECT_FOLDER / "output" / "embedded_chunks.json"
DATABASE_FOLDER = PROJECT_FOLDER / "chroma_db"
EVAL_FILE = PROJECT_FOLDER / "eval_set.json"
RESULTS_FOLDER = PROJECT_FOLDER / "eval_results"

REFUSAL_PHRASE = "could not find enough information"


# ---------------------------------------------------------------- setup
def load_resources():
    if not INPUT_FILE.exists() or not DATABASE_FOLDER.exists():
        raise SystemExit("Run embed.py and vector_store.py first.")

    file_bytes = INPUT_FILE.read_bytes()
    data = json.loads(file_bytes)
    name = f"manualmind_{hashlib.sha256(file_bytes).hexdigest()[:16]}"

    client = chromadb.PersistentClient(path=str(DATABASE_FOLDER))
    if name not in [c.name for c in client.list_collections()]:
        raise SystemExit("Collection missing. Run vector_store.py again.")

    collection = client.get_collection(name=name, embedding_function=None)
    model = SentenceTransformer(data["model"], device="cpu")
    return data, collection, model


def retrieve(collection, model, normalized, question, n):
    emb = model.encode(
        question, normalize_embeddings=normalized, convert_to_numpy=True
    )
    res = collection.query(
        query_embeddings=[emb.tolist()],
        n_results=min(n, collection.count()),
        include=["documents", "metadatas", "distances"],
    )
    return res["documents"][0], res["metadatas"][0], res["distances"][0]


def build_context(passages, metas):
    """Identical to the context format used in rag.py."""
    parts = []
    for i, (p, m) in enumerate(zip(passages, metas), start=1):
        parts.append(
            f"[S{i}] Manual: {m['document']}; PDF page: {m['page']}\n{p}"
        )
    return "\n\n".join(parts)


# ------------------------------------------------------ retrieval metrics
def is_relevant(meta, item):
    if meta["page"] not in set(item.get("relevant_pages", [])):
        return False
    doc = item.get("document")
    return doc is None or meta["document"] == doc


def retrieval_metrics(metas, item, k):
    top = metas[:k]
    flags = [is_relevant(m, item) for m in top]
    relevant_pages = set(item["relevant_pages"])

    found_pages = {m["page"] for m, f in zip(top, flags) if f}
    first = next((i for i, f in enumerate(flags) if f), None)

    dcg = sum(1 / math.log2(i + 2) for i, f in enumerate(flags) if f)
    # Ideal: every returned slot relevant (many chunks exist per page).
    idcg = sum(1 / math.log2(i + 2) for i in range(len(top)))

    return {
        f"hit@{k}": float(any(flags)),
        f"precision@{k}": sum(flags) / k,
        f"recall@{k}": len(found_pages) / len(relevant_pages),
        f"mrr@{k}": 0.0 if first is None else 1 / (first + 1),
        f"ndcg@{k}": dcg / idcg if idcg else 0.0,
    }


# ----------------------------------------------------- generation metrics
def norm(s):
    return re.sub(r"\s+", " ", s.lower()).strip()


def is_refusal(answer):
    return REFUSAL_PHRASE in norm(answer)


def cited_sources(answer):
    cited = set()
    for group in re.findall(r"\[([^\]]*)\]", answer):
        cited.update(int(n) for n in re.findall(r"S(\d+)", group))
    return cited


def judge_faithfulness(question, context, answer):
    prompt = (
        "You are a strict evaluator. Decide whether EVERY factual claim in "
        "the answer is supported by the passages.\n\n"
        f"Passages:\n{context}\n\nQuestion:\n{question}\n\nAnswer:\n{answer}\n\n"
        'Reply with JSON only: {"faithful": true or false, '
        '"unsupported_claims": ["..."]}'
    )
    r = requests.post(
        OLLAMA_URL,
        json={
            "model": LLM_MODEL,
            "messages": [{"role": "user", "content": prompt}],
            "stream": False,
            "format": "json",
            "options": {"temperature": 0, "num_ctx": 4096, "num_predict": 250},
        },
        timeout=(10, 300),
    )
    r.raise_for_status()
    try:
        out = json.loads(r.json()["message"]["content"])
        return float(bool(out.get("faithful"))), out.get("unsupported_claims", [])
    except (json.JSONDecodeError, KeyError):
        return None, ["judge returned invalid JSON"]


def generation_metrics(item, answer, metas, gen_k, faithful):
    answerable = item.get("answerable", True)
    refused = is_refusal(answer)
    m = {"refused": float(refused)}

    m["refusal_correct"] = float(refused != answerable)
    if answerable:
        m["false_refusal"] = float(refused)
    else:
        m["hallucination"] = float(not refused)

    if answerable and item.get("must_include"):
        kws = item["must_include"]
        m["fact_coverage"] = sum(norm(k) in norm(answer) for k in kws) / len(kws)

    if not refused:
        cited = cited_sources(answer)
        m["citation_present"] = float(bool(cited))
        if cited:
            m["citation_valid"] = float(all(1 <= c <= gen_k for c in cited))
            valid = [c for c in cited if 1 <= c <= len(metas)]
            if answerable and valid:
                m["citation_precision"] = sum(
                    is_relevant(metas[c - 1], item) for c in valid
                ) / len(valid)

    if faithful is not None:
        m["faithful"] = faithful
    return m


# ------------------------------------------------------------ aggregation
def aggregate(rows):
    keys = {k for r in rows for k, v in r.items() if isinstance(v, (int, float))}
    return {
        k: round(statistics.mean(r[k] for r in rows if r.get(k) is not None), 4)
        for k in sorted(keys)
        if any(r.get(k) is not None for r in rows)
    }


def print_table(title, metrics):
    print(f"\n=== {title} ===")
    for k, v in metrics.items():
        print(f"{k:<24}{v:>8.4f}")


# ------------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ks", type=int, nargs="+", default=[1, 3, 4, 5, 8])
    ap.add_argument("--gen-k", type=int, default=4, help="passages sent to LLM")
    ap.add_argument("--no-generate", action="store_true")
    ap.add_argument("--judge", action="store_true")
    ap.add_argument("--eval-file", default=str(EVAL_FILE))
    args = ap.parse_args()

    eval_set = json.loads(Path(args.eval_file).read_text(encoding="utf-8"))
    data, collection, model = load_resources()

    max_k = max(args.ks + [args.gen_k])
    retrieval_rows, generation_rows, per_question = [], [], []

    for n, item in enumerate(eval_set, start=1):
        q = item["question"]
        print(f"[{n}/{len(eval_set)}] {q[:70]}")
        passages, metas, dists = retrieve(
            collection, model, data["normalized"], q, max_k
        )
        row = {
            "id": item.get("id", n),
            "question": q,
            "answerable": item.get("answerable", True),
            "top1_distance": round(dists[0], 4),
        }

        if row["answerable"]:
            r = {}
            for k in args.ks:
                r.update(retrieval_metrics(metas, item, k))
            retrieval_rows.append(r)
            row.update(r)
            row["retrieved_pages"] = [m["page"] for m in metas[: args.gen_k]]
            row["relevant_pages"] = item["relevant_pages"]

        if not args.no_generate:
            gen_passages = passages[: args.gen_k]
            gen_metas = metas[: args.gen_k]
            context = build_context(gen_passages, gen_metas)
            try:
                t0 = time.time()
                answer = generate_answer(q, context)
                latency = time.time() - t0
            except requests.exceptions.RequestException as e:
                print(f"  Ollama error: {e}")
                continue

            faithful, unsupported = None, []
            if args.judge and not is_refusal(answer):
                faithful, unsupported = judge_faithfulness(q, context, answer)

            g = generation_metrics(item, answer, gen_metas, args.gen_k, faithful)
            g["latency_s"] = latency
            generation_rows.append(g)
            row.update(g)
            row["answer"] = answer
            if unsupported:
                row["unsupported_claims"] = " | ".join(map(str, unsupported))

        per_question.append(row)

    # ---- report
    report = {
        "llm": LLM_MODEL,
        "embedding_model": data["model"],
        "total_chunks": collection.count(),
        "gen_k": args.gen_k,
        "questions": len(eval_set),
        "answerable": sum(i.get("answerable", True) for i in eval_set),
    }
    if retrieval_rows:
        report["retrieval"] = aggregate(retrieval_rows)
        print_table("RETRIEVAL", report["retrieval"])
    if generation_rows:
        report["generation"] = aggregate(generation_rows)
        print_table("GENERATION", report["generation"])

    RESULTS_FOLDER.mkdir(exist_ok=True)
    (RESULTS_FOLDER / "report.json").write_text(
        json.dumps(report, indent=2), encoding="utf-8"
    )
    fields = sorted({k for r in per_question for k in r})
    with open(RESULTS_FOLDER / "per_question.csv", "w", newline="",
              encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(per_question)

    print(f"\nSaved to {RESULTS_FOLDER}")


if __name__ == "__main__":
    main()