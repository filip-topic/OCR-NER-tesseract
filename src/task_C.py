from __future__ import annotations
import argparse
import csv
import json
import os
import random
import re
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Tuple

import matplotlib.pyplot as plt
import numpy as np
import spacy
from sklearn.manifold import TSNE
from openai import OpenAI

#spaCy model 
DEFAULT_SPACY_MODEL = "de_core_news_md"
try:
    nlp = spacy.load(DEFAULT_SPACY_MODEL)
except OSError:
    # auto-download if missing
    from spacy.cli import download

    download(DEFAULT_SPACY_MODEL)
    nlp = spacy.load(DEFAULT_SPACY_MODEL)

# constants 
DEFAULT_OCR_JSON = "ocr_evaluation_results.json"
RESULTS_DIR = Path("results")
RESULTS_DIR.mkdir(exist_ok=True)

DEEPOINFRA_BASE_URL = "https://api.deepinfra.com/v1/openai"
DEFAULT_LLM_MODELS = ["meta-llama/Meta-Llama-3-70B-Instruct"]

# helper functions 
def pick_best_variant(record: Dict[str, Any]) -> str:
    """Return OCR text of the variant with the lowest CER."""
    best_key, best_score = None, float("inf")
    for k, v in record.items():
        cer = v.get("CER")
        if cer is not None and cer < best_score:
            best_key, best_score = k, cer
    chosen = record.get(best_key) if best_key else next(iter(record.values()))
    return re.sub(r"\s+", " ", chosen.get("text", "")).strip()


def load_ocr_texts(path: str | Path) -> List[Tuple[str, str]]:
    """[(card_id, best_OCR_text), …] for all non-empty texts."""
    with open(path, encoding="utf-8") as f:
        raw = json.load(f)
    return [(cid, pick_best_variant(rec)) for cid, rec in raw.items() if pick_best_variant(rec)]


def dynamic_tsne(vectors: np.ndarray) -> np.ndarray:
    """2-D t-SNE with safe perplexity for small sample counts."""
    n = len(vectors)
    perp = max(1, min(30, n // 3))
    return TSNE(
        n_components=2, perplexity=perp, init="pca", learning_rate="auto", random_state=42
    ).fit_transform(vectors)


def tsne_plot(vectors, labels, title, out_path):
    coords = dynamic_tsne(np.array(vectors))
    plt.figure(figsize=(7, 6))
    plt.scatter(coords[:, 0], coords[:, 1], s=30, alpha=0.6)
    for (x, y), lbl in zip(coords, labels):
        plt.text(x, y, lbl, fontsize=8)
    plt.title(title)
    plt.tight_layout()
    plt.savefig(out_path)
    plt.close()


#  spaCy par
def run_spacy(samples: List[Tuple[str, str]], out_json: Path):
    """
    Run spaCy NER on *samples* ([(id, text)…]) and save JSON with
      { id: { "ocr_text": str,
              "entities": [ {text,label}, … ] } }
    Also produces t-SNE plots & nearest-neighbour console output.
    """
    per_card: Dict[str, Dict[str, Any]] = {}
    by_label_vectors: Dict[str, List[np.ndarray]] = defaultdict(list)
    by_label_tokens: Dict[str, List[str]] = defaultdict(list)
    all_vecs, all_labels = [], []

    print("▶ spaCy NER …")
    for (cid, text), doc in zip(samples, nlp.pipe([t for _, t in samples])):
        ents = [{"text": e.text, "label": e.label_} for e in doc.ents if e.vector_norm]
        per_card[cid] = {"ocr_text": text, "entities": ents}

        for ent in doc.ents:
            if ent.vector_norm:
                by_label_vectors[ent.label_].append(ent.vector)
                by_label_tokens[ent.label_].append(ent.text)
                all_vecs.append(ent.vector)
                all_labels.append(ent.text)

    out_json.write_text(
        json.dumps(per_card, ensure_ascii=False, indent=2),
        encoding = "utf-8")
    print(f"  ↳ JSON saved → {out_json}")

    # t-SNE visualisations
    PLOTS = Path("spacy_outputs")
    PLOTS.mkdir(exist_ok=True)
    print("▶ t-SNE plots …")
    for lab, vecs in by_label_vectors.items():
        if len(vecs) >= 3:
            tsne_plot(vecs, by_label_tokens[lab], f"t-SNE {lab} (n={len(vecs)})", PLOTS / f"tsne_{lab}.png")
    if len(all_vecs) >= 3:
        tsne_plot(all_vecs, all_labels, "t-SNE all entities", PLOTS / "tsne_all.png")

    # nearest-neighbour demo
    print("\n▶ Similar-word demo:")
    uniques = {w.lower() for w in all_labels if " " not in w}
    tokens = [nlp.vocab[w] for w in uniques if nlp.vocab[w].vector_norm]
    for tok in random.sample(tokens, k=min(10, len(tokens))):
        sims = sorted(tokens, key=lambda t: tok.similarity(t), reverse=True)[1:6]
        print(f"  {tok.text}  →  {', '.join(t.text for t in sims)}")


# LLM part
def call_llm(text: str, api_key: str, model: str, prompt_tpl: str) -> Dict[str, str]:
    from openai import OpenAI

    client = OpenAI(api_key=api_key, base_url=DEEPOINFRA_BASE_URL)
    messages = [
        {"role": "system", "content": "You are a precise, JSON-only extraction assistant."},
        {"role": "user", "content": prompt_tpl.format(sample=text)},
    ]
    resp = client.chat.completions.create(model=model, messages=messages, temperature=0.0)
    content = resp.choices[0].message.content.strip()
    m = re.search(r"{.*}", content, re.S)
    if not m:
        raise ValueError("No JSON found in LLM output")
    return json.loads(m.group(0))


def run_llm(samples: List[Tuple[str, str]], prompts_file: Path, models: List[str], api_key: str):
    with open(prompts_file, encoding="utf-8") as f:
        prompt_defs = json.load(f)

    for p in prompt_defs:
        p_id, tpl = p["id"], p["template"]
        for model in models:
            out_path = RESULTS_DIR / f"llm_{model.replace('/','-')}__{p_id}.json"
            results: Dict[str, Any] = {}

            print(f"▶ {model}  |  prompt={p_id}")
            for cid, text in samples:
                try:
                    pred = call_llm(text, api_key, model, tpl)
                    results[cid] = {"ocr_text": text, "prediction": pred}
                except Exception as e:
                    results[cid] = {"ocr_text": text, "error": str(e)}

            out_path.write_text(
                json.dumps(results, ensure_ascii=False, indent=2),
                encoding = "utf-8")
            print(f"  ↳ JSON saved → {out_path}")


#  evaluation part
def load_gt(path: Path) -> Dict[str, str]:
    """Return {id: location_str} from CSV or JSON."""
    if path.suffix.lower() == ".csv":
        with open(path, newline="", encoding="utf-8") as f:
            return {r["id"]: r["location"] for r in csv.DictReader(f)}
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def eval_one(pred_json: Path, gt: Dict[str, str]) -> Tuple[float, float]:
    with open(pred_json, encoding="utf-8") as f:
        preds = json.load(f)
    tp = fp = fn = 0
    for cid, true_loc in gt.items():
        if not true_loc:
            continue  # skip GT-missing rows
        pred_loc = (preds.get(cid, {}).get("prediction", {}).get("Location") or "").strip().lower()
        if pred_loc == true_loc.lower():
            tp += 1
        else:
            fp += 1
            fn += 1
    prec = tp / (tp + fp) if tp + fp else 0
    rec  = tp / (tp + fn) if tp + fn else 0
    f1   = 2 * prec * rec / (prec + rec) if prec + rec else 0
    acc  = tp / (tp + fn) if tp + fn else 0
    return acc, f1


def run_evaluation(gt_path: Path):
    gt = load_gt(gt_path)
    rows = []
    for jf in RESULTS_DIR.glob("llm_*.json"):
        acc, f1 = eval_one(jf, gt)
        rows.append((jf.name, acc, f1))
    for name, acc, f1 in sorted(rows, key=lambda r: -r[1]):
        print(f"{name:50}  acc={acc:.2f}  f1={f1:.2f}")


#  CLI / main entry 
def main_C(argv: List[str] | None = None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--ocr_json", default=DEFAULT_OCR_JSON, help="OCR JSON from Task B")
    ap.add_argument("--sample_ids", help="Comma-sep card IDs to use instead of random 5")
    ap.add_argument("--llm", action="store_true", help="Run LLM extraction grid")
    ap.add_argument("--prompts", default="prompts_example.json", help="JSON list of prompt templates")
    ap.add_argument("--models", default=",".join(DEFAULT_LLM_MODELS), help="Comma-sep model names")
    ap.add_argument("--api_key", default=os.getenv("DEEPINFRA_TOKEN"), help="DeepInfra / OpenAI key")
    ap.add_argument("--evaluate", help="CSV/JSON with ground-truth locations (id,location)")
    args = ap.parse_args(argv)

    # 1) Load OCR texts 
    all_samples = load_ocr_texts(args.ocr_json)
    if not all_samples:
        sys.exit("No OCR texts found.")
    print(f"Total cards loaded for spaCy: {len(all_samples)}")

    # 2) spaCy on all cards 
    run_spacy(all_samples, RESULTS_DIR / "spacy_outputs.json")

    # 3) Select ≤5 samples for LLM / evaluation  
    if args.sample_ids:
        ids_set = set(args.sample_ids.split(","))
        samples = [(cid, txt) for cid, txt in all_samples if cid in ids_set]
    else:
        samples = random.sample(all_samples, k=min(5, len(all_samples)))
    print("Samples for LLM/eval:", ", ".join(cid for cid, _ in samples))

    # 4) LLM grid  
    if args.llm:
        if not args.api_key:
            sys.exit("--llm specified but no API key provided "
                     "(use --api_key or DEEPINFRA_TOKEN env)")
        run_llm(samples, Path(args.prompts), args.models.split(","), args.api_key)

    # 5) Evaluation
    if args.evaluate:
        run_evaluation(Path(args.evaluate))

if __name__ == "__main__":
    main_C()
