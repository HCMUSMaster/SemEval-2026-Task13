"""
Text Classification: Human vs AI-generated Code Detection
Embedding approach using CodeBERT / UniXcoder + ML classifiers
"""

import os
import time
import json
import hashlib
import warnings
from datetime import datetime

import numpy as np
from tqdm import tqdm
import pandas as pd
import torch
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModel
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.metrics import accuracy_score, f1_score, classification_report
from xgboost import XGBClassifier

warnings.filterwarnings("ignore")

# ─────────────────────────────────────────────
# 0. Output directory
# ─────────────────────────────────────────────
run_dir = os.path.join("log", datetime.now().strftime("%Y%m%d_%H%M%S"))
os.makedirs(os.path.join(run_dir, "submission"), exist_ok=True)
os.makedirs(os.path.join(run_dir, "predictions"), exist_ok=True)
print(f"Saving outputs to: {run_dir}/")

# ─────────────────────────────────────────────
# 1. Configuration
# ─────────────────────────────────────────────
DATA_DIR       = "Task_A"
CACHE_DIR      = "embedding_cache"
MAX_LENGTH     = 512    # tokens; covers ~90th pct of code lengths
BATCH_SIZE     = 512     # reduce to 8-16 if GPU OOM
NORMALIZE_EMBS = True   # L2-normalize embeddings before training
TRAIN_SUBSAMPLE = 100_000  # None = use all 500K; lower for fast iteration
XGBOOST_DEVICE = None   # None = auto-detect CUDA

# Each entry: model_name, short cache key, optional text prefix for tokenization.
# UniXcoder encoder-only mode prepends "<encoder-only>" before the code tokens so
# that the leading </s> / <s> token aggregates a code-specific representation.
EMBEDDING_CONFIGS = {
    "CodeBERT": {
        "model_name":  "microsoft/codebert-base",
        "cache_key":   "codebert",
        "text_prefix": None,            # standard [CLS] code [SEP] format
    },
    "UniXcoder": {
        "model_name":  "microsoft/unixcoder-base",
        "cache_key":   "unixcoder",
        "text_prefix": "<encoder-only>",  # replicates encoder-only mode
    },
    "JinJa": {
        "model_name":  "jinaai/jina-embeddings-v2-base-code",
        "cache_key":   "jinja",
        "text_prefix": "None",
    },
    "GraphCode": {
        "model_name":  "microsoft/graphcodebert-base",
        "cache_key":   "graphcode",
        "text_prefix": "None",
    },
}

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {DEVICE}")
if XGBOOST_DEVICE is None:
    XGBOOST_DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# ─────────────────────────────────────────────
# 2. Load Data
# ─────────────────────────────────────────────
print("Loading data...")
train_df = pd.read_parquet(f"{DATA_DIR}/train.parquet")
val_df   = pd.read_parquet(f"{DATA_DIR}/validation.parquet")
test_df  = pd.read_parquet(f"{DATA_DIR}/test.parquet")
pub_df   = pd.read_parquet(f"{DATA_DIR}/test_sample.parquet")

# if TRAIN_SUBSAMPLE is not None:
#     train_df = train_df.sample(n=TRAIN_SUBSAMPLE, random_state=42).reset_index(drop=True)

X_train  = train_df["code"].fillna("").tolist()
y_train  = train_df["label"].tolist()

X_val    = val_df["code"].fillna("").tolist()
y_val    = val_df["label"].tolist()

X_test   = test_df["code"].fillna("").tolist()
test_ids = test_df["ID"].tolist()

X_pub    = pub_df["code"].fillna("").tolist()
y_pub    = pub_df["label"].tolist()

print(f"  Train: {len(X_train):,} | Val: {len(X_val):,} | Test: {len(X_test):,} | Public test: {len(X_pub):,}")

# ─────────────────────────────────────────────
# 3. Evaluation helper
# ─────────────────────────────────────────────
def evaluate(name, clf, X_val_emb, y_val, X_pub_emb, y_pub):
    val_pred = clf.predict(X_val_emb)
    pub_pred = clf.predict(X_pub_emb)
    print(f"\n{'='*55}")
    print(f"  {name}")
    print(f"{'='*55}")
    print(f"  [Validation]  Acc: {accuracy_score(y_val, val_pred):.4f}  F1: {f1_score(y_val, val_pred):.4f}")
    print(f"  [Public Test] Acc: {accuracy_score(y_pub, pub_pred):.4f}  F1: {f1_score(y_pub, pub_pred):.4f}")
    print(f"\n  Validation classification report:")
    print(classification_report(y_val, val_pred, target_names=["Human", "AI"]))
    print(f"\n  Public test classification report:")
    print(classification_report(y_pub, pub_pred, target_names=["Human", "AI"]))
    return val_pred, pub_pred

# ─────────────────────────────────────────────
# 4. Embedding generation with caching
# ─────────────────────────────────────────────
def _cache_path(cache_key: str) -> str:
    cfg_hash = hashlib.md5(
        json.dumps({"max_length": MAX_LENGTH, "normalize": NORMALIZE_EMBS},
                   sort_keys=True).encode()
    ).hexdigest()[:8]
    return os.path.join(CACHE_DIR, cache_key, f"{{split}}_{cfg_hash}.npy")


def get_embeddings(texts: list, split_name: str, tokenizer, model,
                   cache_tmpl: str, text_prefix: str | None) -> np.ndarray:
    """
    Returns CLS-pooled embeddings, shape (N, hidden_dim), float32.
    Caches to disk; reloads on subsequent calls if shape matches.

    Args:
        cache_tmpl:  format string with {split} placeholder, e.g.
                     "embedding_cache/codebert/{split}_abc12345.npy"
        text_prefix: optional string prepended to each code snippet before
                     tokenization (used for UniXcoder encoder-only mode).
    """
    cache_file = cache_tmpl.format(split=split_name)
    os.makedirs(os.path.dirname(cache_file), exist_ok=True)

    # Cache hit
    if os.path.exists(cache_file):
        cached = np.load(cache_file)
        if cached.shape[0] == len(texts):
            print(f"  [cache hit] {split_name}  {cache_file}  shape={cached.shape}")
            return cached
        print(f"  [cache stale] {cache_file}: {cached.shape[0]} rows vs {len(texts)}. Recomputing.")

    # Apply optional prefix
    if text_prefix:
        inputs = [f"{text_prefix} {t}" for t in texts]
    else:
        inputs = texts

    all_embeddings = []
    model.eval()
    t0 = time.time()

    with torch.no_grad():
        for start in tqdm(range(0, len(inputs), BATCH_SIZE),
                          desc=f"  Embedding [{split_name}]", unit="batch"):
            batch = inputs[start: start + BATCH_SIZE]
            batch_dict = tokenizer(
                batch,
                max_length=MAX_LENGTH,
                padding=True,
                truncation=True,
                return_tensors="pt",
            )
            batch_dict = {k: v.to(model.device) for k, v in batch_dict.items()}

            outputs = model(**batch_dict)
            # CLS pooling: first token aggregates full sequence context
            emb = outputs.last_hidden_state[:, 0, :]

            if NORMALIZE_EMBS:
                emb = F.normalize(emb, p=2, dim=1)

            all_embeddings.append(emb.cpu().float().numpy())

    result = np.concatenate(all_embeddings, axis=0)
    np.save(cache_file, result)
    print(f"  Saved {split_name} → {cache_file}  shape={result.shape}  time={time.time()-t0:.1f}s")
    return result

# ─────────────────────────────────────────────
# 5. Classifiers
# ─────────────────────────────────────────────
CLASSIFIERS = {
    "LogisticRegression": LogisticRegression(
        max_iter=1000, C=1.0, n_jobs=4,
    ),
    "LinearSVC": LinearSVC(
        max_iter=2000, C=1.0,
    ),
    "XGBoost": XGBClassifier(
        n_estimators=300, learning_rate=0.1, max_depth=6,
        n_jobs=4, random_state=42, eval_metric="logloss",
        tree_method="hist", device=XGBOOST_DEVICE,
    ),
}

results = []

# ─────────────────────────────────────────────
# 6. Main loop: embedding model × classifier
# ─────────────────────────────────────────────
for emb_name, emb_cfg in EMBEDDING_CONFIGS.items():
    model_name  = emb_cfg["model_name"]
    cache_key   = emb_cfg["cache_key"]
    text_prefix = emb_cfg["text_prefix"]
    cache_tmpl  = _cache_path(cache_key)

    print(f"\n{'#'*55}")
    print(f"  Embedding model: {emb_name}  ({model_name})")
    print(f"{'#'*55}")

    # Load tokenizer + model
    t_model = time.time()
    print(f"  Loading model...")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModel.from_pretrained(
        model_name,
        torch_dtype=torch.float16 if DEVICE.type == "cuda" else torch.float32,
    ).to(DEVICE)
    model.eval()
    print(f"  Model loaded in {time.time()-t_model:.1f}s")

    # Generate / load cached embeddings
    print(f"  Generating embeddings...")
    train_emb = get_embeddings(X_train, "train", tokenizer, model, cache_tmpl, text_prefix)
    val_emb   = get_embeddings(X_val,   "val",   tokenizer, model, cache_tmpl, text_prefix)
    pub_emb   = get_embeddings(X_pub,   "pub",   tokenizer, model, cache_tmpl, text_prefix)
    test_emb  = get_embeddings(X_test,  "test",  tokenizer, model, cache_tmpl, text_prefix)

    # Free GPU memory before classifier training
    del model
    if DEVICE.type == "cuda":
        torch.cuda.empty_cache()

    # Train each classifier on the embeddings
    for clf_name, clf_proto in CLASSIFIERS.items():
        clf = clf_proto.__class__(**clf_proto.get_params())

        t1 = time.time()
        print(f"\n  Training {clf_name}...")
        clf.fit(train_emb, y_train)
        train_time = time.time() - t1
        print(f"  Training done in {train_time:.1f}s")

        val_pred, pub_pred = evaluate(
            f"{emb_name} + {clf_name}", clf,
            val_emb, y_val, pub_emb, y_pub
        )

        results.append({
            "embedding":    emb_name,
            "classifier":   clf_name,
            "train_time_s": round(train_time, 1),
            "val_acc":      round(accuracy_score(y_val, val_pred), 4),
            "val_f1":       round(f1_score(y_val, val_pred), 4),
            "pub_acc":      round(accuracy_score(y_pub, pub_pred), 4),
            "pub_f1":       round(f1_score(y_pub, pub_pred), 4),
            "val_report":   classification_report(y_val, val_pred, target_names=["Human", "AI"]),
            "pub_report":   classification_report(y_pub, pub_pred, target_names=["Human", "AI"]),
        })

        safe_name = f"{cache_key}_{clf_name.lower()}"

        # submission/ — unlabeled test predictions
        test_pred = clf.predict(test_emb)
        sub_path = os.path.join(run_dir, "submission", f"{safe_name}.csv")
        pd.DataFrame({"ID": test_ids, "label": test_pred}).to_csv(sub_path, index=False)
        print(f"  Saved {sub_path}")

        # predictions/ — validation and public test with true labels
        val_pred_path = os.path.join(run_dir, "predictions", f"val_{safe_name}.csv")
        pd.DataFrame({"true_label": y_val, "pred_label": val_pred}).to_csv(val_pred_path, index=False)

        pub_pred_path = os.path.join(run_dir, "predictions", f"pub_{safe_name}.csv")
        pd.DataFrame({"true_label": y_pub, "pred_label": pub_pred}).to_csv(pub_pred_path, index=False)
        print(f"  Saved predictions for val and public test")

# ─────────────────────────────────────────────
# 7. Summary table (console)
# ─────────────────────────────────────────────
sorted_results = sorted(results, key=lambda x: -x["val_f1"])

print(f"\n\n{'='*80}")
print("  SUMMARY")
print(f"{'='*80}")
print(f"  {'Embedding':<20} {'Classifier':<22} {'Val Acc':>8} {'Val F1':>8} {'Pub Acc':>8} {'Pub F1':>8}")
print(f"  {'-'*20} {'-'*22} {'-'*8} {'-'*8} {'-'*8} {'-'*8}")
for r in sorted_results:
    print(f"  {r['embedding']:<20} {r['classifier']:<22} "
          f"{r['val_acc']:>8.4f} {r['val_f1']:>8.4f} {r['pub_acc']:>8.4f} {r['pub_f1']:>8.4f}")
print(f"{'='*80}\n")

# ─────────────────────────────────────────────
# 8. Markdown report
# ─────────────────────────────────────────────
run_ts = os.path.basename(run_dir)
md_lines = [
    f"# Experiment Report — {run_ts}",
    "",
    "## Configuration",
    f"- **Embedding models**: {', '.join(EMBEDDING_CONFIGS.keys())}",
    f"- **Max sequence length**: {MAX_LENGTH} tokens",
    f"- **Embeddings normalized**: {NORMALIZE_EMBS}",
    f"- **Train samples**: {len(X_train):,}{' (subsampled)' if TRAIN_SUBSAMPLE else ' (full)'}",
    f"- **Validation samples**: {len(X_val):,}",
    f"- **Public test samples**: {len(X_pub):,}",
    f"- **Classifiers**: {', '.join(CLASSIFIERS.keys())}",
    "",
    "## Results Summary",
    "",
    "| Embedding | Classifier | Train Time (s) | Val Acc | Val F1 | Pub Acc | Pub F1 |",
    "|---|---|---:|---:|---:|---:|---:|",
]
for r in sorted_results:
    md_lines.append(
        f"| {r['embedding']} | {r['classifier']} | {r['train_time_s']} "
        f"| {r['val_acc']:.4f} | {r['val_f1']:.4f} | {r['pub_acc']:.4f} | {r['pub_f1']:.4f} |"
    )

md_lines += ["", "---", "", "## Detailed Classification Reports", ""]

for r in sorted_results:
    title = f"{r['embedding']} + {r['classifier']}"
    md_lines += [
        f"### {title}",
        "",
        "**Validation set**",
        "```",
        r["val_report"].strip(),
        "```",
        "",
        "**Public test set**",
        "```",
        r["pub_report"].strip(),
        "```",
        "",
    ]

report_path = os.path.join(run_dir, "report.md")
with open(report_path, "w") as f:
    f.write("\n".join(md_lines))
print(f"Report saved to {report_path}")
