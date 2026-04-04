"""
Text Classification: Human vs AI-generated Code Detection
Ensemble approach: combines CodeBERT + UniXcoder embeddings via
  (a) feature concatenation,
  (b) hard / soft voting, and
  (c) stacking (meta-learning).
All embeddings are loaded from cache — no GPU inference required.
"""

import os
import time
import warnings
from datetime import datetime

import numpy as np
import pandas as pd
import torch
from scipy.stats import mode
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
DATA_DIR   = "Task_A"
CACHE_DIR  = "embedding_cache"
CACHE_HASH = "d268f4ff"   # matches MAX_LENGTH=512, normalize=True

XGBOOST_DEVICE = None     # None = auto-detect CUDA

# Feature flags — set False to skip a strategy
STRATEGIES = {
    "concat":   True,
    "voting":   True,
    "stacking": True,
}

VOTING_MODES           = ["hard", "soft"]   # soft requires predict_proba
STACKING_META_C        = 1.0                # meta-learner regularisation

# ─────────────────────────────────────────────
# 2. CUDA detection
# ─────────────────────────────────────────────
if XGBOOST_DEVICE is None:
    XGBOOST_DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
print(f"XGBoost device: {XGBOOST_DEVICE}")

# ─────────────────────────────────────────────
# 3. Base classifier prototypes
# ─────────────────────────────────────────────
_CLASSIFIER_PROTOS = {
    "LogisticRegression": LogisticRegression(max_iter=1000, C=1.0, n_jobs=4),
    "LinearSVC":          LinearSVC(max_iter=2000, C=1.0),
    "XGBoost":            XGBClassifier(
        n_estimators=300, learning_rate=0.1, max_depth=6,
        n_jobs=4, random_state=42, eval_metric="logloss",
        tree_method="hist", device=XGBOOST_DEVICE,
    ),
}


def make_classifier(name: str):
    """Return a fresh (unfitted) copy of the named classifier."""
    proto = _CLASSIFIER_PROTOS[name]
    return proto.__class__(**proto.get_params())


# ─────────────────────────────────────────────
# 4. Data loading  (labels + IDs only)
# ─────────────────────────────────────────────
print("Loading data...")
train_df = pd.read_parquet(f"{DATA_DIR}/train.parquet")
val_df   = pd.read_parquet(f"{DATA_DIR}/validation.parquet")
test_df  = pd.read_parquet(f"{DATA_DIR}/test.parquet")
pub_df   = pd.read_parquet(f"{DATA_DIR}/test_sample.parquet")

y_train  = train_df["label"].tolist()
y_val    = val_df["label"].tolist()
y_pub    = pub_df["label"].tolist()
test_ids = test_df["ID"].tolist()

print(f"  Train: {len(y_train):,} | Val: {len(y_val):,} | "
      f"Test: {len(test_ids):,} | Public test: {len(y_pub):,}")

# ─────────────────────────────────────────────
# 5. Embedding loader
# ─────────────────────────────────────────────
def load_embeddings(model_key: str, split: str) -> np.ndarray:
    """
    Load a pre-computed L2-normalised embedding array.

    Args:
        model_key: "codebert" | "unixcoder"
        split:     "train" | "val" | "pub" | "test"

    Returns:
        np.ndarray shape (N, 768), float32
    """
    path = os.path.join(CACHE_DIR, model_key, f"{split}_{CACHE_HASH}.npy")
    if not os.path.exists(path):
        raise FileNotFoundError(f"Embedding cache missing: {path}")
    emb = np.load(path)
    print(f"  [load] {model_key}/{split}  shape={emb.shape}")
    return emb


# ─────────────────────────────────────────────
# 6. Evaluation helper
# ─────────────────────────────────────────────
def evaluate(name, clf, X_val_emb, y_val, X_pub_emb, y_pub):
    val_pred = clf.predict(X_val_emb)
    pub_pred = clf.predict(X_pub_emb)
    print(f"\n{'='*55}")
    print(f"  {name}")
    print(f"{'='*55}")
    print(f"  [Validation]  Acc: {accuracy_score(y_val, val_pred):.4f}  "
          f"F1: {f1_score(y_val, val_pred):.4f}")
    print(f"  [Public Test] Acc: {accuracy_score(y_pub, pub_pred):.4f}  "
          f"F1: {f1_score(y_pub, pub_pred):.4f}")
    print(f"\n  Validation classification report:")
    print(classification_report(y_val, val_pred, target_names=["Human", "AI"]))
    print(f"\n  Public test classification report:")
    print(classification_report(y_pub, pub_pred, target_names=["Human", "AI"]))
    return val_pred, pub_pred


# ─────────────────────────────────────────────
# 7. Output helper
# ─────────────────────────────────────────────
def save_outputs(safe_name, clf_or_pred, X_test_emb,
                 val_pred, pub_pred):
    """Write submission + prediction CSVs."""
    if callable(getattr(clf_or_pred, "predict", None)):
        test_pred = clf_or_pred.predict(X_test_emb)
    else:
        # pre-computed array passed directly
        test_pred = clf_or_pred

    sub_path = os.path.join(run_dir, "submission", f"{safe_name}.csv")
    pd.DataFrame({"ID": test_ids, "label": test_pred}).to_csv(sub_path, index=False)

    val_path = os.path.join(run_dir, "predictions", f"val_{safe_name}.csv")
    pd.DataFrame({"true_label": y_val, "pred_label": val_pred}).to_csv(val_path, index=False)

    pub_path = os.path.join(run_dir, "predictions", f"pub_{safe_name}.csv")
    pd.DataFrame({"true_label": y_pub, "pred_label": pub_pred}).to_csv(pub_path, index=False)

    print(f"  Saved {sub_path}")


# ─────────────────────────────────────────────
# 8. Load all embeddings upfront
#    Memory: 4 splits × 2 models × 768 × float32 ≈ 6 GB
# ─────────────────────────────────────────────
print("\nLoading all cached embeddings...")
cb_train = load_embeddings("codebert",  "train")
cb_val   = load_embeddings("codebert",  "val")
cb_pub   = load_embeddings("codebert",  "pub")
cb_test  = load_embeddings("codebert",  "test")

ux_train = load_embeddings("unixcoder", "train")
ux_val   = load_embeddings("unixcoder", "val")
ux_pub   = load_embeddings("unixcoder", "pub")
ux_test  = load_embeddings("unixcoder", "test")

results = []


# ─────────────────────────────────────────────
# 9a. Strategy: Feature Concatenation
#     [CodeBERT | UniXcoder] → 1536-dim → each base classifier
# ─────────────────────────────────────────────
def concat_ensemble():
    print(f"\n{'#'*60}")
    print(f"  Strategy: Feature Concatenation  (1536-dim)")
    print(f"{'#'*60}")

    X_train = np.concatenate([cb_train, ux_train], axis=1)
    X_val   = np.concatenate([cb_val,   ux_val],   axis=1)
    X_pub   = np.concatenate([cb_pub,   ux_pub],   axis=1)
    X_test  = np.concatenate([cb_test,  ux_test],  axis=1)

    for clf_name in _CLASSIFIER_PROTOS:
        clf = make_classifier(clf_name)
        print(f"\n  Training {clf_name}...")
        t0 = time.time()
        clf.fit(X_train, y_train)
        train_time = time.time() - t0
        print(f"  Done in {train_time:.1f}s")

        safe_name = f"concat_{clf_name.lower()}"
        val_pred, pub_pred = evaluate(f"Concat + {clf_name}", clf, X_val, y_val, X_pub, y_pub)
        save_outputs(safe_name, clf, X_test, val_pred, pub_pred)

        results.append({
            "strategy":     "concat",
            "detail":       clf_name,
            "train_time_s": round(train_time, 1),
            "val_acc":      round(accuracy_score(y_val, val_pred), 4),
            "val_f1":       round(f1_score(y_val, val_pred), 4),
            "pub_acc":      round(accuracy_score(y_pub, pub_pred), 4),
            "pub_f1":       round(f1_score(y_pub, pub_pred), 4),
            "val_report":   classification_report(y_val, val_pred, target_names=["Human", "AI"]),
            "pub_report":   classification_report(y_pub, pub_pred, target_names=["Human", "AI"]),
        })


# ─────────────────────────────────────────────
# 9b. Strategy: Voting Ensemble
#     Train 6 base models (3 clf × 2 emb), combine predictions
# ─────────────────────────────────────────────
def voting_ensemble():
    print(f"\n{'#'*60}")
    print(f"  Strategy: Voting Ensemble")
    print(f"{'#'*60}")

    # Train all 6 base models
    fitted_models = []   # (clf_name, emb_name, clf, X_val, X_pub, X_test)
    for emb_name, (X_tr, X_va, X_pb, X_te) in [
        ("codebert",  (cb_train, cb_val, cb_pub, cb_test)),
        ("unixcoder", (ux_train, ux_val, ux_pub, ux_test)),
    ]:
        for clf_name in _CLASSIFIER_PROTOS:
            clf = make_classifier(clf_name)
            print(f"\n  Training {emb_name} / {clf_name}...")
            t0 = time.time()
            clf.fit(X_tr, y_train)
            print(f"  Done in {time.time()-t0:.1f}s")
            fitted_models.append((clf_name, emb_name, clf, X_va, X_pb, X_te))

    # --- Hard voting ---
    if "hard" in VOTING_MODES:
        val_mat  = np.vstack([clf.predict(va) for _, _, clf, va, _, _ in fitted_models])
        pub_mat  = np.vstack([clf.predict(pb) for _, _, clf, _, pb, _ in fitted_models])
        test_mat = np.vstack([clf.predict(te) for _, _, clf, _, _, te in fitted_models])

        val_pred  = mode(val_mat,  axis=0, keepdims=False).mode.astype(int)
        pub_pred  = mode(pub_mat,  axis=0, keepdims=False).mode.astype(int)
        test_pred = mode(test_mat, axis=0, keepdims=False).mode.astype(int)

        _log_vote_results("Voting Hard (6 models)", "voting_hard",
                          val_pred, pub_pred, test_pred)

    # --- Soft voting (proba-capable models only) ---
    if "soft" in VOTING_MODES:
        proba_models = [(n, e, c, va, pb, te) for (n, e, c, va, pb, te) in fitted_models
                        if hasattr(c, "predict_proba")]
        if len(proba_models) >= 2:
            val_p  = np.mean([c.predict_proba(va)[:, 1] for _, _, c, va, _, _ in proba_models], axis=0)
            pub_p  = np.mean([c.predict_proba(pb)[:, 1] for _, _, c, _, pb, _ in proba_models], axis=0)
            test_p = np.mean([c.predict_proba(te)[:, 1] for _, _, c, _, _, te in proba_models], axis=0)

            val_pred  = (val_p  >= 0.5).astype(int)
            pub_pred  = (pub_p  >= 0.5).astype(int)
            test_pred = (test_p >= 0.5).astype(int)

            n_soft = len(proba_models)
            _log_vote_results(f"Voting Soft ({n_soft} models)", "voting_soft",
                              val_pred, pub_pred, test_pred)

    return fitted_models   # reused by stacking


def _log_vote_results(name, safe_name, val_pred, pub_pred, test_pred):
    print(f"\n{'='*55}")
    print(f"  {name}")
    print(f"{'='*55}")
    print(f"  [Validation]  Acc: {accuracy_score(y_val, val_pred):.4f}  "
          f"F1: {f1_score(y_val, val_pred):.4f}")
    print(f"  [Public Test] Acc: {accuracy_score(y_pub, pub_pred):.4f}  "
          f"F1: {f1_score(y_pub, pub_pred):.4f}")
    print(classification_report(y_val, val_pred, target_names=["Human", "AI"]))

    sub_path = os.path.join(run_dir, "submission", f"{safe_name}.csv")
    pd.DataFrame({"ID": test_ids, "label": test_pred}).to_csv(sub_path, index=False)
    val_path = os.path.join(run_dir, "predictions", f"val_{safe_name}.csv")
    pd.DataFrame({"true_label": y_val, "pred_label": val_pred}).to_csv(val_path, index=False)
    pub_path = os.path.join(run_dir, "predictions", f"pub_{safe_name}.csv")
    pd.DataFrame({"true_label": y_pub, "pred_label": pub_pred}).to_csv(pub_path, index=False)
    print(f"  Saved {sub_path}")

    results.append({
        "strategy":     safe_name,
        "detail":       name,
        "train_time_s": "-",
        "val_acc":      round(accuracy_score(y_val, val_pred), 4),
        "val_f1":       round(f1_score(y_val, val_pred), 4),
        "pub_acc":      round(accuracy_score(y_pub, pub_pred), 4),
        "pub_f1":       round(f1_score(y_pub, pub_pred), 4),
        "val_report":   classification_report(y_val, val_pred, target_names=["Human", "AI"]),
        "pub_report":   classification_report(y_pub, pub_pred, target_names=["Human", "AI"]),
    })


# ─────────────────────────────────────────────
# 9c. Strategy: Stacking (Meta-Learning)
#     Stage 1: 6 base models on train split
#     Stage 2: level-1 features on val → LogisticRegression meta-learner
#     Anti-leakage: meta-learner sees only val labels, not train labels
# ─────────────────────────────────────────────
def stacking_ensemble(fitted_models=None):
    """
    If fitted_models is provided (from voting_ensemble), reuses them to
    avoid re-training. Otherwise trains 6 base models from scratch.
    """
    print(f"\n{'#'*60}")
    print(f"  Strategy: Stacking (Meta-Learning)")
    print(f"{'#'*60}")

    if fitted_models is None:
        fitted_models = []
        for emb_name, (X_tr, X_va, X_pb, X_te) in [
            ("codebert",  (cb_train, cb_val, cb_pub, cb_test)),
            ("unixcoder", (ux_train, ux_val, ux_pub, ux_test)),
        ]:
            for clf_name in _CLASSIFIER_PROTOS:
                clf = make_classifier(clf_name)
                print(f"\n  Training {emb_name} / {clf_name}...")
                t0 = time.time()
                clf.fit(X_tr, y_train)
                print(f"  Done in {time.time()-t0:.1f}s")
                fitted_models.append((clf_name, emb_name, clf, X_va, X_pb, X_te))

    def _level1_col(clf, X):
        """Continuous signal regardless of classifier type."""
        if hasattr(clf, "predict_proba"):
            return clf.predict_proba(X)[:, 1]
        return clf.decision_function(X)   # LinearSVC

    print("\n  Building level-1 feature matrices...")
    meta_val  = np.column_stack([_level1_col(c, va) for _, _, c, va, _, _ in fitted_models])
    meta_pub  = np.column_stack([_level1_col(c, pb) for _, _, c, _, pb, _ in fitted_models])
    meta_test = np.column_stack([_level1_col(c, te) for _, _, c, _, _, te in fitted_models])
    print(f"  Level-1 shape: val={meta_val.shape}, pub={meta_pub.shape}, test={meta_test.shape}")

    # Meta-learner trained on val level-1 features (no train-set leakage)
    meta_clf = LogisticRegression(max_iter=1000, C=STACKING_META_C, n_jobs=4)
    print("\n  Training meta-learner (LogisticRegression)...")
    t0 = time.time()
    meta_clf.fit(meta_val, y_val)
    train_time = time.time() - t0
    print(f"  Done in {train_time:.1f}s")

    val_pred, pub_pred = evaluate(
        "Stacking (meta=LogisticRegression)", meta_clf,
        meta_val, y_val, meta_pub, y_pub,
    )
    save_outputs("stacking_lr_meta", meta_clf, meta_test, val_pred, pub_pred)

    results.append({
        "strategy":     "stacking",
        "detail":       "meta=LogisticRegression",
        "train_time_s": round(train_time, 1),
        "val_acc":      round(accuracy_score(y_val, val_pred), 4),
        "val_f1":       round(f1_score(y_val, val_pred), 4),
        "pub_acc":      round(accuracy_score(y_pub, pub_pred), 4),
        "pub_f1":       round(f1_score(y_pub, pub_pred), 4),
        "val_report":   classification_report(y_val, val_pred, target_names=["Human", "AI"]),
        "pub_report":   classification_report(y_pub, pub_pred, target_names=["Human", "AI"]),
    })


# ─────────────────────────────────────────────
# 10. Main dispatch
# ─────────────────────────────────────────────
_voting_models = None   # shared between voting and stacking

if STRATEGIES["concat"]:
    concat_ensemble()

if STRATEGIES["voting"]:
    _voting_models = voting_ensemble()

if STRATEGIES["stacking"]:
    # Reuse already-trained voting base models if available
    stacking_ensemble(fitted_models=_voting_models)

# ─────────────────────────────────────────────
# 11. Summary table
# ─────────────────────────────────────────────
sorted_results = sorted(results, key=lambda x: -x["pub_f1"])

print(f"\n\n{'='*80}")
print("  SUMMARY  (sorted by pub_f1 desc)")
print(f"{'='*80}")
print(f"  {'Strategy':<14} {'Detail':<28} {'Val Acc':>8} {'Val F1':>8} {'Pub Acc':>8} {'Pub F1':>8}")
print(f"  {'-'*14} {'-'*28} {'-'*8} {'-'*8} {'-'*8} {'-'*8}")
for r in sorted_results:
    print(f"  {r['strategy']:<14} {r['detail']:<28} "
          f"{r['val_acc']:>8.4f} {r['val_f1']:>8.4f} "
          f"{r['pub_acc']:>8.4f} {r['pub_f1']:>8.4f}")
print(f"{'='*80}\n")

# ─────────────────────────────────────────────
# 12. Markdown report
# ─────────────────────────────────────────────
run_ts = os.path.basename(run_dir)
md_lines = [
    f"# Ensemble Experiment Report — {run_ts}",
    "",
    "## Configuration",
    f"- **Strategies**: {', '.join(k for k, v in STRATEGIES.items() if v)}",
    f"- **Base classifiers**: {', '.join(_CLASSIFIER_PROTOS.keys())}",
    f"- **Voting modes**: {', '.join(VOTING_MODES)}",
    f"- **Stacking meta-learner**: LogisticRegression (C={STACKING_META_C})",
    f"- **Embeddings**: CodeBERT (768-dim) + UniXcoder (768-dim), L2-normalised, pre-computed",
    f"- **Train samples**: {len(y_train):,} | Val: {len(y_val):,} | "
    f"Pub: {len(y_pub):,} | Test: {len(test_ids):,}",
    "",
    "## Results Summary",
    "",
    "| Strategy | Detail | Train Time (s) | Val Acc | Val F1 | Pub Acc | Pub F1 |",
    "|---|---|---:|---:|---:|---:|---:|",
]
for r in sorted_results:
    md_lines.append(
        f"| {r['strategy']} | {r['detail']} | {r['train_time_s']} "
        f"| {r['val_acc']:.4f} | {r['val_f1']:.4f} | {r['pub_acc']:.4f} | {r['pub_f1']:.4f} |"
    )

md_lines += ["", "---", "", "## Detailed Classification Reports", ""]
for r in sorted_results:
    title = f"{r['strategy']} — {r['detail']}"
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
