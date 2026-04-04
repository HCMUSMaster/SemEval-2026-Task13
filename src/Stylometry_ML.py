"""
Text Classification: Human vs AI-generated Code Detection
Features: Stylometry + AST Structural Features
Models: LogisticRegression, LinearSVC, XGBoost
"""

import os
import re
import ast
import time
import hashlib
import pickle
import warnings
from datetime import datetime

import numpy as np
import pandas as pd
from tqdm import tqdm
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC, SVC
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score, f1_score, classification_report

warnings.filterwarnings("ignore")

# ─────────────────────────────────────────────
# 0. Output directory
# ─────────────────────────────────────────────
run_dir = os.path.join("log", datetime.now().strftime("%Y%m%d_%H%M%S"))
os.makedirs(os.path.join(run_dir, "submission"), exist_ok=True)
os.makedirs(os.path.join(run_dir, "predictions"), exist_ok=True)
print(f"Saving outputs to: {run_dir}/")


# ─────────────────────────────────────────────
# 1. Feature extractors
# ─────────────────────────────────────────────
def stylometry_features(code: str) -> dict:
    lines = code.split("\n")
    indent_sizes = []
    for l in lines:
        indent = len(l) - len(l.lstrip(" "))
        if indent > 0:
            indent_sizes.append(indent)

    avg_indent = np.mean(indent_sizes) if indent_sizes else 0
    line_lengths = [len(l) for l in lines]

    return {
        "num_lines": len(lines),
        "avg_line_length": np.mean(line_lengths),
        "max_line_length": max(line_lengths) if line_lengths else 0,
        "blank_ratio": sum(1 for l in lines if l.strip() == "") / max(len(lines), 1),
        "comment_ratio": code.count("#") / max(len(lines), 1),
        "avg_indent": avg_indent,
        "snake_case_vars": len(re.findall(r"[a-z]+_[a-z]+", code)),
        "camel_case_vars": len(re.findall(r"[a-z]+[A-Z][a-z]+", code)),
        "num_loops": len(re.findall(r"\bfor\b|\bwhile\b", code)),
        "num_conditionals": len(re.findall(r"\bif\b|\belif\b", code)),
    }


class ASTCounter(ast.NodeVisitor):
    def __init__(self):
        self.counts = {
            "functions": 0,
            "classes": 0,
            "loops": 0,
            "ifs": 0,
            "imports": 0,
        }

    def visit_FunctionDef(self, node):
        self.counts["functions"] += 1
        self.generic_visit(node)

    def visit_AsyncFunctionDef(self, node):
        self.counts["functions"] += 1
        self.generic_visit(node)

    def visit_ClassDef(self, node):
        self.counts["classes"] += 1
        self.generic_visit(node)

    def visit_For(self, node):
        self.counts["loops"] += 1
        self.generic_visit(node)

    def visit_While(self, node):
        self.counts["loops"] += 1
        self.generic_visit(node)

    def visit_If(self, node):
        self.counts["ifs"] += 1
        self.generic_visit(node)

    def visit_Import(self, node):
        self.counts["imports"] += 1
        self.generic_visit(node)

    def visit_ImportFrom(self, node):
        self.counts["imports"] += 1
        self.generic_visit(node)


def structural_features(code: str) -> dict:
    try:
        tree = ast.parse(code)
        counter = ASTCounter()
        counter.visit(tree)
        return counter.counts
    except Exception:
        return {"functions": 0, "classes": 0, "loops": 0, "ifs": 0, "imports": 0}


CACHE_DIR = "cache/stylometry"
os.makedirs(CACHE_DIR, exist_ok=True)


def _build_matrix(corpus: list[str], extractor, tag: str):
    key = hashlib.md5(("".join(corpus) + tag).encode()).hexdigest()
    cache_path = os.path.join(CACHE_DIR, f"{tag}_{key}.pkl")
    if os.path.exists(cache_path):
        print(f"  [{tag}] Loading from cache ({cache_path})")
        with open(cache_path, "rb") as f:
            return pickle.load(f)
    rows = [extractor(code) for code in tqdm(corpus, desc=tag, leave=False)]
    df = pd.DataFrame(rows)
    result = df.values.astype(np.float32), df.columns.tolist()
    with open(cache_path, "wb") as f:
        pickle.dump(result, f)
    return result


def build_feature_matrix(corpus: list[str]) -> np.ndarray:
    stylo_mat, stylo_cols = _build_matrix(corpus, stylometry_features, "stylometry")
    ast_mat,   ast_cols   = _build_matrix(corpus, structural_features,  "ast")
    return np.hstack([stylo_mat, ast_mat]), stylo_cols + ast_cols


# ─────────────────────────────────────────────
# 2. Load Data
# ─────────────────────────────────────────────
DATA_DIR = "Task_A"

print("Loading data...")
train_df = pd.read_parquet(f"{DATA_DIR}/train.parquet")
val_df   = pd.read_parquet(f"{DATA_DIR}/validation.parquet")
test_df  = pd.read_parquet(f"{DATA_DIR}/test.parquet")
pub_df   = pd.read_parquet(f"{DATA_DIR}/test_sample.parquet")

X_train_raw = train_df["code"].fillna("").tolist()
y_train     = train_df["label"].tolist()

X_val_raw   = val_df["code"].fillna("").tolist()
y_val       = val_df["label"].tolist()

X_test_raw  = test_df["code"].fillna("").tolist()
test_ids    = test_df["ID"].tolist()

X_pub_raw   = pub_df["code"].fillna("").tolist()
y_pub       = pub_df["label"].tolist()

print(f"  Train: {len(X_train_raw):,} | Val: {len(X_val_raw):,} | Test: {len(X_test_raw):,} | Public: {len(X_pub_raw):,}")


# ─────────────────────────────────────────────
# 3. Build hand-crafted feature matrices
# ─────────────────────────────────────────────
print("\nExtracting stylometry + structural features...")
t0 = time.time()

X_train_stylo, stylo_cols = _build_matrix(X_train_raw, stylometry_features, "stylometry")
X_val_stylo,   _          = _build_matrix(X_val_raw,   stylometry_features, "stylometry")
X_test_stylo,  _          = _build_matrix(X_test_raw,  stylometry_features, "stylometry")
X_pub_stylo,   _          = _build_matrix(X_pub_raw,   stylometry_features, "stylometry")

X_train_ast,   ast_cols   = _build_matrix(X_train_raw, structural_features, "ast")
X_val_ast,     _          = _build_matrix(X_val_raw,   structural_features, "ast")
X_test_ast,    _          = _build_matrix(X_test_raw,  structural_features, "ast")
X_pub_ast,     _          = _build_matrix(X_pub_raw,   structural_features, "ast")

X_train_hc = np.hstack([X_train_stylo, X_train_ast])
X_val_hc   = np.hstack([X_val_stylo,   X_val_ast])
X_test_hc  = np.hstack([X_test_stylo,  X_test_ast])
X_pub_hc   = np.hstack([X_pub_stylo,   X_pub_ast])

print(f"  Done in {time.time()-t0:.1f}s | stylometry: {stylo_cols} | ast: {ast_cols}")

# Scale each feature set independently
scaler_stylo = StandardScaler()
X_train_stylo = scaler_stylo.fit_transform(X_train_stylo)
X_val_stylo   = scaler_stylo.transform(X_val_stylo)
X_test_stylo  = scaler_stylo.transform(X_test_stylo)
X_pub_stylo   = scaler_stylo.transform(X_pub_stylo)

scaler_ast = StandardScaler()
X_train_ast = scaler_ast.fit_transform(X_train_ast)
X_val_ast   = scaler_ast.transform(X_val_ast)
X_test_ast  = scaler_ast.transform(X_test_ast)
X_pub_ast   = scaler_ast.transform(X_pub_ast)

scaler_hc = StandardScaler()
X_train_hc = scaler_hc.fit_transform(X_train_hc)
X_val_hc   = scaler_hc.transform(X_val_hc)
X_test_hc  = scaler_hc.transform(X_test_hc)
X_pub_hc   = scaler_hc.transform(X_pub_hc)


# ─────────────────────────────────────────────
# 4. Evaluation helper
# ─────────────────────────────────────────────
def evaluate(name, clf, X_val_vec, y_val, X_pub_vec, y_pub):
    val_pred = clf.predict(X_val_vec)
    pub_pred = clf.predict(X_pub_vec)
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
# 6. Feature sets × Classifiers
# ─────────────────────────────────────────────
FEATURE_SETS = {
    "Stylometry":  (X_train_stylo, X_val_stylo, X_test_stylo, X_pub_stylo),
    "Structural":  (X_train_ast,   X_val_ast,   X_test_ast,   X_pub_ast),
    "HandCrafted": (X_train_hc,    X_val_hc,    X_test_hc,    X_pub_hc),
}

CLASSIFIERS = {
    "LogisticRegression": LogisticRegression(max_iter=1000, C=2.0, n_jobs=4),
    "LinearSVC": LinearSVC(max_iter=2000, C=1.0),
    # "SVC": SVC(max_iter=5000, random_state=42),
    "XGBoost": XGBClassifier(
        n_estimators=500, learning_rate=0.1, max_depth=6,
        n_jobs=4, random_state=42, eval_metric="logloss",
        tree_method="hist",
    ),
}

results = []

for feat_name, (Xtr, Xv, Xte, Xpu) in FEATURE_SETS.items():
    print(f"\n{'#'*55}")
    print(f"  Feature set: {feat_name}")
    print(f"{'#'*55}")

    for clf_name, clf_proto in CLASSIFIERS.items():
        clf = clf_proto.__class__(**clf_proto.get_params())

        t1 = time.time()
        print(f"\n  Training {clf_name}...")
        clf.fit(Xtr, y_train)
        train_time = time.time() - t1
        print(f"  Training done in {train_time:.1f}s")

        val_pred, pub_pred = evaluate(
            f"{feat_name} + {clf_name}", clf, Xv, y_val, Xpu, y_pub
        )

        results.append({
            "features":    feat_name,
            "classifier":  clf_name,
            "train_time_s": round(train_time, 1),
            "val_acc":  round(accuracy_score(y_val, val_pred), 4),
            "val_f1":   round(f1_score(y_val, val_pred), 4),
            "pub_acc":  round(accuracy_score(y_pub, pub_pred), 4),
            "pub_f1":   round(f1_score(y_pub, pub_pred), 4),
            "val_report": classification_report(y_val, val_pred, target_names=["Human", "AI"]),
            "pub_report": classification_report(y_pub, pub_pred, target_names=["Human", "AI"]),
        })

        safe_name = f"{feat_name.lower().replace('-','').replace('+','_').replace(' ','_')}_{clf_name.lower()}"

        # submission — unlabeled test predictions
        test_pred = clf.predict(Xte)
        sub_path = os.path.join(run_dir, "submission", f"{safe_name}.csv")
        pd.DataFrame({"ID": test_ids, "label": test_pred}).to_csv(sub_path, index=False)
        print(f"  Saved {sub_path}")

        # predictions — val and public test with true labels
        pd.DataFrame({"true_label": y_val, "pred_label": val_pred}).to_csv(
            os.path.join(run_dir, "predictions", f"val_{safe_name}.csv"), index=False
        )
        pd.DataFrame({"true_label": y_pub, "pred_label": pub_pred}).to_csv(
            os.path.join(run_dir, "predictions", f"pub_{safe_name}.csv"), index=False
        )
        print(f"  Saved predictions for val and public test")


# ─────────────────────────────────────────────
# 7. Summary table
# ─────────────────────────────────────────────
sorted_results = sorted(results, key=lambda x: -x["val_f1"])

print(f"\n\n{'='*80}")
print("  SUMMARY")
print(f"{'='*80}")
print(f"  {'Features':<24} {'Classifier':<22} {'Val Acc':>8} {'Val F1':>8} {'Pub Acc':>8} {'Pub F1':>8}")
print(f"  {'-'*24} {'-'*22} {'-'*8} {'-'*8} {'-'*8} {'-'*8}")
for r in sorted_results:
    print(f"  {r['features']:<24} {r['classifier']:<22} {r['val_acc']:>8.4f} {r['val_f1']:>8.4f} {r['pub_acc']:>8.4f} {r['pub_f1']:>8.4f}")
print(f"{'='*80}\n")


# ─────────────────────────────────────────────
# 8. Markdown report
# ─────────────────────────────────────────────
run_ts = os.path.basename(run_dir)
md_lines = [
    f"# Experiment Report — {run_ts}",
    "",
    "## Configuration",
    f"- **Train samples**: {len(X_train_raw):,}",
    f"- **Validation samples**: {len(X_val_raw):,}",
    f"- **Public test samples**: {len(X_pub_raw):,}",
    f"- **Feature sets**: {', '.join(FEATURE_SETS.keys())}",
    f"- **Classifiers**: {', '.join(CLASSIFIERS.keys())}",
    "",
    "## Results Summary",
    "",
    "| Features | Classifier | Train Time (s) | Val Acc | Val F1 | Pub Acc | Pub F1 |",
    "|---|---|---:|---:|---:|---:|---:|",
]
for r in sorted_results:
    md_lines.append(
        f"| {r['features']} | {r['classifier']} | {r['train_time_s']} "
        f"| {r['val_acc']:.4f} | {r['val_f1']:.4f} | {r['pub_acc']:.4f} | {r['pub_f1']:.4f} |"
    )

md_lines += ["", "---", "", "## Detailed Classification Reports", ""]
for r in sorted_results:
    title = f"{r['features']} + {r['classifier']}"
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
