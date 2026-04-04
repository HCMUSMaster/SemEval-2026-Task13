"""
Text Classification: Human vs AI-generated Code Detection
Models: CountVectorizer + TF-IDF with LogisticRegression, LinearSVC, RandomForest
"""

import os
import time
import warnings
from datetime import datetime
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC, SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
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
# 1. Load Data
# ─────────────────────────────────────────────
DATA_DIR = "Task_A"

print("Loading data...")
train_df = pd.read_parquet(f"{DATA_DIR}/train.parquet")
val_df   = pd.read_parquet(f"{DATA_DIR}/validation.parquet")
test_df  = pd.read_parquet(f"{DATA_DIR}/test.parquet")
pub_df   = pd.read_parquet(f"{DATA_DIR}/test_sample.parquet")

# train_df = train_df.sample(frac=0.5, random_state=42).reset_index(drop=True)
X_train = train_df["code"].fillna("").tolist()
y_train = train_df["label"].tolist()

X_val   = val_df["code"].fillna("").tolist()
y_val   = val_df["label"].tolist()

X_test  = test_df["code"].fillna("").tolist()
test_ids = test_df["ID"].tolist()

X_pub   = pub_df["code"].fillna("").tolist()
y_pub   = pub_df["label"].tolist()

print(f"  Train (100%): {len(X_train):,} | Val: {len(X_val):,} | Test: {len(X_test):,} | Public test: {len(X_pub):,}")


# ─────────────────────────────────────────────
# 2. Evaluation helper
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
    print()
    print(f"\n  Public test classification report:")               
    print(classification_report(y_pub, pub_pred, target_names=["Human", "AI"]))    
    return val_pred, pub_pred


# ─────────────────────────────────────────────
# 3. Pipelines
# ─────────────────────────────────────────────
VECTORIZERS = {
    "CountVectorizer": CountVectorizer(
        max_features=100_000,
        # ngram_range=(1, 2),
        analyzer="char",
        min_df=2,
    ),
    "TF-IDF": TfidfVectorizer(
        max_features=100_000,
        # ngram_range=(1, 2),
        analyzer="char",
        sublinear_tf=True,
        min_df=2,
    ),
}

CLASSIFIERS = {
    # "DecisionTree": DecisionTreeClassifier(random_state=42),
    "LogisticRegression": LogisticRegression(
        max_iter=1000, C=1.0, n_jobs=4
    ),
    # "SVC": SVC(random_state=42,),
    # "RandomForest": RandomForestClassifier(
    #     n_estimators=100, n_jobs=4, random_state=42, max_features="sqrt"
    # ),
    "XGBoost": XGBClassifier(
        n_estimators=300, learning_rate=0.1, max_depth=6,
        n_jobs=4, random_state=42, eval_metric="logloss",
        tree_method="hist",
    ),
}

results = []  # collect dicts with all metrics for report

for vec_name, vectorizer in VECTORIZERS.items():
    print(f"\n{'#'*55}")
    print(f"  Vectorizer: {vec_name}")
    print(f"{'#'*55}")

    t0 = time.time()
    print(f"  Fitting vectorizer on {len(X_train):,} samples...")
    X_train_vec = vectorizer.fit_transform(X_train)
    X_val_vec   = vectorizer.transform(X_val)
    X_pub_vec   = vectorizer.transform(X_pub)
    X_test_vec  = vectorizer.transform(X_test)
    print(f"  Vectorizer done in {time.time()-t0:.1f}s | shape: {X_train_vec.shape}")

    for clf_name, clf_proto in CLASSIFIERS.items():
        # RandomForest with sparse input is fine for sklearn >= 0.24
        # But it's memory-heavy; subsample to 50k if OOM
        clf = clf_proto.__class__(**clf_proto.get_params())

        t1 = time.time()
        print(f"\n  Training {clf_name}...")
        # try:
        clf.fit(X_train_vec, y_train,)
        # except MemoryError:
        #     print(f"  OOM for {clf_name} with full data, subsampling to 50k...")
        #     idx = np.random.choice(len(X_train), 50_000, replace=False)
        #     clf.fit(X_train_vec[idx], np.array(y_train)[idx])
        train_time = time.time() - t1
        print(f"  Training done in {train_time:.1f}s")

        val_pred, pub_pred = evaluate(
            f"{vec_name} + {clf_name}", clf,
            X_val_vec, y_val, X_pub_vec, y_pub
        )

        results.append({
            "vectorizer": vec_name,
            "classifier": clf_name,
            "train_time_s": round(train_time, 1),
            "val_acc":  round(accuracy_score(y_val, val_pred), 4),
            "val_f1":   round(f1_score(y_val, val_pred), 4),
            "pub_acc":  round(accuracy_score(y_pub, pub_pred), 4),
            "pub_f1":   round(f1_score(y_pub, pub_pred), 4),
            "val_report": classification_report(y_val, val_pred, target_names=["Human", "AI"]),
            "pub_report": classification_report(y_pub, pub_pred, target_names=["Human", "AI"]),
        })

        safe_name = f"{vec_name.lower().replace('-','').replace(' ','_')}_{clf_name.lower()}"

        # submission/ — unlabeled test predictions
        test_pred = clf.predict(X_test_vec)
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
# 4. Summary table (console)
# ─────────────────────────────────────────────
sorted_results = sorted(results, key=lambda x: -x["val_f1"])

print(f"\n\n{'='*75}")
print("  SUMMARY")
print(f"{'='*75}")
print(f"  {'Vectorizer':<20} {'Classifier':<22} {'Val Acc':>8} {'Val F1':>8} {'Pub Acc':>8} {'Pub F1':>8}")
print(f"  {'-'*20} {'-'*22} {'-'*8} {'-'*8} {'-'*8} {'-'*8}")
for r in sorted_results:
    print(f"  {r['vectorizer']:<20} {r['classifier']:<22} {r['val_acc']:>8.4f} {r['val_f1']:>8.4f} {r['pub_acc']:>8.4f} {r['pub_f1']:>8.4f}")
print(f"{'='*75}\n")

# ─────────────────────────────────────────────
# 5. Markdown report
# ─────────────────────────────────────────────
run_ts = os.path.basename(run_dir)
md_lines = [
    f"# Experiment Report — {run_ts}",
    "",
    "## Configuration",
    f"- **Train samples**: {len(X_train):,} (20% of full train set)",
    f"- **Validation samples**: {len(X_val):,}",
    f"- **Public test samples**: {len(X_pub):,}",
    f"- **Vectorizers**: {', '.join(VECTORIZERS.keys())}",
    f"- **Classifiers**: {', '.join(CLASSIFIERS.keys())}",
    "",
    "## Results Summary",
    "",
    "| Vectorizer | Classifier | Train Time (s) | Val Acc | Val F1 | Pub Acc | Pub F1 |",
    "|---|---|---:|---:|---:|---:|---:|",
]
for r in sorted_results:
    md_lines.append(
        f"| {r['vectorizer']} | {r['classifier']} | {r['train_time_s']} "
        f"| {r['val_acc']:.4f} | {r['val_f1']:.4f} | {r['pub_acc']:.4f} | {r['pub_f1']:.4f} |"
    )

md_lines += ["", "---", "", "## Detailed Classification Reports", ""]

for r in sorted_results:
    title = f"{r['vectorizer']} + {r['classifier']}"
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
