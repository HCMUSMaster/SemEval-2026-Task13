"""
Text Classification: Human vs AI-generated Code Detection
Fine-tuning approach: head-only | LoRA | full fine-tuning
Multi-GPU: run with  accelerate launch Finetune_ML.py
                  or torchrun --nproc_per_node=N Finetune_ML.py
"""

import logging
import os
import time
import warnings
from datetime import datetime

import numpy as np
import pandas as pd
import torch
from datasets import Dataset
from peft import LoraConfig, TaskType, get_peft_model
from sklearn.metrics import accuracy_score, classification_report, f1_score
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    DataCollatorWithPadding,
    EarlyStoppingCallback,
    Trainer,
    TrainingArguments,
    set_seed,
)

warnings.filterwarnings("ignore")

# ─────────────────────────────────────────────
# 0. Logging
# ─────────────────────────────────────────────
logging.basicConfig(
    format="%(asctime)s | %(levelname)s | %(message)s",
    datefmt="%H:%M:%S",
    level=logging.INFO,
)
logger = logging.getLogger(__name__)

# ─────────────────────────────────────────────
# 1. Configuration  ← edit here
# ─────────────────────────────────────────────
DATA_DIR       = "Task_A"
MAX_LENGTH     = 512
SEED           = 42
NUM_EPOCHS     = 3
BATCH_SIZE     = 32          # per-device; effective = BATCH_SIZE × n_gpus × GRAD_ACCUM
GRAD_ACCUM     = 2
LR             = 2e-5
WEIGHT_DECAY   = 0.01
WARMUP_RATIO   = 0.06
MAX_TRAIN      = 50_000        # None = full 500K; set e.g. 50_000 for quick runs
MAX_VAL      = 5_000        # None = full 500K; set e.g. 50_000 for quick runs

MODELS = [
    # {"name": "CodeBERT",   "model_id": "microsoft/codebert-base"},
    {"name": "UniXcoder",  "model_id": "microsoft/unixcoder-base"},
]

# Strategies to run (subset or all)
STRATEGIES = [
    "head_only", 
    # "lora", 
    # "full"
]

# LoRA config (used by "lora" strategy)
LORA_R          = 128
LORA_ALPHA      = 256
LORA_DROPOUT    = 0.1
LORA_MODULES    = ["query", "value"]   # attention projection layers

# ─────────────────────────────────────────────
# 2. Output dir
# ─────────────────────────────────────────────
run_dir = os.path.join("log", datetime.now().strftime("%Y%m%d_%H%M%S"))
os.makedirs(run_dir, exist_ok=True)
logger.info(f"Outputs → {run_dir}/")

# ─────────────────────────────────────────────
# 3. Load data
# ─────────────────────────────────────────────
logger.info("Loading data…")
train_df = pd.read_parquet(f"{DATA_DIR}/train.parquet")
val_df   = pd.read_parquet(f"{DATA_DIR}/validation.parquet")
test_df  = pd.read_parquet(f"{DATA_DIR}/test.parquet")
pub_df   = pd.read_parquet(f"{DATA_DIR}/test_sample.parquet")

if MAX_TRAIN is not None:
    train_df = train_df.sample(n=MAX_TRAIN, random_state=SEED).reset_index(drop=True)

if MAX_VAL is not None:
    val_df = val_df.sample(n=MAX_VAL, random_state=SEED).reset_index(drop=True)

train_df["code"] = train_df["code"].fillna("")
val_df["code"]   = val_df["code"].fillna("")
test_df["code"]  = test_df["code"].fillna("")
pub_df["code"]   = pub_df["code"].fillna("")

logger.info(f"  Train {len(train_df):,} | Val {len(val_df):,} | "
            f"Test {len(test_df):,} | Pub {len(pub_df):,}")

# ─────────────────────────────────────────────
# 4. Helpers
# ─────────────────────────────────────────────
USE_FP16 = torch.cuda.is_available()


def make_hf_dataset(df: pd.DataFrame, has_label: bool = True) -> Dataset:
    cols = {"text": df["code"].tolist()}
    if has_label:
        cols["labels"] = df["label"].astype(int).tolist()
    return Dataset.from_dict(cols)


def tokenize_fn(batch, tokenizer):
    return tokenizer(
        batch["text"],
        max_length=MAX_LENGTH,
        truncation=True,
        padding=False,           # DataCollatorWithPadding handles dynamic padding
    )


def apply_strategy(model, strategy: str):
    """Modify model in-place for the chosen strategy."""
    if strategy == "head_only":
        # Freeze everything except the classifier head
        for name, param in model.named_parameters():
            if "classifier" not in name and "pooler" not in name:
                param.requires_grad = False
    elif strategy == "lora":
        lora_cfg = LoraConfig(
            task_type=TaskType.SEQ_CLS,
            r=LORA_R,
            lora_alpha=LORA_ALPHA,
            lora_dropout=LORA_DROPOUT,
            target_modules=LORA_MODULES,
            bias="none",
        )
        model = get_peft_model(model, lora_cfg)
        model.print_trainable_parameters()
    # "full" → nothing to do
    return model


def trainable_params(model) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def compute_metrics(eval_pred):
    logits, labels = eval_pred
    preds = np.argmax(logits, axis=1)
    return {
        "accuracy": float(accuracy_score(labels, preds)),
        "f1":       float(f1_score(labels, preds)),
    }


def predict(trainer: Trainer, dataset: Dataset) -> np.ndarray:
    out = trainer.predict(dataset)
    return np.argmax(out.predictions, axis=1)


# ─────────────────────────────────────────────
# 5. Main experiment loop
# ─────────────────────────────────────────────
set_seed(SEED)
results = []

for model_cfg in MODELS:
    model_name = model_cfg["name"]
    model_id   = model_cfg["model_id"]

    logger.info(f"\n{'#'*60}")
    logger.info(f"  Model: {model_name}  ({model_id})")
    logger.info(f"{'#'*60}")

    # Tokenizer (shared across strategies for this backbone)
    tokenizer = AutoTokenizer.from_pretrained(model_id)

    # Tokenised splits (reused across strategies)
    def _tok(ds):
        return ds.map(lambda b: tokenize_fn(b, tokenizer),
                      batched=True, remove_columns=["text"])

    tok_train = _tok(make_hf_dataset(train_df))
    tok_val   = _tok(make_hf_dataset(val_df))
    tok_test  = _tok(make_hf_dataset(test_df,  has_label=False))
    tok_pub   = _tok(make_hf_dataset(pub_df))

    collator = DataCollatorWithPadding(tokenizer)

    for strategy in STRATEGIES:
        tag = f"{model_name}_{strategy}"
        logger.info(f"\n{'='*60}")
        logger.info(f"  Strategy: {strategy}   [{tag}]")
        logger.info(f"{'='*60}")

        # Per-run output dirs
        ckpt_dir = os.path.join(run_dir, tag, "checkpoint")
        sub_dir  = os.path.join(run_dir, tag, "submission")
        pred_dir = os.path.join(run_dir, tag, "predictions")
        for d in (ckpt_dir, sub_dir, pred_dir):
            os.makedirs(d, exist_ok=True)

        # Fresh model for each strategy
        model = AutoModelForSequenceClassification.from_pretrained(
            model_id, num_labels=2, ignore_mismatched_sizes=True
        )
        model = apply_strategy(model, strategy)
        logger.info(f"  Trainable params: {trainable_params(model):,}")

        training_args = TrainingArguments(
            output_dir=ckpt_dir,
            num_train_epochs=NUM_EPOCHS,
            per_device_train_batch_size=BATCH_SIZE,
            per_device_eval_batch_size=BATCH_SIZE * 2,
            gradient_accumulation_steps=GRAD_ACCUM,
            learning_rate=LR,
            weight_decay=WEIGHT_DECAY,
            warmup_ratio=WARMUP_RATIO,
            fp16=USE_FP16,
            eval_strategy="epoch",
            save_strategy="epoch",
            load_best_model_at_end=True,
            metric_for_best_model="f1",
            greater_is_better=True,
            logging_steps=50,
            report_to="comet_ml",
            seed=SEED,
            dataloader_num_workers=4,
            remove_unused_columns=True,
        )

        trainer = Trainer(
            model=model,
            args=training_args,
            train_dataset=tok_train,
            eval_dataset=tok_val,
            compute_metrics=compute_metrics,
            processing_class=tokenizer,
            data_collator=collator,
            callbacks=[EarlyStoppingCallback(early_stopping_patience=5)],
        )

        t0 = time.time()
        train_result = trainer.train()
        train_time = time.time() - t0
        logger.info(f"  Training finished in {train_time:.1f}s")

        # ── Evaluate on val & pub ──────────────────────────────
        val_preds = predict(trainer, tok_val)
        pub_preds = predict(trainer, tok_pub)
        y_val = val_df["label"].tolist()
        y_pub = pub_df["label"].tolist()

        val_acc = accuracy_score(y_val, val_preds)
        val_f1  = f1_score(y_val, val_preds)
        pub_acc = accuracy_score(y_pub, pub_preds)
        pub_f1  = f1_score(y_pub, pub_preds)
        val_report = classification_report(y_val, val_preds, target_names=["Human", "AI"])
        pub_report = classification_report(y_pub, pub_preds, target_names=["Human", "AI"])

        logger.info(f"  [Val]  Acc {val_acc:.4f}  F1 {val_f1:.4f}")
        logger.info(f"  [Pub]  Acc {pub_acc:.4f}  F1 {pub_f1:.4f}")

        # ── Submission (unlabeled test) ────────────────────────
        # Only main process writes in multi-GPU setups (trainer.is_world_process_zero())
        if trainer.is_world_process_zero():
            test_preds = predict(trainer, tok_test)
            test_ids   = test_df["ID"].tolist()

            pd.DataFrame({"ID": test_ids, "label": test_preds}).to_csv(
                os.path.join(sub_dir, "submission.csv"), index=False
            )
            pd.DataFrame({"true_label": y_val, "pred_label": val_preds}).to_csv(
                os.path.join(pred_dir, "val.csv"), index=False
            )
            pd.DataFrame({"true_label": y_pub, "pred_label": pub_preds}).to_csv(
                os.path.join(pred_dir, "pub.csv"), index=False
            )
            logger.info(f"  Saved outputs → {os.path.join(run_dir, tag)}/")

        results.append({
            "model":      model_name,
            "strategy":   strategy,
            "train_time": round(train_time, 1),
            "val_acc":    round(val_acc, 4),
            "val_f1":     round(val_f1, 4),
            "pub_acc":    round(pub_acc, 4),
            "pub_f1":     round(pub_f1, 4),
            "val_report": val_report,
            "pub_report": pub_report,
        })

        # Free GPU memory before next strategy
        del model, trainer
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

# ─────────────────────────────────────────────
# 6. Summary table
# ─────────────────────────────────────────────
sorted_results = sorted(results, key=lambda x: -x["val_f1"])

print(f"\n\n{'='*75}")
print("  SUMMARY")
print(f"{'='*75}")
print(f"  {'Model':<14} {'Strategy':<12} {'Val Acc':>8} {'Val F1':>8} {'Pub Acc':>8} {'Pub F1':>8}")
print(f"  {'-'*14} {'-'*12} {'-'*8} {'-'*8} {'-'*8} {'-'*8}")
for r in sorted_results:
    print(f"  {r['model']:<14} {r['strategy']:<12} "
          f"{r['val_acc']:>8.4f} {r['val_f1']:>8.4f} "
          f"{r['pub_acc']:>8.4f} {r['pub_f1']:>8.4f}")
print(f"{'='*75}\n")

# ─────────────────────────────────────────────
# 7. Markdown report
# ─────────────────────────────────────────────
run_ts = os.path.basename(run_dir)
md = [
    f"# Fine-tuning Report — {run_ts}",
    "",
    "## Configuration",
    f"- **Models**: {', '.join(m['name'] for m in MODELS)}",
    f"- **Strategies**: {', '.join(STRATEGIES)}",
    f"- **Max sequence length**: {MAX_LENGTH}",
    f"- **Epochs**: {NUM_EPOCHS}  |  **Batch size (per device)**: {BATCH_SIZE}  |  **Grad accum**: {GRAD_ACCUM}",
    f"- **Learning rate**: {LR}  |  **Weight decay**: {WEIGHT_DECAY}",
    f"- **Train samples**: {len(train_df):,}{' (subsampled)' if MAX_TRAIN else ' (full)'}",
    f"- **Val samples**: {len(val_df):,}  |  **Pub test samples**: {len(pub_df):,}",
    "",
    "## Results",
    "",
    "| Model | Strategy | Train Time (s) | Val Acc | Val F1 | Pub Acc | Pub F1 |",
    "|---|---|---:|---:|---:|---:|---:|",
]

for r in sorted_results:
    md.append(
        f"| {r['model']} | {r['strategy']} | {r['train_time']} "
        f"| {r['val_acc']:.4f} | {r['val_f1']:.4f} "
        f"| {r['pub_acc']:.4f} | {r['pub_f1']:.4f} |"
    )

md += ["", "---", "", "## Detailed Classification Reports", ""]
for r in sorted_results:
    md += [
        f"### {r['model']} — {r['strategy']}",
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
    f.write("\n".join(md))
logger.info(f"Report → {report_path}")
