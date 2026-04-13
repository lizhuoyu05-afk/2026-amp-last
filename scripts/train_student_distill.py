from __future__ import annotations

import argparse
import os
import json
import pandas as pd
import torch
import torch.nn as nn
from transformers import TrainingArguments, Trainer, set_seed

from src.data import EscapeCSVDataset, Collator, load_tokenizer
from src.modeling import ESMForMultiLabel, LABEL_NAMES

def load_teacher_logits(parquet_path: str) -> pd.DataFrame:
    df = pd.read_parquet(parquet_path)
    if "seq_id" not in df.columns:
        raise ValueError("teacher logits parquet must contain 'seq_id'")
    return df.set_index("seq_id")

class DistillTrainer(Trainer):
    def __init__(self, *args, lambda_soft: float = 0.5, teacher_map=None, **kwargs):
        super().__init__(*args, **kwargs)
        self.lambda_soft = float(lambda_soft)
        self.teacher_map = teacher_map
        self.bce = nn.BCEWithLogitsLoss()

    # 🔥 修复了这里：完全针对单标签任务重写的损失计算，去掉了多头子标签逻辑
    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        labels = inputs.pop("labels")
        seq_ids = inputs.pop("seq_id")
        outputs = model(**inputs, labels=None)
        logits = outputs["logits"]

        # 1. 硬标签损失 (来自 DBAASP 的真实 0 和 1)
        labels = labels.float()
        loss_hard = self.bce(logits, labels)

        # 2. 软标签损失 (模仿 Teacher 大模型输出的精准概率)
        with torch.no_grad():
            zt = torch.stack([self.teacher_map[sid] for sid in seq_ids], dim=0).to(logits.device)
            pt = torch.sigmoid(zt)

        loss_soft = self.bce(logits, pt)
        
        # 3. 综合损失
        loss = (1 - self.lambda_soft) * loss_hard + self.lambda_soft * loss_soft

        return (loss, outputs) if return_outputs else loss

def compute_metrics(eval_pred):
    from sklearn.metrics import average_precision_score, roc_auc_score, matthews_corrcoef
    import numpy as np
    logits, labels = eval_pred
    probs = 1 / (1 + np.exp(-logits))
    y = labels[:, 0]
    p = probs[:, 0]
    metrics = {}
    
    try:
        metrics["pr_auc"] = float(average_precision_score(y, p))
        metrics["roc_auc"] = float(roc_auc_score(y, p))
    except Exception:
        metrics["pr_auc"] = float("nan")
        metrics["roc_auc"] = float("nan")

    best_mcc = -1
    best_thr = 0.5
    for thr in np.linspace(0.1, 0.9, 81):
        pred = (p >= thr).astype(int)
        mcc = matthews_corrcoef(y, pred)
        if mcc > best_mcc:
            best_mcc = mcc
            best_thr = thr

    metrics["best_mcc"] = float(best_mcc)
    metrics["best_thr"] = float(best_thr)
    metrics["mcc@0.5"] = float(matthews_corrcoef(y, (p >= 0.5).astype(int)))
    return metrics

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--student_model_dir", required=True)
    ap.add_argument("--data_dir", required=True)
    ap.add_argument("--train_csv", default="Fold1.csv")
    ap.add_argument("--val_csv", default="Fold2.csv")
    ap.add_argument("--teacher_logits_train", required=True)
    ap.add_argument("--teacher_logits_val", required=True)
    ap.add_argument("--output_dir", required=True)
    ap.add_argument("--max_len", type=int, default=256)
    ap.add_argument("--epochs", type=int, default=20)
    ap.add_argument("--batch_size", type=int, default=32)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--fp16", type=int, default=1)
    ap.add_argument("--lambda_soft", type=float, default=0.5)
    
    # 占位符：兼容你在命令行传入的旧参数，防止报错
    ap.add_argument("--pos_weight_mode", type=str, default="none")
    ap.add_argument("--sampler", type=str, default="none")
    ap.add_argument("--alpha_sub", type=float, default=0.0)
    args = ap.parse_args()

    set_seed(args.seed)

    train_path = os.path.join(args.data_dir, args.train_csv)
    val_path = os.path.join(args.data_dir, args.val_csv)

    tokenizer = load_tokenizer(args.student_model_dir)
    collator = Collator(tokenizer, max_length=args.max_len)

    train_ds = EscapeCSVDataset(train_path)
    val_ds = EscapeCSVDataset(val_path)

    t_train = load_teacher_logits(args.teacher_logits_train)
    t_val = load_teacher_logits(args.teacher_logits_val)
    t_all = pd.concat([t_train, t_val], axis=0)

    def row_to_tensor(row) -> torch.Tensor:
        z = [float(row[f"z_{name}"]) for name in LABEL_NAMES]
        return torch.tensor(z, dtype=torch.float32)

    teacher_map = {sid: row_to_tensor(t_all.loc[sid]) for sid in t_all.index.unique()}

    model = ESMForMultiLabel(args.student_model_dir)

    training_args = TrainingArguments(
        output_dir=args.output_dir,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        num_train_epochs=args.epochs,
        learning_rate=args.lr,
        eval_strategy="epoch",
        save_strategy="epoch",
        save_total_limit=2,
        
        # 🔥 跟 Teacher 一样，让代码自动保存 MCC 最高的一轮
        load_best_model_at_end=True,
        metric_for_best_model="best_mcc",
        greater_is_better=True,
        
        fp16=bool(args.fp16),
        logging_steps=50,
        report_to="none",
        remove_unused_columns=False,
    )

    trainer = DistillTrainer(
        model=model,
        args=training_args,
        train_dataset=train_ds,
        eval_dataset=val_ds,
        processing_class=tokenizer,
        data_collator=collator,
        compute_metrics=compute_metrics,
        lambda_soft=args.lambda_soft,
        teacher_map=teacher_map,
    )

    trainer.train()

    best_dir = os.path.join(args.output_dir, "best")
    os.makedirs(best_dir, exist_ok=True)
    trainer.save_model(best_dir)
    tokenizer.save_pretrained(best_dir)

    with open(os.path.join(args.output_dir, "run_args.json"), "w", encoding="utf-8") as f:
        json.dump(vars(args), f, ensure_ascii=False, indent=2)

    metrics = trainer.evaluate()
    with open(os.path.join(args.output_dir, "val_metrics.json"), "w", encoding="utf-8") as f:
        json.dump(metrics, f, ensure_ascii=False, indent=2)

if __name__ == "__main__":
    main()