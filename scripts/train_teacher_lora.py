from __future__ import annotations

import argparse
import os
import json
from typing import Dict

import numpy as np
import torch
from transformers import TrainingArguments, Trainer, set_seed

from src.data import EscapeCSVDataset, Collator, load_tokenizer, LABEL_COLS
from src.modeling import ESMForMultiLabel

def compute_metrics(eval_pred):
    from sklearn.metrics import average_precision_score, roc_auc_score, matthews_corrcoef
    import numpy as np

    logits, labels = eval_pred
    probs = 1 / (1 + np.exp(-logits))
    metrics = {}
    
    y = labels[:, 0]
    p = probs[:, 0]
    
    try:
        metrics["pr_auc"] = float(average_precision_score(y, p))
        metrics["roc_auc"] = float(roc_auc_score(y, p))
    except Exception:
        metrics["pr_auc"] = float("nan")
        metrics["roc_auc"] = float("nan")

    # 🔥 核心提分点：动态搜索最佳 MCC 阈值
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

def maybe_apply_lora(model, use_lora: bool, r: int, alpha: int, dropout: float):
    if not use_lora:
        return model
    try:
        from peft import LoraConfig, get_peft_model, TaskType
    except Exception as e:
        raise SystemExit("peft is required for LoRA. pip install peft") from e

    # 🔥 核心提分点：火力全开，把 key 和 dense(FFN) 也加入微调
    lora_cfg = LoraConfig(
        r=r,
        lora_alpha=alpha,
        lora_dropout=dropout,
        bias="none",
        task_type=TaskType.FEATURE_EXTRACTION,
        target_modules=["query", "key", "value", "dense"],
        modules_to_save=["classifier"],
    )
    model = get_peft_model(model, lora_cfg)
    model.print_trainable_parameters()
    return model

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model_dir", required=True)
    ap.add_argument("--data_dir", required=True)
    ap.add_argument("--train_csv", default="Fold1.csv")
    ap.add_argument("--val_csv", default="Fold2.csv")
    ap.add_argument("--output_dir", required=True)
    ap.add_argument("--max_len", type=int, default=256)
    ap.add_argument("--epochs", type=int, default=5)
    ap.add_argument("--batch_size", type=int, default=8)
    ap.add_argument("--lr", type=float, default=2e-4)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--fp16", type=int, default=1)
    ap.add_argument("--freeze_backbone", type=int, default=0)
    # LoRA
    ap.add_argument("--use_lora", type=int, default=1)
    ap.add_argument("--lora_r", type=int, default=8)
    ap.add_argument("--lora_alpha", type=int, default=16)
    ap.add_argument("--lora_dropout", type=float, default=0.05)
    ap.add_argument("--grad_ckpt", type=int, default=0)
    ap.add_argument("--merge_lora", type=int, default=0, help="Merge LoRA into base weights before saving")
    args = ap.parse_args()

    set_seed(args.seed)
    train_path = os.path.join(args.data_dir, args.train_csv)
    val_path = os.path.join(args.data_dir, args.val_csv)

    tokenizer = load_tokenizer(args.model_dir)
    collator = Collator(tokenizer, max_length=args.max_len)

    train_ds = EscapeCSVDataset(train_path)
    val_ds = EscapeCSVDataset(val_path)

    model = ESMForMultiLabel(args.model_dir)
    if args.freeze_backbone:
        model.freeze_backbone()

    if args.grad_ckpt and hasattr(model.backbone, "gradient_checkpointing_enable"):
        model.backbone.gradient_checkpointing_enable()

    model = maybe_apply_lora(model, bool(args.use_lora), args.lora_r, args.lora_alpha, args.lora_dropout)

    training_args = TrainingArguments(
        output_dir=args.output_dir,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        num_train_epochs=args.epochs,
        learning_rate=args.lr,
        eval_strategy="epoch",  
        save_strategy="epoch",
        save_total_limit=2,
        
        # 🔥 核心提分点：早停保存巅峰时刻的权重
        load_best_model_at_end=True,         
        metric_for_best_model="best_mcc",    
        greater_is_better=True,              
        
        fp16=bool(args.fp16),
        logging_steps=50,
        report_to="none",
        remove_unused_columns=False,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_ds,
        eval_dataset=val_ds,
        processing_class=tokenizer, 
        data_collator=collator,
        compute_metrics=compute_metrics,
    )

    trainer.train()

    if args.merge_lora and hasattr(trainer.model, "merge_and_unload"):
        trainer.model = trainer.model.merge_and_unload()

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