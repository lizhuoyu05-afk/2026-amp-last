from __future__ import annotations

import argparse
import os
import pandas as pd
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from src.data import EscapeCSVDataset, Collator, load_tokenizer, LABEL_COLS
from src.modeling import LABEL_NAMES, load_model_for_inference

@torch.no_grad()
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt_dir", required=True, help="Folder saved by Trainer.save_model()")
    ap.add_argument("--data_dir", required=True)
    ap.add_argument("--csv", required=True)
    ap.add_argument("--out", required=True, help="Output parquet path")
    ap.add_argument("--base_model_dir", default=None, help="Required if ckpt_dir is a LoRA adapter")
    ap.add_argument("--merge_lora", type=int, default=1, help="Merge LoRA into base weights for inference")
    ap.add_argument("--max_len", type=int, default=256)
    ap.add_argument("--batch_size", type=int, default=32)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = ap.parse_args()

    csv_path = os.path.join(args.data_dir, args.csv)
    ds = EscapeCSVDataset(csv_path)

    tokenizer = load_tokenizer(args.ckpt_dir)
    collator = Collator(tokenizer, max_length=args.max_len)
    dl = DataLoader(ds, batch_size=args.batch_size, shuffle=False, collate_fn=collator)

    model = load_model_for_inference(
        args.ckpt_dir,
        base_model_dir=args.base_model_dir,
        merge_lora=bool(args.merge_lora),
    ).to(args.device)
    model.eval()

    rows = []
    for batch in tqdm(dl, desc="export"):
        input_ids = batch["input_ids"].to(args.device)
        attention_mask = batch["attention_mask"].to(args.device)
        labels = batch["labels"].cpu().numpy()
        out = model(input_ids=input_ids, attention_mask=attention_mask)
        logits = out["logits"].cpu().numpy()
        for sid, y, z in zip(batch["seq_id"], labels, logits):
            row = {"seq_id": sid}
            for i, name in enumerate(LABEL_NAMES):
                row[f"y_{name}"] = float(y[i])
                row[f"z_{name}"] = float(z[i])
            rows.append(row)

    df = pd.DataFrame(rows)
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    df.to_parquet(args.out, index=False)
    print(f"Saved logits to {args.out} shape={df.shape}")

if __name__ == "__main__":
    main()
