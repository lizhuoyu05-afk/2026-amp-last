from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Dict, Optional, Tuple

import torch
import torch.nn as nn
from transformers import AutoConfig, AutoModel

# === 任务改为细胞毒性 ===
NUM_LABELS = 1
LABEL_NAMES = ["Cytotoxicity"]

def mean_pool(last_hidden: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
    mask = attention_mask.unsqueeze(-1).to(last_hidden.dtype)
    summed = (last_hidden * mask).sum(dim=1)
    denom = mask.sum(dim=1).clamp(min=1e-6)
    return summed / denom

class ESMForMultiLabel(nn.Module):
    def __init__(self, model_dir: str, hidden_dropout: float = 0.1):
        super().__init__()
        # === 移除 local_files_only=True ===
        self.config = AutoConfig.from_pretrained(model_dir)
        self.backbone = AutoModel.from_pretrained(model_dir)
        hid = getattr(self.config, "hidden_size", None) or self.backbone.config.hidden_size
        self.dropout = nn.Dropout(hidden_dropout)
        self.classifier = nn.Linear(hid, NUM_LABELS)

    def freeze_backbone(self):
        for p in self.backbone.parameters():
            p.requires_grad = False

    def forward(
        self,
        input_ids: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        inputs_embeds: Optional[torch.Tensor] = None,
        labels: Optional[torch.Tensor] = None,
        **kwargs,
    ):
        kwargs.pop("seq_id", None)
        out = self.backbone(
            input_ids=input_ids,
            attention_mask=attention_mask,
            inputs_embeds=inputs_embeds,
            **kwargs,
        )
        last_hidden = out.last_hidden_state

        if attention_mask is None:
            pad_id = getattr(self.config, "pad_token_id", 1) or 1
            if input_ids is not None:
                attention_mask = (input_ids != pad_id).long()
            else:
                attention_mask = torch.ones(
                    last_hidden.shape[:2], device=last_hidden.device, dtype=torch.long
                )

        pooled = mean_pool(last_hidden, attention_mask)
        logits = self.classifier(self.dropout(pooled))
        loss = None
        if labels is not None:
            # === 撤销 pos_weight，依靠后续的动态阈值搜索来对抗不平衡 ===
            loss_fn = nn.BCEWithLogitsLoss()
            loss = loss_fn(logits, labels)
        return {"loss": loss, "logits": logits}

def is_adapter_checkpoint(model_dir: str) -> bool:
    return os.path.isfile(os.path.join(model_dir, "adapter_config.json"))

def load_model_for_inference(
    ckpt_dir: str,
    base_model_dir: Optional[str] = None,
    merge_lora: bool = True,
) -> nn.Module:
    if is_adapter_checkpoint(ckpt_dir):
        if base_model_dir is None:
            raise ValueError("LoRA adapter detected. Please provide --base_model_dir.")
        model = ESMForMultiLabel(base_model_dir)
        from peft import PeftModel
        model = PeftModel.from_pretrained(model, ckpt_dir)
        if merge_lora and hasattr(model, "merge_and_unload"):
            model = model.merge_and_unload()
        return model

    if os.path.isfile(os.path.join(ckpt_dir, "config.json")):
        return ESMForMultiLabel(ckpt_dir)

    safe_path = os.path.join(ckpt_dir, "model.safetensors")
    bin_path = os.path.join(ckpt_dir, "pytorch_model.bin")
    if base_model_dir and (os.path.isfile(safe_path) or os.path.isfile(bin_path)):
        model = ESMForMultiLabel(base_model_dir)
        if os.path.isfile(safe_path):
            from safetensors.torch import load_file
            state = load_file(safe_path)
        else:
            state = torch.load(bin_path, map_location="cpu")
        model.load_state_dict(state, strict=False)
        return model

    raise ValueError(
        "Checkpoint missing config.json. Provide --base_model_dir to load state_dict."
    )