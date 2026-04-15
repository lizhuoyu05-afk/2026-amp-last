from __future__ import annotations

import csv
import io
import os
from typing import Any

import torch
from flask import Flask, jsonify, render_template, request
from transformers import AutoTokenizer

from src.modeling import load_model_for_inference

app = Flask(__name__)


THRESHOLD = float(os.getenv("PREDICT_THRESHOLD", "0.59"))
MAX_BATCH_ROWS = int(os.getenv("MAX_BATCH_ROWS", "500"))
MAX_CSV_BYTES = int(os.getenv("MAX_CSV_BYTES", str(2 * 1024 * 1024)))
MAX_SEQ_LEN = int(os.getenv("MAX_SEQ_LEN", "256"))

# 默认走本地推理；如需切回远程可自行改为 remote。
INFER_MODE = os.getenv("INFER_MODE", "local").strip().lower()
LOCAL_CKPT_DIR = os.getenv("LOCAL_CKPT_DIR", "./runs/cytotox_student_35m/best").strip()
LOCAL_BASE_MODEL_DIR = os.getenv("LOCAL_BASE_MODEL_DIR", "facebook/esm2_t12_35M_UR50D").strip()

# 兼容旧配置（remote 模式）
REMOTE_PREDICT_URL = os.getenv("REMOTE_PREDICT_URL", "").strip()
REMOTE_TIMEOUT = float(os.getenv("REMOTE_TIMEOUT", "20"))
REMOTE_API_KEY = os.getenv("REMOTE_API_KEY", "").strip()
THRESHOLD = 0.59
MAX_BATCH_ROWS = int(os.getenv("MAX_BATCH_ROWS", "500"))
MAX_CSV_BYTES = int(os.getenv("MAX_CSV_BYTES", str(2 * 1024 * 1024)))


_TOKENIZER: Any = None
_MODEL: Any = None
_DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")



@app.get("/")
def index() -> str:
    return render_template(
        "index.html",
        remote_configured=bool(REMOTE_PREDICT_URL),
        infer_mode=INFER_MODE,
        local_ckpt_dir=LOCAL_CKPT_DIR,
    )


@app.post("/api/predict")
def predict() -> Any:
    payload = request.get_json(silent=True) or {}
    sequence = _normalize_sequence(payload.get("sequence", ""))

    if not sequence:
        return jsonify({"error": "请输入待预测的序列。"}), 400

    response = _predict_sequence(sequence)
    if "error" in response:
        return jsonify(response), response.get("status", 500)

    return jsonify(response)


@app.post("/api/predict-batch")
def predict_batch() -> Any:
    csv_text, source = _read_csv_text_from_request()
    if csv_text is None:
        return source

    reader = csv.DictReader(io.StringIO(csv_text))
    if not reader.fieldnames:
        return jsonify({"error": "CSV 缺少表头。"}), 400

    seq_column = _find_sequence_column(reader.fieldnames)
    if not seq_column:
        return jsonify(
            {
                "error": "未找到序列列，请使用 Sequence / sequence / seq / peptide 作为列名。",
                "columns": reader.fieldnames,
            }
        ), 400

    rows = list(reader)
    if not rows:
        return jsonify({"error": "CSV 没有数据行。"}), 400

    if len(rows) > MAX_BATCH_ROWS:
        return jsonify(
            {
                "error": f"CSV 行数超过限制（最多 {MAX_BATCH_ROWS} 行）。",
                "max_rows": MAX_BATCH_ROWS,
                "actual_rows": len(rows),
            }
        ), 400

    results: list[dict[str, Any]] = []
    success_count = 0
    sequence_cache: dict[str, dict[str, Any]] = {}

    for idx, row in enumerate(rows, start=1):
        sequence = _normalize_sequence(row.get(seq_column, ""))
        if not sequence:
            results.append(
                {
                    "row": idx,
                    "sequence": "",
                    "probability": None,
                    "verdict": "",
                    "status": "error",
                    "error": "该行序列为空",
                }
            )
            continue

        pred = sequence_cache.get(sequence)
        if pred is None:
            pred = _predict_sequence(sequence)
            sequence_cache[sequence] = pred

        if "error" in pred:
            results.append(
                {
                    "row": idx,
                    "sequence": sequence,
                    "probability": None,
                    "verdict": "",
                    "status": "error",
                    "error": pred["error"],
                }
            )
            continue

        success_count += 1
        results.append(
            {
                "row": idx,
                "sequence": sequence,
                "probability": pred["probability"],
                "verdict": pred["verdict"],
                "status": "ok",
                "error": "",
            }
        )

    sorted_results = sorted(results, key=_batch_sort_key)

    return jsonify(
        {
            "source": source,

            "total": len(results),

            "total": len(sorted_results),

            "success": success_count,
            "failed": len(sorted_results) - success_count,
            "threshold": THRESHOLD,
            "sequence_column": seq_column,
            "unique_sequences": len(sequence_cache),

            "results": results,

            "sorted_by": "status_ok_then_probability_desc_then_row_asc",
            "results": sorted_results,

        }
    )


def _read_csv_text_from_request() -> tuple[str | None, tuple[Any, int] | str]:
    file = request.files.get("file")
    if file is not None:
        raw = file.stream.read(MAX_CSV_BYTES + 1)
        if len(raw) > MAX_CSV_BYTES:
            return None, (
                jsonify(
                    {
                        "error": f"CSV 文件过大（最多 {MAX_CSV_BYTES} bytes）。",
                        "max_bytes": MAX_CSV_BYTES,
                    }
                ),
                400,
            )

        try:
            return raw.decode("utf-8-sig"), "multipart"
        except UnicodeDecodeError:
            return None, (jsonify({"error": "CSV 编码错误，请使用 UTF-8。"}), 400)

    payload = request.get_json(silent=True) or {}
    csv_text = payload.get("csv_text")
    if isinstance(csv_text, str) and csv_text.strip():
        if len(csv_text.encode("utf-8")) > MAX_CSV_BYTES:
            return None, (
                jsonify(
                    {
                        "error": f"CSV 文本过大（最多 {MAX_CSV_BYTES} bytes）。",
                        "max_bytes": MAX_CSV_BYTES,
                    }
                ),
                400,
            )
        return csv_text, "json"

    return None, (jsonify({"error": "请上传 CSV 文件（file 字段）。"}), 400)


def _predict_sequence(sequence: str) -> dict[str, Any]:
    if INFER_MODE == "local":
        return _predict_sequence_local(sequence)
    return _predict_sequence_remote(sequence)


def _predict_sequence_local(sequence: str) -> dict[str, Any]:
    try:
        tokenizer, model = _get_local_predictor()
    except Exception as exc:  # noqa: BLE001
        return {
            "error": f"本地模型加载失败: {exc}",
            "status": 500,
            "mode": "local",
        }

    try:
        inputs = tokenizer(
            [sequence],
            padding=True,
            truncation=True,
            max_length=MAX_SEQ_LEN,
            return_tensors="pt",
        )
        inputs = {k: v.to(_DEVICE) for k, v in inputs.items()}

        with torch.no_grad():
            logits = model(**inputs)["logits"]
            prob = float(torch.sigmoid(logits).squeeze().detach().cpu().item())
    except Exception as exc:  # noqa: BLE001
        return {
            "error": f"本地推理失败: {exc}",
            "status": 500,
            "mode": "local",
        }

    probability = max(0.0, min(1.0, prob))
    verdict = "高危 / Toxic" if probability >= THRESHOLD else "低危 / Non-toxic"
    return {
        "sequence": sequence,
        "probability": probability,
        "threshold": THRESHOLD,
        "verdict": verdict,
        "mode": "local",
    }


def _get_local_predictor() -> tuple[Any, Any]:
    global _TOKENIZER, _MODEL
    if _TOKENIZER is not None and _MODEL is not None:
        return _TOKENIZER, _MODEL

    ckpt_dir = os.path.abspath(LOCAL_CKPT_DIR)
    if not os.path.isdir(ckpt_dir):
        raise FileNotFoundError(
            f"LOCAL_CKPT_DIR 不存在: {ckpt_dir}。请设置本地模型路径。"
        )

    _TOKENIZER = AutoTokenizer.from_pretrained(ckpt_dir)
    _MODEL = load_model_for_inference(
        ckpt_dir=ckpt_dir,
        base_model_dir=LOCAL_BASE_MODEL_DIR or None,
        merge_lora=True,
    )
    _MODEL.to(_DEVICE)
    _MODEL.eval()
    return _TOKENIZER, _MODEL


def _predict_sequence_remote(sequence: str) -> dict[str, Any]:
    import requests

    if not REMOTE_PREDICT_URL:
        return {
            "error": "未配置远程服务地址。请设置 REMOTE_PREDICT_URL，或将 INFER_MODE=local。",
            "status": 500,
            "mode": "remote",
        }

    remote_payload = {"sequence": sequence}
    headers = {"Content-Type": "application/json"}
    if REMOTE_API_KEY:
        headers["Authorization"] = f"Bearer {REMOTE_API_KEY}"

    try:
        response = requests.post(
            REMOTE_PREDICT_URL,
            json=remote_payload,
            headers=headers,
            timeout=REMOTE_TIMEOUT,
        )
        response.raise_for_status()
        remote_data = response.json()
    except requests.RequestException as exc:
        return {"error": f"远程服务连接失败: {exc}", "status": 502, "mode": "remote"}
    except ValueError:
        return {"error": "远程服务返回了非 JSON 数据。", "status": 502, "mode": "remote"}

    probability = _extract_probability(remote_data)
    if probability is None:
        return {
            "error": "远程服务返回格式不符合预期，未找到 cytotoxicity 概率字段。",
            "status": 502,
            "mode": "remote",
        }

    probability = max(0.0, min(1.0, float(probability)))
    verdict = "高危 / Toxic" if probability >= THRESHOLD else "低危 / Non-toxic"
    return {
        "sequence": sequence,
        "probability": probability,
        "threshold": THRESHOLD,
        "verdict": verdict,
        "remote_raw": remote_data,
        "mode": "remote",
    }


def _find_sequence_column(columns: list[str]) -> str | None:
    lookup = {col.lower().strip(): col for col in columns}
    for candidate in ["sequence", "seq", "peptide", "sequence_aa"]:
        if candidate in lookup:
            return lookup[candidate]
    return None


def _extract_probability(data: Any) -> float | None:
    candidates = ["probability", "p", "score", "cytotoxicity", "toxic_prob"]
    if isinstance(data, dict):
        for key in candidates:
            value = data.get(key)
            if _is_number(value):
                return float(value)

        nested_keys = ["result", "data", "prediction", "pred"]
        for key in nested_keys:
            nested_value = data.get(key)
            if nested_value is not None:
                found = _extract_probability(nested_value)
                if found is not None:
                    return found

    if isinstance(data, list):
        for item in data:
            found = _extract_probability(item)
            if found is not None:
                return found

    if _is_number(data):
        return float(data)

    return None


def _normalize_sequence(value: Any) -> str:
    return str(value).strip().upper()



def _batch_sort_key(item: dict[str, Any]) -> tuple[int, float, int]:
    status_rank = 0 if item.get("status") == "ok" else 1
    probability = item.get("probability")
    if probability is None:
        probability = -1.0
    return status_rank, -float(probability), int(item.get("row", 0))


def _is_number(value: Any) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool)


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=7860, debug=True)
