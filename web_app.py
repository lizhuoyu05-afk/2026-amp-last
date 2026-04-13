from __future__ import annotations

import os
from typing import Any

import requests
from flask import Flask, jsonify, render_template, request

app = Flask(__name__)


REMOTE_PREDICT_URL = os.getenv("REMOTE_PREDICT_URL", "").strip()
REMOTE_TIMEOUT = float(os.getenv("REMOTE_TIMEOUT", "20"))
REMOTE_API_KEY = os.getenv("REMOTE_API_KEY", "").strip()


@app.get("/")
def index() -> str:
    return render_template("index.html", remote_configured=bool(REMOTE_PREDICT_URL))


@app.post("/api/predict")
def predict() -> Any:
    payload = request.get_json(silent=True) or {}
    sequence = str(payload.get("sequence", "")).strip().upper()

    if not sequence:
        return jsonify({"error": "请输入待预测的序列。"}), 400

    if not REMOTE_PREDICT_URL:
        return jsonify(
            {
                "error": (
                    "未配置远程服务地址。请设置环境变量 REMOTE_PREDICT_URL，"
                    "例如 https://your-server/predict"
                )
            }
        ), 500

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
        return jsonify({"error": f"远程服务连接失败: {exc}"}), 502
    except ValueError:
        return jsonify({"error": "远程服务返回了非 JSON 数据。"}), 502

    probability = _extract_probability(remote_data)
    if probability is None:
        return (
            jsonify(
                {
                    "error": "远程服务返回格式不符合预期，未找到 cytotoxicity 概率字段。",
                    "raw": remote_data,
                }
            ),
            502,
        )

    probability = max(0.0, min(1.0, float(probability)))
    verdict = "高危 / Toxic" if probability >= 0.59 else "低危 / Non-toxic"

    return jsonify(
        {
            "sequence": sequence,
            "probability": probability,
            "threshold": 0.59,
            "verdict": verdict,
            "remote_raw": remote_data,
        }
    )


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


def _is_number(value: Any) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool)


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=7860, debug=True)
