"""Real PyTorch backend for the Fall 2025 CS188 Project 5 companion website.

The official submission remains models.py, losses.py and train.py. The web app
never reports model predictions from randomly initialized Q3/Q4/Q5 classifiers.
"""
from __future__ import annotations

import json
import math
import os
import sys
from pathlib import Path
from typing import Any

import numpy as np
import torch
from fastapi import FastAPI, HTTPException
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import models  # noqa: E402
import train  # noqa: E402

os.environ.setdefault("OMP_NUM_THREADS", "1")
torch.set_num_threads(1)

app = FastAPI(title="CS188 Project 5 — Nhóm 9", version="1.0")
_cache: dict[str, torch.nn.Module] = {}
_regression_model: torch.nn.Module | None = None


class Pixels(BaseModel):
    pixels: list[float] = Field(min_length=784, max_length=784)


class Word(BaseModel):
    word: str = Field(min_length=1, max_length=32)


class ConvInput(BaseModel):
    image: list[list[float]]
    kernel: list[list[float]]


class AttentionInput(BaseModel):
    tokens: list[list[float]]


def _valid_numbers(values: list[float], lo: float = -1e9, hi: float = 1e9) -> bool:
    return all(math.isfinite(v) and lo <= v <= hi for v in values)


def _checkpoint(question: str):
    return ROOT / "checkpoints" / f"{question}.pt"


def _load_model(question: str):
    path = _checkpoint(question)
    if not path.is_file():
        raise HTTPException(409, detail=f"Chưa có checkpoints/{question}.pt. Hãy huấn luyện và xuất checkpoint trước khi dự đoán.")
    if question not in _cache:
        classes = {
            "q3": models.DigitClassificationModel,
            "q4": models.LanguageIDModel,
            "q5": models.DigitConvolutionalModel,
        }
        model = classes[question]()
        try:
            state_dict = torch.load(path, map_location="cpu", weights_only=True)
            model.load_state_dict(state_dict, strict=True)
        except Exception as exc:
            raise HTTPException(409, detail=f"Checkpoint {question} không tương thích với code hiện tại: {type(exc).__name__}") from exc
        model.eval()
        _cache[question] = model
    return _cache[question]


def _status():
    data_lang = ROOT / "data" / "lang_id.npz"
    vocab_meta = ROOT / "checkpoints" / "q4_chars.json"
    return {
        "backend": True,
        "q1": True,
        "q2": True,
        "q3": _checkpoint("q3").is_file(),
        "q4": _checkpoint("q4").is_file() and (data_lang.is_file() or vocab_meta.is_file()),
        "q5": _checkpoint("q5").is_file(),
        "q6": True,
        "q4_vocab": data_lang.is_file() or vocab_meta.is_file(),
        "trained_on_official_data": False,  # Never inferred solely from file presence.
        "note": "Q3/Q4/Q5 chỉ dự đoán khi có checkpoint phù hợp; website không lưu trọng số đã huấn luyện.",
    }


@app.get("/api/status")
def status():
    return _status()


class TinyDataset(torch.utils.data.Dataset):
    def __init__(self, xs, ys):
        self.xs = torch.as_tensor(xs, dtype=torch.float32)
        self.ys = torch.as_tensor(ys, dtype=torch.float32)

    def __len__(self):
        return len(self.xs)

    def __getitem__(self, i):
        return {"x": self.xs[i], "label": self.ys[i]}


@app.post("/api/q1")
def q1():
    # Small separable toy dataset; actual group's PerceptronModel + train loop.
    coords = [(-2.2, -1.8), (-1.8, 0.1), (-1.4, -0.8), (-0.7, -2.1),
              (-2.1, 1.0), (-0.5, -1.0), (1.6, 2.0), (1.7, 0.7),
              (0.3, 2.3), (2.0, -0.2), (0.2, 1.7), (2.4, 1.5)]
    # Labels from x + y > 0; margin avoids label ambiguity.
    labels = [1 if x + y > 0 else -1 for x, y in coords]
    xs = [[x, y, 1] for x, y in coords]
    model = models.PerceptronModel(3)
    before = model.get_weights().detach().flatten().tolist()
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(188)
        train.train_perceptron(model, TinyDataset(xs, labels))
    after = model.get_weights().detach().flatten().tolist()
    with torch.no_grad():
        scores = model(torch.tensor(xs, dtype=torch.float32)).flatten()
        predicted = torch.where(scores >= 0, 1, -1).tolist()
    return {
        "samples": [{"x": x, "y": y, "label": label} for (x, y), label in zip(coords, labels)],
        "initial_weights": before,
        "trained_weights": after,
        "accuracy": sum(p == y for p, y in zip(predicted, labels)) / len(labels),
        "training": "train.train_perceptron trên 12 điểm minh họa, không phải dữ liệu chấm Berkeley",
    }


@app.post("/api/q2/train")
def q2_train():
    global _regression_model
    torch.manual_seed(188)
    model = models.RegressionModel()
    xs = torch.linspace(-2 * math.pi, 2 * math.pi, 200, dtype=torch.float32).view(-1, 1)
    ys = torch.sin(xs)
    train.train_regression(model, TinyDataset(xs, ys))
    model.eval()
    with torch.no_grad():
        loss = torch.mean((model(xs) - ys) ** 2).item()
    _regression_model = model
    return {"mse": loss, "trained": True, "note": "MSE trên 200 điểm sin(x) mẫu, KHÔNG phải điểm autograder Berkeley"}


@app.get("/api/q2/curve")
def q2_curve():
    if _regression_model is None:
        raise HTTPException(409, detail="Chưa huấn luyện Q2. Hãy chọn 'Huấn luyện demo' trước.")
    x = torch.linspace(-2 * math.pi, 2 * math.pi, 121, dtype=torch.float32).view(-1, 1)
    with torch.no_grad():
        pred = _regression_model(x).view(-1).tolist()
    return {"x": x.view(-1).tolist(), "prediction": pred, "target": torch.sin(x).view(-1).tolist()}


def _validate_pixels(pixels):
    if not _valid_numbers(pixels, 0, 1):
        raise HTTPException(422, detail="Ảnh phải gồm đúng 784 giá trị trong [0,1].")
    return torch.tensor([pixels], dtype=torch.float32)


@app.post("/api/q3")
def q3(data: Pixels):
    model = _load_model("q3")
    with torch.no_grad():
        logits = model(_validate_pixels(data.pixels))
        probs = torch.softmax(logits, dim=1)[0].tolist()
    return {"label": int(np.argmax(probs)), "probabilities": probs, "source": "DigitClassificationModel với checkpoint q3.pt"}


def _alphabet():
    data_path = ROOT / "data" / "lang_id.npz"
    meta_path = ROOT / "checkpoints" / "q4_chars.json"
    if data_path.exists():
        with np.load(data_path, allow_pickle=False) as data:
            return [str(ch) for ch in data["chars"].tolist()]
    if meta_path.exists():
        chars = json.loads(meta_path.read_text(encoding="utf-8"))
        if isinstance(chars, list) and len(chars) == 47:
            return [str(ch) for ch in chars]
    raise HTTPException(409, detail="Thiếu bảng chữ cái của Q4 (data/lang_id.npz hoặc checkpoints/q4_chars.json).")


@app.post("/api/q4")
def q4(data: Word):
    model = _load_model("q4")
    alphabet = _alphabet()
    index = {c: i for i, c in enumerate(alphabet)}
    word = data.word.strip().lower()
    if not word:
        raise HTTPException(422, detail="Hãy nhập một từ để nhận dạng ngôn ngữ.")
    unknown = sorted(set(word) - set(index))
    if unknown:
        raise HTTPException(422, detail="Ký tự ngoài bảng chữ cái của bộ dữ liệu: " + ", ".join(unknown))
    xs = torch.stack([torch.nn.functional.one_hot(torch.tensor(index[ch]), num_classes=47).float() for ch in word]).unsqueeze(1)
    with torch.no_grad():
        logits = model(xs)
        probs = torch.softmax(logits, dim=1)[0].tolist()
    return {"language": model.languages[int(np.argmax(probs))], "probabilities": probs, "languages": model.languages, "source": "LanguageIDModel với checkpoint q4.pt"}


@app.post("/api/q5/convolve")
def q5_convolve(data: ConvInput):
    # Square demo supports 2..12 image and 1..5 kernel, all checked carefully.
    rows = data.image
    kernel = data.kernel
    if not (2 <= len(rows) <= 12 and rows and 2 <= len(rows[0]) <= 12 and all(len(r) == len(rows[0]) for r in rows)):
        raise HTTPException(422, detail="Ảnh 2D phải là ma trận chữ nhật 2–12 hàng/cột.")
    if not (1 <= len(kernel) <= 5 and kernel and 1 <= len(kernel[0]) <= 5 and all(len(r) == len(kernel[0]) for r in kernel)):
        raise HTTPException(422, detail="Kernel phải là ma trận chữ nhật 1–5 hàng/cột.")
    if len(kernel) > len(rows) or len(kernel[0]) > len(rows[0]):
        raise HTTPException(422, detail="Kernel không được lớn hơn ảnh.")
    if not _valid_numbers([v for row in rows for v in row] + [v for row in kernel for v in row], -1000, 1000):
        raise HTTPException(422, detail="Mọi giá trị phải là số hữu hạn trong [-1000, 1000].")
    inp = torch.tensor(rows, dtype=torch.float32)
    weight = torch.tensor(kernel, dtype=torch.float32)
    with torch.no_grad():
        out = models.Convolve(inp, weight).tolist()
    return {"output": out, "shape": [len(out), len(out[0])], "source": "models.Convolve: valid cross-correlation, stride 1, không padding"}


@app.post("/api/q5/predict")
def q5_predict(data: Pixels):
    model = _load_model("q5")
    with torch.no_grad():
        logits = model(_validate_pixels(data.pixels))
        probs = torch.softmax(logits, dim=1)[0].tolist()
    return {"label": int(np.argmax(probs)), "probabilities": probs, "source": "DigitConvolutionalModel với checkpoint q5.pt"}


@app.post("/api/q6")
def q6(data: AttentionInput):
    tokens = data.tokens
    if not (2 <= len(tokens) <= 6 and all(len(row) == 4 for row in tokens)):
        raise HTTPException(422, detail="Cần 2–6 token, mỗi token là vector 4 chiều.")
    if not _valid_numbers([x for row in tokens for x in row], -100, 100):
        raise HTTPException(422, detail="Giá trị token phải là số hữu hạn trong [-100,100].")
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(188)
        model = models.Attention(layer_size=4, block_size=6).eval()
    x = torch.tensor(tokens, dtype=torch.float32).unsqueeze(0)
    with torch.no_grad():
        actual = model(x).squeeze(0).tolist()
        q = model.q_layer(x)
        k = model.k_layer(x)
        scores = q @ k.transpose(1, 2) / math.sqrt(4)
        mask = model.mask[:, :, :len(tokens), :len(tokens)] == 0
        scores = scores.masked_fill(mask, float("-inf"))[0]
        weights = torch.softmax(scores, dim=-1).squeeze(0).tolist()
    return {
        "attention": weights,
        "output": actual,
        "mask": [[j <= i for j in range(len(tokens))] for i in range(len(tokens))],
        "source": "models.Attention.forward với linear layers khởi tạo seed 188 (chưa huấn luyện)",
    }


app.mount("/", StaticFiles(directory=str(ROOT / "docs"), html=True), name="website")