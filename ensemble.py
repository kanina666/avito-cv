"""Стэкинг: логрегрессия по логитам (Paddle x0_25, Paddle x1_0, своя torch-модель), обучается ТОЛЬКО на реальной валидации.
Вход: cache/paddle_*.npz (paddle_logits.py в .venv_paddle), cache/model.pt. Выход: submission_v4.csv"""
import pickle
import sys

import cv2
import numpy as np
import pandas as pd
import torch
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_predict

import config as C
from infer import brier, predict_logits, sigmoid, tta_logit
from model import build_model

PADDLE = ["paddle_raw.npz", "paddle_PP-LCNet_x1_0_textline_ori.npz"]
OUT = sys.argv[1] if len(sys.argv) > 1 else "submission_v4.csv"
CKPT = sys.argv[2] if len(sys.argv) > 2 else "model.pt"   # чекпойнт своей torch-модели


def lg(p):
    p = np.clip(p, 1e-6, 1 - 1e-6)
    return np.log(p / (1 - p))


def feats(paddle, mine, part):
    """Логит-TTA каждой модели: 0.5 * (z(x) - z(rot180(x)))."""
    cols = [.5 * (lg(r[f"{part}_p"]) - lg(r[f"{part}_pr"])) for r in paddle] + [mine]
    return np.stack(cols, 1)


C.seed_everything()
R = [np.load(f"{C.CACHE}/{f}") for f in PADDLE]
model = build_model(False)
model.load_state_dict(torch.load(f"{C.CACHE}/{CKPT}"))

X, Y, S = [], [], []
for n in ["icdar", "cyr_hold"]:
    crops, y = pickle.load(open(f"{C.CACHE}/val_{n}.pkl", "rb"))
    z, zr = predict_logits(model, crops)
    X.append(feats(R, tta_logit(z, zr), n)); Y.append(y); S += [n] * len(y)
X, Y, S = np.concatenate(X), np.concatenate(Y), np.array(S)

# оценка: кросс-валидация с обучением на одном источнике и проверкой на другом + обычная 5-fold
for k, name in enumerate(["paddle x0_25", "paddle x1_0", "torch"]):
    print(f"{name:13s} одиночная модель Brier icdar {brier(sigmoid(X[S=='icdar', k]), Y[S=='icdar']):.4f} "
          f"cyr {brier(sigmoid(X[S=='cyr_hold', k]), Y[S=='cyr_hold']):.4f}")
for cols, nm in [([0, 2], "x0_25+torch"), ([1, 2], "x1_0+torch"), ([0, 1, 2], "все три"), ([0, 1], "два paddle")]:
    cv = cross_val_predict(LogisticRegression(C=10), X[:, cols], Y, cv=5, method="predict_proba")[:, 1]
    print(f"stack {nm:12s} 5-fold Brier {brier(cv, Y):.4f}  icdar {brier(cv[S=='icdar'], Y[S=='icdar']):.4f}")

lr = LogisticRegression(C=10).fit(X, Y)
print("weights", lr.coef_.round(3), lr.intercept_.round(3))

sub = pd.read_csv(C.SAMPLE_SUB)
crops = [cv2.cvtColor(cv2.imread(f"{C.TEST_DIR}/{i}.png"), cv2.COLOR_BGR2RGB) for i in sub.image_id]
z, zr = predict_logits(model, crops)
p = lr.predict_proba(feats(R, tta_logit(z, zr), "test"))[:, 1]
pd.DataFrame({"image_id": sub.image_id, "p_180": p}).to_csv(OUT, index=False)
print(OUT, "mean", p.mean().round(3), "frac>.5", (p > .5).mean().round(3), "hist", np.histogram(p, 10, (0, 1))[0])
