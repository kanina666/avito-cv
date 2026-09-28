"""Сырые вероятности p180 готовой Paddle-модели (по умолчанию PP-LCNet_x0_25_textline_ori; имя модели - argv[1]) для кропа и его поворота на 180°.
Запускать в .venv_paddle (Paddle и torch в одном процессе конфликтуют по DLL). Результат: cache/paddle_raw.npz"""
import pickle
import sys

import cv2
import numpy as np
import pandas as pd
from paddleocr import TextLineOrientationClassification

NAME = sys.argv[1] if len(sys.argv) > 1 else "PP-LCNet_x0_25_textline_ori"
m = TextLineOrientationClassification(model_name=NAME)


def p180(ims):
    """BGR-кропы -> p(180°). batch_size=1: батчи разного размера в Paddle дают неверные классы."""
    o = []
    for x in ims:
        r = next(iter(m.predict([x], batch_size=1)))
        s = float(np.ravel(r["scores"])[0])
        o.append(s if int(np.ravel(r["class_ids"])[0]) == 1 else 1 - s)
    return np.array(o)


def both(bgr):
    return p180(bgr), p180([cv2.rotate(x, cv2.ROTATE_180) for x in bgr])


out = {}
for n in ["icdar", "cyr_hold"]:  # валидационные кропы уже повёрнуты согласно метке (см. FixedDS)
    c, y = pickle.load(open(f"cache/val_{n}.pkl", "rb"))
    out[f"{n}_p"], out[f"{n}_pr"] = both([cv2.cvtColor(x, cv2.COLOR_RGB2BGR) for x in c])
    out[f"{n}_y"] = y
ids = pd.read_csv("data/sample_submission.csv").image_id
out["test_p"], out["test_pr"] = both([cv2.imread(f"data/test/images/{i}.png") for i in ids])
np.savez("cache/paddle_raw.npz" if NAME.startswith("PP-LCNet_x0_25") else f"cache/paddle_{NAME}.npz", **out)
print("saved")
