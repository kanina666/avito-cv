"""Baseline v1: готовая PaddleOCR-модель ориентации строки (0°/180°) + TTA поворотом на 180°."""
import sys
import random

import numpy as np
import pandas as pd
import cv2
from paddleocr import TextLineOrientationClassification

SEED = 42
random.seed(SEED)
np.random.seed(SEED)

DATA = "data"
N = int(sys.argv[1]) if len(sys.argv) > 1 else None  # для быстрой проверки на части теста

sub = pd.read_csv(f"{DATA}/sample_submission.csv")
ids = sub["image_id"].tolist()[:N]
model = TextLineOrientationClassification(model_name="PP-LCNet_x0_25_textline_ori")


def p180(imgs, bs=1):
    """Вероятность класса 180° для списка BGR-картинок (в модели 2 класса, отдаётся top-1 score)."""
    out = []
    for i in range(0, len(imgs), bs):
        for r in model.predict(imgs[i:i + bs], batch_size=bs):
            score = float(np.ravel(r["scores"])[0])
            out.append(score if int(np.ravel(r["class_ids"])[0]) == 1 else 1.0 - score)
    return np.array(out)


imgs = [cv2.imread(f"{DATA}/test/images/{i}.png") for i in ids]
p = p180(imgs)
p_rot = p180([cv2.rotate(x, cv2.ROTATE_180) for x in imgs])
# TTA: p = 0.5 * (p(x) + 1 - p(rot180(x)))
p_tta = 0.5 * (p + 1.0 - p_rot)

out = pd.DataFrame({"image_id": ids, "p_180": np.clip(p_tta, 0, 1)})
out.to_csv("submission_v1.csv" if N is None else "sub_v1_debug.csv", index=False)
print("mean p180", out.p_180.mean(), "frac>0.5", (out.p_180 > 0.5).mean(),
      "agree p vs TTA", ((p > .5) == (p_tta > .5)).mean())
