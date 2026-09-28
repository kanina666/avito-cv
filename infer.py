"""Инференс: логиты с TTA (поворот исходного кропа на 180° ДО паддинга), калибровка температурой."""
import cv2
import numpy as np
import torch
from scipy.optimize import minimize_scalar

import config as C
from dataset import letterbox, to_tensor


@torch.no_grad()
def predict_logits(model, crops, bs=512, device="cuda"):
    """crops: список RGB uint8 массивов (в исходной ориентации).
    Возвращает (z, z_rot): логиты для кропа и для кропа, повёрнутого на 180°."""
    model.eval().to(device)
    res = []
    for rot in (False, True):
        zs = []
        for i in range(0, len(crops), bs):
            b = [letterbox(cv2.rotate(c, cv2.ROTATE_180) if rot else c) for c in crops[i:i + bs]]
            x = to_tensor(torch.from_numpy(np.stack(b)).to(device))
            with torch.autocast(device, dtype=torch.bfloat16):
                zs.append(model(x.contiguous(memory_format=torch.channels_last)).float().squeeze(1).cpu())
        res.append(torch.cat(zs).numpy())
    return res[0], res[1]


def sigmoid(z):
    return 1 / (1 + np.exp(-z))


def tta_logit(z, z_rot):
    """Для перевёрнутой картинки логит поворота противоположен -> усредняем в пространстве логитов."""
    return 0.5 * (z - z_rot)


def brier(p, y):
    return float(np.mean((p - y) ** 2))


def fit_temperature(z, y):
    """Один параметр T (p = sigmoid(z / T)), минимизирует Brier на отложенной реальной валидации."""
    r = minimize_scalar(lambda t: brier(sigmoid(z / t), y), bounds=(0.2, 10), method="bounded")
    return float(r.x)
