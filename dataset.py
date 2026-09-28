"""Датасеты: препроцессинг кропа, аугментации, случайный поворот на 180° с меткой."""
import random

import cv2
import numpy as np
import torch
from torch.utils.data import Dataset

import config as C


def letterbox(img: np.ndarray) -> np.ndarray:
    """RGB-кроп любого размера -> uint8 [H, W, 3]: ресайз по высоте до H с сохранением пропорций,
    ширина не более W (длинные сжимаются), справа паддинг серым (после нормализации это 0).
    Вертикальные кропы в тесте не встречаются, поэтому специальной обработки нет: они просто
    сжимаются по той же схеме."""
    h, w = img.shape[:2]
    nw = min(C.W, max(8, round(w * C.H / h)))
    img = cv2.resize(img, (nw, C.H), interpolation=cv2.INTER_AREA if h > C.H else cv2.INTER_CUBIC)
    out = np.full((C.H, C.W, 3), 127, np.uint8)
    out[:, :nw] = img
    return out


def to_tensor(batch_uint8: torch.Tensor) -> torch.Tensor:
    """uint8 [B,H,W,3] (на GPU) -> float [B,3,H,W] нормализованный в [-1, 1]."""
    return (batch_uint8.permute(0, 3, 1, 2).float() - 127.5) / 127.5


def augment(im: np.ndarray, rng: random.Random) -> np.ndarray:
    """Аугментации кропа (высота C.H, ширина <= C.W), выполняются ДО поворота на 180°."""
    h, w = im.shape[:2]
    # горизонтальный масштаб -> разные aspect ratio
    if rng.random() < 0.5:
        nw = int(np.clip(w * rng.uniform(0.6, 1.4), 12, C.W))
        im = cv2.resize(im, (nw, h))
        w = nw
    # обрезка края (детектор часто режет текст) - убираем до 15% с каждого края
    if rng.random() < 0.3 and w > 40:
        a, b = int(w * rng.uniform(0, .15)), int(w * rng.uniform(0, .15))
        im = im[:, a:w - b]
        w = im.shape[1]
    # малый поворот + перспектива
    if rng.random() < 0.5:
        ang, sh = rng.uniform(-6, 6), rng.uniform(-0.15, 0.15)
        M = cv2.getRotationMatrix2D((w / 2, h / 2), ang, 1.0)
        M[0, 1] += sh
        im = cv2.warpAffine(im, M, (w, h), borderMode=cv2.BORDER_REPLICATE)
    if rng.random() < 0.3:
        d = rng.uniform(0, 0.12) * h
        src = np.float32([[0, 0], [w, 0], [w, h], [0, h]])
        dst = src + np.float32([[rng.uniform(0, d), rng.uniform(0, d)], [-rng.uniform(0, d), rng.uniform(0, d)],
                                [-rng.uniform(0, d), -rng.uniform(0, d)], [rng.uniform(0, d), -rng.uniform(0, d)]])
        im = cv2.warpPerspective(im, cv2.getPerspectiveTransform(src, dst), (w, h), borderMode=cv2.BORDER_REPLICATE)
    # цвет: яркость/контраст, ч/б, инверсия
    if rng.random() < 0.7:
        im = np.clip(im.astype(np.float32) * rng.uniform(0.6, 1.4) + rng.uniform(-40, 40), 0, 255).astype(np.uint8)
    if rng.random() < 0.15:
        im = np.repeat(cv2.cvtColor(im, cv2.COLOR_RGB2GRAY)[..., None], 3, 2)
    if rng.random() < 0.1:
        im = 255 - im
    # качество: blur, шум, JPEG, даунскейл-апскейл
    if rng.random() < 0.3:
        k = rng.choice([3, 5])
        im = cv2.GaussianBlur(im, (k, k), 0)
    if rng.random() < 0.3:
        im = np.clip(im.astype(np.float32) + np.random.normal(0, rng.uniform(2, 12), im.shape), 0, 255).astype(np.uint8)
    if rng.random() < 0.3:
        im = cv2.imdecode(cv2.imencode(".jpg", im, [cv2.IMWRITE_JPEG_QUALITY, rng.randint(20, 80)])[1], 1)
    if rng.random() < 0.2:
        f = rng.uniform(0.3, 0.7)
        im = cv2.resize(cv2.resize(im, (max(8, int(w * f)), max(8, int(h * f)))), (w, h))
    return im


class OrientDS(Dataset):
    """Обучающий датасет: метка 180° выбирается случайно на каждом обращении."""

    def __init__(self, crops, seed=C.SEED):
        self.crops, self.seed = crops, seed

    def __len__(self):
        return len(self.crops)

    def __getitem__(self, i):
        rng = random.Random(self.seed * 1000003 + i + random.randrange(1 << 30))  # разный результат каждую эпоху
        im = augment(self.crops[i], rng)
        y = rng.random() < 0.5
        if y:
            im = cv2.rotate(im, cv2.ROTATE_180)   # поворот до паддинга, как и при инференсе
        return torch.from_numpy(letterbox(im)), torch.tensor(float(y))


class FixedDS(Dataset):
    """Валидационный датасет: детерминированные метки (по seed), без аугментаций."""

    def __init__(self, crops, seed=C.SEED):
        self.y = np.random.RandomState(seed).rand(len(crops)) < 0.5
        self.crops = [cv2.rotate(c, cv2.ROTATE_180) if y else c for c, y in zip(crops, self.y)]

    def __len__(self):
        return len(self.crops)

    def __getitem__(self, i):
        return torch.from_numpy(letterbox(self.crops[i])), torch.tensor(float(self.y[i]))
