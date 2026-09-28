"""Подготовка данных: скачивание реальных кропов с HF, PIL-синтетика, кэш в cache/*.pkl.

Каждый источник -> список uint8 RGB-массивов высоты H (ширина <= W), в нормальной ориентации.
Метка (поворот на 180°) назначается позже, в Dataset.
"""
import io
import sys
import os
import pickle
import random
import string
import urllib.request

import cv2
import numpy as np
import pandas as pd
from PIL import Image, ImageDraw, ImageFont

import config as C

HF = "https://huggingface.co/api/datasets/{}/parquet/default/{}/{}.parquet"


def fit_height(img: np.ndarray):
    """Ресайз по высоте до C.H с сохранением пропорций, ширина не больше C.W. None для плохих кропов."""
    h, w = img.shape[:2]
    if h < 8 or w < 8 or w / h < C.MIN_AR:
        return None
    nw = min(C.W, max(8, round(w * C.H / h)))
    return cv2.resize(img, (nw, C.H), interpolation=cv2.INTER_AREA if h > C.H else cv2.INTER_CUBIC)


def _download(url: str, dst: str):
    if not os.path.exists(dst):
        print("download", url)
        req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
        with urllib.request.urlopen(req) as r, open(dst, "wb") as f:
            f.write(r.read())
    return dst


def from_parquet(ds: str, split: str, shards, n: int, name: str):
    """Читает parquet-шарды датасета HF, собирает до n подходящих кропов."""
    out = []
    for s in shards:
        p = _download(HF.format(ds, split, s), os.path.join(C.CACHE, f"{name}_{split}_{s}.parquet"))
        df = pd.read_parquet(p)
        for d in df["image"]:
            if d is None or d.get("bytes") is None:  # в некоторых шардах есть пустые строки
                continue
            im = cv2.imdecode(np.frombuffer(d["bytes"], np.uint8), cv2.IMREAD_COLOR)
            if im is None:
                continue
            im = fit_height(cv2.cvtColor(im, cv2.COLOR_BGR2RGB))
            if im is not None:
                out.append(im)
            if len(out) >= n:
                return out
    return out


def from_zip(url: str, n: int, name: str):
    """Датасеты, где картинки лежат jpg-файлами в zip-архиве репозитория (parquet хранит лишь пути)."""
    import zipfile
    z = zipfile.ZipFile(_download(url, os.path.join(C.CACHE, f"{name}.zip")))
    files = sorted(f for f in z.namelist() if f.lower().endswith((".jpg", ".png")))
    random.Random(C.SEED).shuffle(files)
    out = []
    for f in files:
        im = cv2.imdecode(np.frombuffer(z.read(f), np.uint8), cv2.IMREAD_COLOR)
        if im is None:
            continue
        im = fit_height(cv2.cvtColor(im, cv2.COLOR_BGR2RGB))
        if im is not None:
            out.append(im)
        if len(out) >= n:
            break
    return out


# ---------- PIL-синтетика ----------
def _usable_fonts():
    """Шрифты Windows, в которых есть кириллица (глиф 'Ж' отличается от .notdef)."""
    fdir = r"C:\Windows\Fonts"
    fonts = []
    for f in sorted(os.listdir(fdir)):
        if not f.lower().endswith((".ttf", ".otf")):
            continue
        try:
            ft = ImageFont.truetype(os.path.join(fdir, f), 32)
            a, b = ft.getmask("Жщ"), ft.getmask(chr(0xFFFE) * 2)
            if a.getbbox() and (a.size, a.getbbox()) != (b.size, b.getbbox()):  # глиф != .notdef
                fonts.append(os.path.join(fdir, f))
        except Exception:
            pass
    return fonts


def _rand_text(rng: random.Random):
    cyr = "абвгдеёжзийклмнопрстуфхцчшщъыьэюя"
    lat = string.ascii_lowercase
    kind = rng.random()
    if kind < 0.3:  # цена/число
        return rng.choice(["", "№ "]) + f"{rng.randint(1, 999999):,}".replace(",", " ") + rng.choice([" ₽", " руб", "", " р."])
    alpha = cyr if kind < 0.7 else lat
    words = ["".join(rng.choice(alpha) for _ in range(rng.randint(2, 9))) for _ in range(rng.randint(1, 4))]
    t = " ".join(words)
    t = rng.choice([t, t.upper(), t.capitalize()])
    if rng.random() < 0.2:
        t += " " + str(rng.randint(0, 9999))
    return t


def synth(n: int, seed: int):
    rng = random.Random(seed)
    fonts = _usable_fonts()
    print("synth fonts:", len(fonts))
    out = []
    while len(out) < n:
        ft = ImageFont.truetype(rng.choice(fonts), rng.randint(24, 64))
        txt = _rand_text(rng)
        l, t, r, b = ft.getbbox(txt)
        pad = rng.randint(2, 12)
        w, h = r - l + 2 * pad, b - t + 2 * pad
        bg = tuple(rng.randint(0, 255) for _ in range(3))
        fg = tuple(rng.randint(0, 255) for _ in range(3))
        if sum(abs(x - y) for x, y in zip(bg, fg)) < 150:  # нужен контраст
            fg = tuple(255 - x for x in bg)
        im = Image.new("RGB", (w, h), bg)
        ImageDraw.Draw(im).text((pad - l, pad - t), txt, font=ft, fill=fg)
        a = fit_height(np.array(im))
        if a is not None:
            out.append(a)
    return out


def build(name: str):
    path = os.path.join(C.CACHE, f"{name}.pkl")
    if os.path.exists(path):
        return pickle.load(open(path, "rb"))
    if name == "textocr":
        data = from_parquet("MiXaiLL76/TextOCR_OCR", "train", range(5), C.N_TEXTOCR, "textocr")
    elif name == "textocr2":
        data = from_parquet("MiXaiLL76/TextOCR_OCR", "train", range(5), C.N_TEXTOCR2, "textocr")
    elif name == "cyr2":
        data = from_zip("https://huggingface.co/datasets/DonkeySmall/OCR-Cyrillic-Printed-10/resolve/main/data_10.zip",
                        C.N_CYR2, "cyr")
    elif name == "synth2":
        data = synth(C.N_SYNTH2, C.SEED)
    elif name == "cyr":
        data = from_zip("https://huggingface.co/datasets/DonkeySmall/OCR-Cyrillic-Printed-10/resolve/main/data_10.zip",
                        C.N_CYR, "cyr")
    elif name == "synth":
        data = synth(C.N_SYNTH, C.SEED)
    elif name == "icdar_val":  # отдельный реальный источник для валидации/калибровки
        data = from_parquet("MiXaiLL76/ICDAR2015_OCR", "train", [0], 10**9, "icdar") + \
               from_parquet("MiXaiLL76/ICDAR2015_OCR", "test", [0], 10**9, "icdar")
    else:
        raise ValueError(name)
    pickle.dump(data, open(path, "wb"))
    return data


def dump_val():
    """Реальная валидация: ICDAR и hold-out кириллицы, с детерминированной меткой поворота (FixedDS).
    cache/val_*.pkl = (кропы уже повёрнутые согласно метке, метки). Читает и Paddle-скрипт (без torch)."""
    from dataset import FixedDS
    from train import HOLD
    for n, crops in [("icdar", build("icdar_val")), ("cyr_hold", build("cyr")[:HOLD])]:
        ds = FixedDS(crops)
        pickle.dump((ds.crops, ds.y.astype(np.float32)), open(os.path.join(C.CACHE, f"val_{n}.pkl"), "wb"))


if __name__ == "__main__":
    C.seed_everything()
    for n in ["icdar_val", "textocr", "cyr", "synth"] + (["textocr2", "cyr2", "synth2"] if "full" in sys.argv else []):
        d = build(n)
        print(n, len(d), "mean width", np.mean([x.shape[1] for x in d]))
    dump_val()
