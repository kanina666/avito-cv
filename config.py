"""Единый конфиг: seed, пути, параметры препроцессинга и обучения."""
import os
import random

import numpy as np

SEED = 42

ROOT = os.path.dirname(os.path.abspath(__file__))
TEST_DIR = os.path.join(ROOT, "data", "test", "images")
SAMPLE_SUB = os.path.join(ROOT, "data", "sample_submission.csv")
CACHE = os.path.join(ROOT, "cache")          # кэш кропов и весов
os.makedirs(CACHE, exist_ok=True)

# Препроцессинг: ресайз по высоте с сохранением пропорций, паддинг справа до фикс. ширины.
# Вертикальных кропов (h > w) в тесте нет (проверено), поэтому в трейне такие отбрасываются.
H, W = 48, 256
MIN_AR = 1.5                                 # отсев почти квадратных/вертикальных кропов

# Сколько примеров брать из каждого источника
N_TEXTOCR = 40000                            # реальные сцены (обучение)
N_CYR = 20000                                # реальная кириллица (обучение)
N_SYNTH = 20000                              # PIL-синтетика (обучение)

# Расширенный набор для дообучения (первые N_* примеров совпадают с базовым набором -> hold-out тот же)
N_TEXTOCR2, N_CYR2, N_SYNTH2 = 85000, 40000, 40000

# Обучение
BATCH = 256
EPOCHS = 3
LR = 1e-3
BACKBONE = "mobilenetv3_small_100"


def seed_everything(seed: int = SEED):
    """Фиксируем все генераторы случайных чисел."""
    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    try:
        import torch
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    except ImportError:
        pass


def worker_init_fn(worker_id: int):
    """Seed для воркеров DataLoader."""
    random.seed(SEED + worker_id)
    np.random.seed(SEED + worker_id)
