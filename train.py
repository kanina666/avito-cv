"""Обучение: BCEWithLogits, AdamW, cosine, bf16-AMP. Валидация по отдельным источникам."""
import time

import numpy as np
import torch
from torch.utils.data import ConcatDataset, DataLoader

import config as C
from data_prep import build
from dataset import FixedDS, OrientDS, to_tensor
from infer import brier, predict_logits, sigmoid, tta_logit
from model import build_model

HOLD = 3000  # сколько кропов cyr/synth откладываем на валидацию (не участвуют в обучении)


def load_splits(full=False):
    sfx = "2" if full else ""   # full: расширенный набор (тот же hold-out: первые HOLD кропов cyr/synth)
    tx, cy, sy, ic = (build(n) for n in ["textocr" + sfx, "cyr" + sfx, "synth" + sfx, "icdar_val"])
    train = tx + cy[HOLD:] + sy[HOLD:]
    val = {"icdar": ic, "cyr_hold": cy[:HOLD], "synth_hold": sy[:HOLD]}
    return train, val


@torch.no_grad()
def evaluate(model, val_ds, device="cuda"):
    """Brier без TTA по каждому источнику."""
    model.eval()
    out = {}
    for name, ds in val_ds.items():
        zs = []
        for x, _ in DataLoader(ds, batch_size=512):
            with torch.autocast(device, dtype=torch.bfloat16):
                zs.append(model(to_tensor(x.to(device)).contiguous(memory_format=torch.channels_last)).float().squeeze(1).cpu())
        out[name] = brier(sigmoid(torch.cat(zs).numpy()), ds.y.astype(np.float32))
    model.train()
    return out


def main(ft=False):
    """ft=True: дообучение model.pt на расширенном наборе (3 эпохи, lr 4e-4) -> model_ft.pt"""
    C.seed_everything()
    dev = "cuda"
    train, val = load_splits(full=ft)
    val_ds = {k: FixedDS(v) for k, v in val.items()}
    g = torch.Generator().manual_seed(C.SEED)
    dl = DataLoader(OrientDS(train), batch_size=C.BATCH, shuffle=True, drop_last=True, num_workers=6,
                    persistent_workers=True, worker_init_fn=C.worker_init_fn, generator=g, pin_memory=True)
    LR = 4e-4 if ft else C.LR
    model = build_model()
    if ft:
        model.load_state_dict(torch.load(f"{C.CACHE}/model.pt"))
    model = model.to(dev).to(memory_format=torch.channels_last)
    opt = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=1e-2)
    total = C.EPOCHS * len(dl)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=LR, total_steps=total, pct_start=0.05)
    lossf = torch.nn.BCEWithLogitsLoss()
    best, step, t0 = 9, 0, time.time()
    print(f"train={len(train)} steps={total}", {k: len(v) for k, v in val.items()})
    for ep in range(C.EPOCHS):
        for x, y in dl:
            x = to_tensor(x.to(dev, non_blocking=True)).contiguous(memory_format=torch.channels_last)
            with torch.autocast(dev, dtype=torch.bfloat16):
                loss = lossf(model(x).float().squeeze(1), y.to(dev))
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            sched.step()
            step += 1
            if step % 100 == 0:
                print(f"ep{ep} step {step}/{total} loss {loss.item():.4f} {time.time() - t0:.0f}s", flush=True)
            if step % 300 == 0 or step == total:
                m = evaluate(model, val_ds)
                print("  val Brier", {k: round(v, 4) for k, v in m.items()}, flush=True)
                if m["icdar"] < best:  # выбор чекпойнта по реальному источнику
                    best = m["icdar"]
                    torch.save(model.state_dict(), f"{C.CACHE}/{'model_ft.pt' if ft else 'model.pt'}")
    print("best icdar Brier", best)


if __name__ == "__main__":
    import sys
    main(ft="ft" in sys.argv)
