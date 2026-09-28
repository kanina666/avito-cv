"""Экспорт torch-модели в ONNX, замер размера и латентности (CPU: onnxruntime, GPU: torch CUDA), batch 1 и 256."""
import os
import time

import numpy as np
import onnxruntime as ort
import torch

import config as C
from model import build_model

C.seed_everything()
m = build_model(False)
m.load_state_dict(torch.load(f"{C.CACHE}/model.pt"))
m.eval()
path = f"{C.CACHE}/orientation.onnx"
torch.onnx.export(m, torch.randn(1, 3, C.H, C.W), path, input_names=["x"], output_names=["logit"],
                  dynamic_axes={"x": {0: "b"}, "logit": {0: "b"}}, opset_version=17, dynamo=False)
print(f"ONNX size: {os.path.getsize(path) / 1e6:.2f} MB, params: {sum(p.numel() for p in m.parameters()) / 1e6:.2f}M")


def timeit(fn, n):
    for _ in range(3):
        fn()
    t = time.perf_counter()
    for _ in range(n):
        fn()
    return (time.perf_counter() - t) / n


# CPU (onnxruntime)
so = ort.SessionOptions()
sess = ort.InferenceSession(path, so, providers=["CPUExecutionProvider"])
for b, n in [(1, 200), (256, 5)]:
    x = np.random.randn(b, 3, C.H, C.W).astype(np.float32)
    dt = timeit(lambda: sess.run(None, {"x": x}), n)
    print(f"CPU  ORT   batch {b:3d}: {dt * 1e3:8.2f} ms/batch = {dt * 1e3 / b:.3f} ms/crop")

# GPU (torch, fp16)
mg = m.cuda().half()
for b, n in [(1, 300), (256, 30)]:
    x = torch.randn(b, 3, C.H, C.W, device="cuda").half()

    def run():
        with torch.no_grad():
            mg(x)
        torch.cuda.synchronize()
    dt = timeit(run, n)
    print(f"GPU  torch batch {b:3d}: {dt * 1e3:8.2f} ms/batch = {dt * 1e3 / b:.3f} ms/crop")
