"""
Візуалізація передбачень моделі: рамки ground truth (пунктир) та передбачення
(суцільна лінія з підписом класу і впевненістю) на кількох тестових зображеннях.
"""
from __future__ import annotations

from pathlib import Path

import matplotlib
matplotlib.use("Agg")  # без GUI — лише збереження у файл
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import torch

from . import config as C


@torch.no_grad()
def save_predictions(model, dataset, device, out_path: Path, title: str = "",
                     n_images: int = 7, score_thr: float = 0.5) -> None:
    model.eval()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    # По одному зображенню кожного класу (тестовий набір впорядкований за класами)
    per_class = len(dataset) // len(C.CLASSES)
    idxs = [i * per_class for i in range(min(n_images, len(C.CLASSES)))]

    # Сітка 2 x 4: у звіті на сторінці A4 зображення лишаються достатньо великими
    ncols = 4
    nrows = (len(idxs) + ncols - 1) // ncols
    fig, axes = plt.subplots(nrows, ncols, figsize=(3.0 * ncols, 3.2 * nrows))
    axes = axes.flatten()
    for ax in axes[len(idxs):]:
        ax.axis("off")
    for ax, idx in zip(axes, idxs):
        image, target = dataset[idx]
        pred = model([image.to(device)])[0]
        ax.imshow(image.permute(1, 2, 0).numpy())
        for b in target["boxes"]:
            x1, y1, x2, y2 = b.tolist()
            ax.add_patch(patches.Rectangle((x1, y1), x2 - x1, y2 - y1, fill=False,
                                           edgecolor="#ffffff", linewidth=1.5, linestyle="--"))
        for b, l, s in zip(pred["boxes"].cpu(), pred["labels"].cpu(), pred["scores"].cpu()):
            if s < score_thr:
                continue
            x1, y1, x2, y2 = b.tolist()
            ok = int(l) == int(target["labels"][0])
            color = "#1baf7a" if ok else "#e34948"   # зелений – правильний клас, червоний – ні
            ax.add_patch(patches.Rectangle((x1, y1), x2 - x1, y2 - y1, fill=False,
                                           edgecolor=color, linewidth=2))
            ax.text(x1, max(y1 - 3, 8), f"{C.CLASSES[int(l) - 1]} {s:.2f}", fontsize=7,
                    color="white", bbox=dict(facecolor=color, alpha=0.85, pad=1, edgecolor="none"))
        ax.set_title(f"GT: {C.CLASSES[int(target['labels'][0]) - 1]}", fontsize=8)
        ax.axis("off")
    if title:
        fig.suptitle(title, fontsize=10)
    fig.tight_layout()
    fig.savefig(out_path, dpi=110)
    plt.close(fig)
