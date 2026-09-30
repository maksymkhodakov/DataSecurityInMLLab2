"""
Оцінювання глобальної моделі на СПІЛЬНОМУ тестовому наборі сервера.

Тестовий набір не перетинається з даними жодного клієнта і збалансований
по класах (TEST_PER_CLASS зображень кожного класу), тому він однаково
чесно оцінює моделі з усіх сценаріїв розбиття.

Метрики (стандарт COCO, через torchmetrics + pycocotools):
  * mAP@0.5       – середня точність при порозі IoU = 0.5 (основна метрика);
  * mAP@0.5:0.95  – усереднення по порогах IoU 0.5…0.95 з кроком 0.05;
  * mAR@100       – середня повнота (до 100 детекцій на зображення);
  * AP по кожному класу (IoU 0.5:0.95) та AP50 по кожному класу.

Коротко про терміни:
  * IoU (Intersection over Union) – площа перетину передбаченої та справжньої
    рамок, поділена на площу їх об'єднання. Детекція вважається правильною,
    якщо IoU ≥ порогу і клас збігається.
  * AP (Average Precision) – площа під кривою precision–recall для одного класу,
    побудованою при зміні порогу впевненості моделі.
  * mAP – середнє AP по всіх класах. mAP@0.5 оцінює, чи знайшла модель об'єкт
    взагалі; mAP@0.5:0.95 додатково вимагає точної локалізації рамки.
"""
from __future__ import annotations

import torch
from torch.utils.data import DataLoader
from torchmetrics.detection import MeanAveragePrecision

from . import config as C
from .data import collate_fn


# no_grad вимикає побудову графа обчислень: на інференсі градієнти не потрібні,
# а це суттєво економить пам'ять і прискорює роботу.
@torch.no_grad()
def evaluate(model: torch.nn.Module, dataset, device: torch.device) -> dict:
    # eval() перемикає Faster R-CNN у режим інференсу: модель повертає список
    # передбачень {"boxes", "labels", "scores"} замість словника втрат.
    model.eval()
    # shuffle=False — порядок не впливає на метрику, але так оцінка детермінована.
    loader = DataLoader(dataset, batch_size=16, shuffle=False, collate_fn=collate_fn, num_workers=0)

    # Метрика з усіма порогами IoU (для mAP, mAP50, AP по класах)
    metric = MeanAveragePrecision(iou_type="bbox", class_metrics=True)
    # Окрема метрика лише з порогом 0.5 — щоб отримати AP50 по кожному класу
    # (стандартний map_per_class усереднений по всіх порогах 0.5…0.95).
    metric50 = MeanAveragePrecision(iou_type="bbox", class_metrics=True, iou_thresholds=[0.5])

    for images, targets in loader:
        images = [img.to(device) for img in images]
        outputs = model(images)
        # torchmetrics накопичує передбачення і обчислює метрику на CPU,
        # тому переносимо результати з GPU/MPS.
        preds = [{k: v.detach().cpu() for k, v in o.items()} for o in outputs]
        tgts = [{"boxes": t["boxes"], "labels": t["labels"]} for t in targets]
        metric.update(preds, tgts)
        metric50.update(preds, tgts)

    # compute() зіставляє всі передбачення з усіма справжніми рамками по всьому
    # набору (а не по батчах) — саме так рахується mAP за протоколом COCO.
    res = metric.compute()
    res50 = metric50.compute()

    def per_class(r, key):
        # map_per_class повертається для класів, що зустрічаються у даних (r["classes"])
        # Перетворюємо у словник {мітка: AP} і розкладаємо у список фіксованої довжини
        # в порядку config.CLASSES. Мітки в моделі починаються з 1 (0 — фон), тому i + 1.
        # torchmetrics повертає −1 для класів без прикладів — замінюємо на 0.
        vals = {int(c): float(v) for c, v in zip(r["classes"].reshape(-1), r[key].reshape(-1))}
        return [max(vals.get(i + 1, 0.0), 0.0) for i in range(len(C.CLASSES))]

    # Переводимо тензори у звичайні float, щоб результат можна було записати в JSON.
    return {
        "map": float(res["map"]),
        "map_50": float(res["map_50"]),
        "map_75": float(res["map_75"]),
        "mar_100": float(res["mar_100"]),
        "ap_per_class": per_class(res, "map_per_class"),
        "ap50_per_class": per_class(res50, "map_per_class"),
    }
