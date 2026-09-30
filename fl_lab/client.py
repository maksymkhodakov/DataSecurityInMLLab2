"""
Клієнт (воркер) федеративного навчання.

Кожен клієнт:
  * володіє ТІЛЬКИ своєю частиною даних (екземпляр ArthropodDetectionDataset
    з власними індексами) — дані ніколи не покидають клієнта;
  * отримує від сервера поточні глобальні ваги;
  * виконує LOCAL_EPOCHS епох SGD на своїх даних;
  * повертає серверу оновлені ваги, кількість прикладів і статистику втрат.

Для стратегії FedProx до локальної функції втрат додається проксимальний член
    (mu / 2) * || w - w_global ||^2,
який «притягує» локальну модель до глобальної і зменшує дрейф клієнтів
при неоднорідних (non-IID) даних.
"""
from __future__ import annotations

import time

import torch
from torch.utils.data import DataLoader

from . import config as C
from .data import ArthropodDetectionDataset, collate_fn
from .model import get_weights, set_weights


class FLClient:
    def __init__(self, client_id: int, dataset: ArthropodDetectionDataset):
        self.client_id = client_id
        self.dataset = dataset          # приватні дані клієнта
        self.num_examples = len(dataset)

    def fit(
        self,
        model: torch.nn.Module,
        global_weights,
        keys: list[str],
        device: torch.device,
        round_idx: int,
        proximal_mu: float = 0.0,
    ) -> dict:
        """Один раунд локального навчання. Повертає словник з оновленими вагами
        та метриками. `model` — робочий екземпляр моделі (перевикористовується,
        щоб не тримати в пам'яті окрему копію на кожного клієнта)."""
        # 1. Отримуємо глобальну модель від сервера
        set_weights(model, global_weights)
        model.train()

        params = [p for n, p in model.named_parameters() if p.requires_grad]
        # Копія глобальних ваг на пристрої — потрібна для проксимального члена FedProx
        global_params = [global_weights[k].to(device) for k in keys] if proximal_mu > 0 else None

        # Оптимізатор створюється заново кожного раунду (стан моментів не зберігається
        # між раундами — стандартна практика у FedAvg/FedProx).
        optimizer = torch.optim.SGD(
            params, lr=C.CLIENT_LR, momentum=C.CLIENT_MOMENTUM, weight_decay=C.CLIENT_WEIGHT_DECAY
        )

        # Фіксуємо генератор перемішування для відтворюваності
        g = torch.Generator()
        g.manual_seed(C.SEED * 1000 + round_idx * 10 + self.client_id)
        loader = DataLoader(
            self.dataset, batch_size=C.BATCH_SIZE, shuffle=True,
            collate_fn=collate_fn, num_workers=0, generator=g,  # num_workers=0 — економія RAM
        )

        t0 = time.time()
        sums = {"loss": 0.0, "det_loss": 0.0, "prox_loss": 0.0}
        n_batches = 0
        for _ in range(C.LOCAL_EPOCHS):
            for images, targets in loader:
                images = [img.to(device) for img in images]
                targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

                # Faster R-CNN у режимі train повертає словник з 4 втрат:
                # loss_classifier, loss_box_reg (ROI-голова), loss_objectness, loss_rpn_box_reg (RPN)
                loss_dict = model(images, targets)
                det_loss = sum(loss_dict.values())
                loss = det_loss

                # ---- FedProx: проксимальний член ----
                prox = torch.tensor(0.0, device=device)
                if proximal_mu > 0:
                    for p, gp in zip(params, global_params):
                        prox = prox + (p - gp).pow(2).sum()
                    prox = 0.5 * proximal_mu * prox
                    loss = loss + prox

                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                # Обрізання градієнта — захист від вибухових градієнтів на початку навчання
                torch.nn.utils.clip_grad_norm_(params, max_norm=10.0)
                optimizer.step()

                sums["loss"] += loss.item()
                sums["det_loss"] += det_loss.item()
                sums["prox_loss"] += prox.item()
                n_batches += 1

        # 2. «Відправляємо» серверу оновлені ваги (копія на CPU)
        new_weights = get_weights(model, keys)
        del global_params
        return {
            "client_id": self.client_id,
            "weights": new_weights,
            "num_examples": self.num_examples,
            "train_loss": sums["loss"] / max(n_batches, 1),
            "det_loss": sums["det_loss"] / max(n_batches, 1),
            "prox_loss": sums["prox_loss"] / max(n_batches, 1),
            "num_batches": n_batches,
            "time_sec": time.time() - t0,
        }
