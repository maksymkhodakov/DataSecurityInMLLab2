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

Навіщо це потрібно. Коли дані клієнтів сильно відрізняються (наприклад, один
клієнт бачить лише павуків і жуків, інший — лише метеликів і бабок), кожен клієнт
під час локального навчання «тягне» модель у свій бік: ваги віддаляються від
спільного оптимуму і після усереднення можуть давати гіршу модель, ніж будь-яка
з локальних. Проксимальний член штрафує за відхилення від глобальних ваг, тому
локальні оновлення лишаються «поруч» із глобальною моделлю.

Градієнт проксимального члена по w дорівнює mu * (w - w_global), тобто до
звичайного градієнта додається сила, пропорційна відстані до глобальної моделі.
"""
from __future__ import annotations

import time

import torch
from torch.utils.data import DataLoader

from . import config as C
from .data import ArthropodDetectionDataset, collate_fn
from .model import get_weights, set_weights


class FLClient:
    """Один учасник федеративного навчання.

    У реальній системі це був би окремий пристрій (телефон, лікарня, лабораторія),
    а тут — об'єкт у тому ж процесі. Ізоляцію даних забезпечує те, що клієнт
    тримає посилання лише на власний Dataset і повертає назовні тільки ваги
    та агреговані числові метрики (жодних зображень чи анотацій).
    """

    def __init__(self, client_id: int, dataset: ArthropodDetectionDataset):
        self.client_id = client_id
        self.dataset = dataset          # приватні дані клієнта
        # Кількість прикладів n_k повідомляється серверу: саме з цих чисел
        # обчислюються ваги n_k / n у зваженому усередненні (FedAvg / FedAdam).
        # Самі дані при цьому не передаються — лише їх кількість.
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
        щоб не тримати в пам'яті окрему копію на кожного клієнта).

        Аргументи:
            model         – робоча модель; її ваги буде перезаписано глобальними;
            global_weights– OrderedDict {ім'я параметра: тензор} від сервера (w_t);
            keys          – імена навчуваних параметрів (порядок збігається з params);
            device        – пристрій для обчислень (cpu / mps / cuda);
            round_idx     – номер раунду (використовується для seed перемішування);
            proximal_mu   – коефіцієнт mu для FedProx; 0 означає звичайний FedAvg.
        """
        # 1. Отримуємо глобальну модель від сервера.
        # Кожен клієнт починає раунд з ОДНАКОВИХ ваг w_t — це ключова умова
        # коректного усереднення: сервер усереднює зміщення від спільної точки.
        set_weights(model, global_weights)
        # train() вмикає режим навчання: Faster R-CNN у цьому режимі приймає
        # targets і повертає словник втрат замість передбачень.
        model.train()

        # Беремо лише параметри, що навчаються (заморожений backbone виключено).
        # Порядок збігається з `keys`, бо обидва списки побудовано через
        # model.named_parameters() з тим самим фільтром requires_grad.
        params = [p for n, p in model.named_parameters() if p.requires_grad]
        # Копія глобальних ваг на пристрої — потрібна для проксимального члена FedProx.
        # Для FedAvg / FedAdam (mu = 0) копію не створюємо, щоб не витрачати пам'ять.
        global_params = [global_weights[k].to(device) for k in keys] if proximal_mu > 0 else None

        # Оптимізатор створюється заново кожного раунду (стан моментів не зберігається
        # між раундами — стандартна практика у FedAvg/FedProx).
        # Momentum 0.9 згладжує шум градієнтів на маленьких батчах,
        # weight_decay — L2-регуляризація проти перенавчання на малій вибірці.
        optimizer = torch.optim.SGD(
            params, lr=C.CLIENT_LR, momentum=C.CLIENT_MOMENTUM, weight_decay=C.CLIENT_WEIGHT_DECAY
        )

        # Фіксуємо генератор перемішування для відтворюваності.
        # Seed залежить від номера раунду і клієнта, тому:
        #   * у різних раундах порядок батчів різний (як у звичайному навчанні);
        #   * різні клієнти перемішують дані по-різному;
        #   * при повторному запуску експерименту все відтворюється точно так само,
        #     а різні стратегії бачать однакову послідовність батчів (чесне порівняння).
        g = torch.Generator()
        g.manual_seed(C.SEED * 1000 + round_idx * 10 + self.client_id)
        loader = DataLoader(
            self.dataset, batch_size=C.BATCH_SIZE, shuffle=True,
            collate_fn=collate_fn, num_workers=0, generator=g,  # num_workers=0 — економія RAM
        )

        t0 = time.time()
        # Накопичувачі для середніх втрат за раунд (для графіків і звіту):
        #   loss      – повна оптимізована втрата (детекція + проксимальний член);
        #   det_loss  – лише втрата детекції (порівнювана між стратегіями);
        #   prox_loss – лише проксимальний член (≠ 0 тільки для FedProx).
        sums = {"loss": 0.0, "det_loss": 0.0, "prox_loss": 0.0}
        n_batches = 0
        for _ in range(C.LOCAL_EPOCHS):
            for images, targets in loader:
                # images  – список тензорів (3, H, W); розміри можуть відрізнятися;
                # targets – список словників {"boxes": (N, 4), "labels": (N,)}.
                images = [img.to(device) for img in images]
                targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

                # Faster R-CNN у режимі train повертає словник з 4 втрат:
                # loss_classifier, loss_box_reg (ROI-голова), loss_objectness, loss_rpn_box_reg (RPN)
                #   * loss_objectness  – чи містить якір (anchor) об'єкт взагалі (RPN);
                #   * loss_rpn_box_reg – уточнення координат пропозицій RPN;
                #   * loss_classifier  – крос-ентропія класу для кожної пропозиції (RoI);
                #   * loss_box_reg     – фінальне уточнення рамки для відповідного класу.
                # Сумуємо їх з однаковими вагами — так навчається torchvision-детектор.
                loss_dict = model(images, targets)
                det_loss = sum(loss_dict.values())
                loss = det_loss

                # ---- FedProx: проксимальний член ----
                # prox = (mu / 2) * Σ_i ||w_i − w_global_i||², сума по всіх навчуваних тензорах.
                # На першому кроці раунду w == w_global, тож prox = 0; далі він росте
                # разом із відхиленням локальної моделі від глобальної.
                prox = torch.tensor(0.0, device=device)
                if proximal_mu > 0:
                    for p, gp in zip(params, global_params):
                        prox = prox + (p - gp).pow(2).sum()
                    prox = 0.5 * proximal_mu * prox
                    loss = loss + prox

                # Стандартний крок оптимізації. set_to_none=True звільняє пам'ять
                # під градієнти замість заповнення їх нулями (трохи швидше).
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                # Обрізання градієнта — захист від вибухових градієнтів на початку навчання.
                # Нова голова класифікатора ініціалізована випадково, тому перші батчі
                # можуть давати дуже великі градієнти; обмеження норми до 10 не дає
                # одному «поганому» батчу зіпсувати попередньо навчені шари FPN / RPN.
                torch.nn.utils.clip_grad_norm_(params, max_norm=10.0)
                optimizer.step()

                # .item() переносить скаляр на CPU і відриває його від графа обчислень,
                # щоб не тримати граф у пам'яті до кінця епохи.
                sums["loss"] += loss.item()
                sums["det_loss"] += det_loss.item()
                sums["prox_loss"] += prox.item()
                n_batches += 1

        # 2. «Відправляємо» серверу оновлені ваги (копія на CPU).
        # Копія обов'язкова: робоча модель далі буде перезаписана вагами
        # наступного клієнта, а серверу потрібні ваги саме цього клієнта.
        new_weights = get_weights(model, keys)
        del global_params  # звільняємо копію глобальних ваг на пристрої
        # Повідомлення серверу. Зверніть увагу: тут немає жодних даних клієнта —
        # лише ваги, кількість прикладів і усереднені метрики навчання.
        # max(n_batches, 1) захищає від ділення на нуль, якщо датасет клієнта порожній.
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
