"""
Сервер федеративного навчання (симуляція в одному процесі).

Алгоритм одного експерименту:
    1. Сервер ініціалізує глобальну модель (однакову для всіх експериментів).
    2. Для кожного раунду t = 1..T:
         a) сервер розсилає глобальні ваги w_t обом клієнтам;
         b) кожен клієнт навчає модель ЛОКАЛЬНО на своїх приватних даних
            і повертає лише оновлені ваги + кількість прикладів
            (дані клієнта серверу не передаються);
         c) сервер агрегує ваги обраною стратегією (FedAvg / FedAdam / FedProx);
         d) сервер оцінює нову глобальну модель на спільному тестовому наборі.
    3. Історія метрик по раундах зберігається в JSON (після КОЖНОГО раунду,
       тож у разі переривання результат не втрачається).

Примітка щодо пам'яті: використовується один робочий екземпляр моделі,
в який по черзі завантажуються ваги глобальної моделі / клієнтів.
Так у RAM одночасно лежать лише кілька копій навчуваних ваг (~64 МБ кожна).

Примітка щодо симуляції: у реальній системі клієнти навчаються паралельно на
своїх пристроях, а тут — послідовно в одному процесі. На результат це не
впливає: кожен клієнт стартує з тих самих глобальних ваг w_t і не бачить
оновлень іншого клієнта до агрегації (синхронний FL).
"""
from __future__ import annotations

import gc
import json
import time
from pathlib import Path

import torch

from . import config as C
from .client import FLClient
from .data import ArthropodDetectionDataset, load_cache
from .evaluate import evaluate
from .model import create_model, get_weights, set_weights, trainable_keys, weights_nbytes
from .partition import split_indices
from .strategies import make_strategy


def _free_memory(device: torch.device) -> None:
    """Примусово звільняє пам'ять між раундами / експериментами.

    gc.collect() знищує Python-об'єкти без посилань (старі копії ваг),
    а empty_cache() повертає звільнену пам'ять GPU/MPS операційній системі —
    інакше кешуючий алокатор PyTorch тримав би її зарезервованою.
    """
    gc.collect()
    if device.type == "mps":
        torch.mps.empty_cache()
    elif device.type == "cuda":
        torch.cuda.empty_cache()


def run_federated(strategy_name: str, scenario: str, device: torch.device,
                  num_rounds: int = C.NUM_ROUNDS, out_path: Path | None = None,
                  cache: dict | None = None, visualize: bool = True) -> dict:
    """Запускає один експеримент FL (стратегія × сценарій) і повертає історію метрик.

    cache можна передати ззовні, щоб при серії експериментів не читати
    файл кешу з диска щоразу заново.
    """
    cache = cache or load_cache()
    train_records, test_records = cache["train"], cache["test"]

    # ---------------- Розбиття даних між клієнтами ----------------
    # Сервер НЕ зберігає посилань на навчальні записи: кожен клієнт отримує
    # власний Dataset лише зі своїми індексами.
    parts = split_indices(train_records, scenario)
    # Клієнти нумеруються з 1 (0 зарезервовано для «централізованого» клієнта
    # у run_centralized). train=True — вмикає аугментацію на клієнтах.
    clients = [
        FLClient(cid, ArthropodDetectionDataset(train_records, idxs, train=True))
        for cid, idxs in enumerate(parts, start=1)
    ]
    del train_records  # далі сервер працює тільки з тестовим набором
    test_set = ArthropodDetectionDataset(test_records)

    # ---------------- Ініціалізація ----------------
    model = create_model().to(device)
    keys = trainable_keys(model)
    # global_weights — «стан сервера» w_t: саме ці ваги розсилаються клієнтам
    # і оновлюються після агрегації. Зберігаються на CPU.
    global_weights = get_weights(model, keys)
    strategy = make_strategy(strategy_name)

    # Історія експерименту — все, що потрібно для графіків і звіту:
    # конфігурація, розподіл даних клієнтів, обсяг комунікації і метрики по раундах.
    history = {
        "strategy": strategy_name,
        "scenario": scenario,
        "device": str(device),
        "config": {
            "num_rounds": num_rounds, "local_epochs": C.LOCAL_EPOCHS, "batch_size": C.BATCH_SIZE,
            "client_lr": C.CLIENT_LR, "fedprox_mu": C.FEDPROX_MU, "fedadam_eta": getattr(strategy, "eta", C.FEDADAM_ETA),
            "fedadam_tau": C.FEDADAM_TAU, "fedadam_betas": [C.FEDADAM_BETA1, C.FEDADAM_BETA2],
        },
        # Серверу відомі лише кількість прикладів і гістограма класів клієнтів —
        # вони записуються сюди виключно для аналізу експерименту у звіті.
        "clients": [
            {"client_id": c.client_id, "num_examples": c.num_examples,
             "class_hist": c.dataset.class_histogram()} for c in clients
        ],
        "num_trainable_params": int(sum(v.numel() for v in global_weights.values())),
        # Обсяг одного оновлення (в одну сторону). За раунд кожен клієнт отримує
        # і відправляє по одній такій порції ваг.
        "bytes_per_update": weights_nbytes(global_weights),
        "rounds": [],
    }

    print(f"\n=== {strategy_name.upper()} | сценарій «{scenario}» | пристрій {device} ===")
    for c in history["clients"]:
        print(f"  Клієнт {c['client_id']}: {c['num_examples']} зобр., класи {c['class_hist']}")

    # Раунд 0 — оцінка стартової моделі (для графіків прогресу)
    # Нова голова класифікатора ще не навчена, тому mAP тут близький до нуля —
    # це точка відліку, однакова для всіх стратегій.
    ev = evaluate(model, test_set, device)
    history["rounds"].append({"round": 0, **ev, "clients": [], "time_sec": 0.0})
    print(f"  Раунд  0: mAP@0.5={ev['map_50']:.4f}  mAP@0.5:0.95={ev['map']:.4f}")

    t_start = time.time()
    for rnd in range(1, num_rounds + 1):
        t0 = time.time()
        # a) + b) локальне навчання клієнтів
        # Кожен клієнт отримує ОДНІ й ті самі global_weights (w_t) — client.fit
        # спочатку записує їх у робочу модель, тому попередній клієнт не впливає
        # на наступного. strategy.proximal_mu > 0 лише для FedProx.
        results = []
        for client in clients:
            res = client.fit(model, global_weights, keys, device, rnd, proximal_mu=strategy.proximal_mu)
            results.append(res)

        # Для останнього раунду додатково оцінюємо ЛОКАЛЬНІ моделі клієнтів
        # на спільному тесті — це показує «дрейф» клієнтів до агрегації.
        # Напр., у сценарії missing_classes локальна модель клієнта 1 не вміє
        # розпізнавати Lepidoptera / Odonata, а глобальна після агрегації — вміє.
        local_evals = {}
        if rnd == num_rounds:
            for res in results:
                set_weights(model, res["weights"])
                le = evaluate(model, test_set, device)
                local_evals[res["client_id"]] = le

        # c) агрегація на сервері
        # Стратегія отримує старі глобальні ваги w_t (потрібні FedAdam для
        # обчислення псевдоградієнта) та результати клієнтів, повертає w_{t+1}.
        global_weights = strategy.aggregate(global_weights, results)
        set_weights(model, global_weights)

        # d) оцінка глобальної моделі
        ev = evaluate(model, test_set, device)
        # У лог записуємо метрики клієнтів БЕЗ ваг (ваги важать десятки МБ і для
        # аналізу не потрібні).
        round_info = {
            "round": rnd, **ev,
            "clients": [
                {k: r[k] for k in ("client_id", "num_examples", "train_loss", "det_loss",
                                   "prox_loss", "num_batches", "time_sec")}
                for r in results
            ],
            "time_sec": time.time() - t0,
        }
        if local_evals:
            # Ключі JSON мають бути рядками, тому client_id перетворюємо на str.
            round_info["local_models_eval"] = {str(k): v for k, v in local_evals.items()}
        history["rounds"].append(round_info)

        losses = ", ".join(f"к{r['client_id']}={r['train_loss']:.3f}" for r in results)
        print(f"  Раунд {rnd:2d}: mAP@0.5={ev['map_50']:.4f}  mAP@0.5:0.95={ev['map']:.4f}  "
              f"loss[{losses}]  ({round_info['time_sec']:.0f} c)", flush=True)

        # Ваги клієнтів після агрегації більше не потрібні — звільняємо пам'ять.
        del results
        _free_memory(device)
        # Проміжне збереження: якщо процес перервуть, пройдені раунди не втратяться.
        if out_path is not None:
            _save_json(history, out_path)

    history["total_time_sec"] = time.time() - t_start
    if out_path is not None:
        # Прапорець finished відрізняє завершений експеримент від перерваного:
        # run_experiments.py пропускає лише ті, де він встановлений.
        history["finished"] = True
        _save_json(history, out_path)

    if visualize:
        # Локальний імпорт: matplotlib завантажується лише тоді, коли він потрібен.
        from .visualize import save_predictions
        save_predictions(model, test_set, device,
                         C.FIGURES_DIR / f"pred_{strategy_name}_{scenario}.png",
                         title=f"{strategy_name} / {scenario}")

    del model, global_weights, clients, strategy
    _free_memory(device)
    return history


def run_centralized(device: torch.device, epochs: int = C.NUM_ROUNDS,
                    out_path: Path | None = None, cache: dict | None = None) -> dict:
    """Базова лінія: централізоване навчання на ВСІХ навчальних даних
    (ніби один клієнт має весь набір). Використовується лише для порівняння —
    у реальному FL такий варіант неможливий через приватність даних.

    Одна епоха тут відповідає одному раунду FL (за раунд кожен клієнт теж
    проходить свої дані один раз), тому графіки можна накладати на одну вісь.
    Це «верхня межа», до якої прагне якість федеративних стратегій.
    """
    cache = cache or load_cache()
    # Перевикористовуємо FLClient з усіма даними: той самий цикл навчання
    # (оптимізатор, батчі, аугментація) гарантує чесне порівняння з FL.
    full = FLClient(0, ArthropodDetectionDataset(cache["train"], train=True))
    test_set = ArthropodDetectionDataset(cache["test"])
    model = create_model().to(device)
    keys = trainable_keys(model)
    weights = get_weights(model, keys)

    history = {"strategy": "centralized", "scenario": "all_data", "rounds": []}
    ev = evaluate(model, test_set, device)
    history["rounds"].append({"round": 0, **ev})
    print("\n=== ЦЕНТРАЛІЗОВАНЕ НАВЧАННЯ (базова лінія) ===")
    for ep in range(1, epochs + 1):
        t0 = time.time()
        # Ваги після епохи одразу стають стартовими для наступної (агрегації немає).
        # Примітка: fit() щоепохи створює новий SGD, тож momentum скидається — так
        # само, як і в FL-клієнтів між раундами.
        res = full.fit(model, weights, keys, device, ep)
        weights = res["weights"]
        set_weights(model, weights)
        ev = evaluate(model, test_set, device)
        history["rounds"].append({"round": ep, **ev, "train_loss": res["train_loss"],
                                  "time_sec": time.time() - t0})
        print(f"  Епоха {ep:2d}: mAP@0.5={ev['map_50']:.4f}  mAP@0.5:0.95={ev['map']:.4f}  "
              f"loss={res['train_loss']:.3f}", flush=True)
        if out_path is not None:
            _save_json(history, out_path)
    history["finished"] = True
    if out_path is not None:
        _save_json(history, out_path)
    del model
    _free_memory(device)
    return history


def _save_json(obj: dict, path: Path) -> None:
    """Атомарно записує словник у JSON.

    Спочатку пишемо у тимчасовий файл, потім перейменовуємо його в цільовий.
    Перейменування в межах однієї файлової системи — атомарна операція, тому
    навіть якщо процес буде вбито посеред запису, на диску лишиться або стара,
    або нова повна версія файлу, але ніколи не «обрізаний» JSON.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp")
    with open(tmp, "w", encoding="utf-8") as f:
        # ensure_ascii=False — кирилиця зберігається як є, а не як \uXXXX.
        json.dump(obj, f, ensure_ascii=False, indent=1)
    tmp.replace(path)  # атомарний запис: файл не зіпсується при перериванні
