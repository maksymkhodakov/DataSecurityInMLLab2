"""
Розбиття навчального набору на ДВІ частини (по одній на кожного клієнта/воркера).

Реалізовано 4 сценарії з умови лабораторної роботи:

  1. iid_equal          – частини однакові за розміром і збалансовані по класах
                          (кожен клас ділиться 50 / 50);
  2. unequal_balanced   – частини різного розміру, але збалансовані по класах
                          (кожен клас ділиться 75 / 25 → пропорції класів однакові);
  3. unequal_imbalanced – частини різного розміру та з дисбалансом класів
                          (частка класу у клієнта 1 змінюється від 95 % до 15 %);
  4. missing_classes    – кожній частині бракує щонайменше двох класів
                          (клієнт 1 не бачить Lepidoptera та Odonata,
                           клієнт 2 не бачить Araneae та Coleoptera).

Функції повертають лише СПИСКИ ІНДЕКСІВ. Далі кожен клієнт отримує Dataset,
побудований тільки зі своїх індексів.
"""
from __future__ import annotations

import random
from collections import defaultdict

from . import config as C

# Частка зображень кожного класу, яка потрапляє до клієнта 1 (решта → клієнт 2).
# Порядок відповідає config.CLASSES:
#   Araneae, Coleoptera, Diptera, Hemiptera, Hymenoptera, Lepidoptera, Odonata
SCENARIO_FRACTIONS: dict[str, list[float]] = {
    "iid_equal":          [0.50, 0.50, 0.50, 0.50, 0.50, 0.50, 0.50],
    "unequal_balanced":   [0.75, 0.75, 0.75, 0.75, 0.75, 0.75, 0.75],
    "unequal_imbalanced": [0.95, 0.90, 0.85, 0.75, 0.60, 0.30, 0.15],
    # 1.0 – клас лише у клієнта 1, 0.0 – лише у клієнта 2, 0.5 – спільний
    "missing_classes":    [1.00, 1.00, 0.50, 0.50, 0.50, 0.00, 0.00],
}

SCENARIO_TITLES = {
    "iid_equal": "Однаковий розмір, збалансовані класи",
    "unequal_balanced": "Різний розмір, збалансовані класи",
    "unequal_imbalanced": "Різний розмір, дисбаланс класів",
    "missing_classes": "Кожній частині бракує ≥2 класів",
}


def split_indices(records: list[dict], scenario: str, seed: int = C.SEED) -> list[list[int]]:
    """Повертає [індекси_клієнта_1, індекси_клієнта_2] для заданого сценарію.

    Розбиття виконується окремо для кожного класу (стратифіковано), що дає
    точний контроль над розміром частин і розподілом класів.
    """
    if scenario not in SCENARIO_FRACTIONS:
        raise ValueError(f"Невідомий сценарій: {scenario}")
    fractions = SCENARIO_FRACTIONS[scenario]
    rng = random.Random(seed)

    # Групуємо індекси записів за класом
    by_class: dict[int, list[int]] = defaultdict(list)
    for idx, rec in enumerate(records):
        by_class[rec["class_idx"]].append(idx)

    client_1, client_2 = [], []
    for class_idx, idxs in sorted(by_class.items()):
        idxs = idxs[:]
        rng.shuffle(idxs)
        k = round(len(idxs) * fractions[class_idx])
        client_1.extend(idxs[:k])
        client_2.extend(idxs[k:])

    # Перемішуємо, щоб у локальних батчах класи йшли впереміш
    rng.shuffle(client_1)
    rng.shuffle(client_2)

    # Перевірка: частини не перетинаються і разом покривають весь набір
    assert not set(client_1) & set(client_2), "Частини клієнтів перетинаються!"
    assert len(client_1) + len(client_2) == len(records)
    return [client_1, client_2]
