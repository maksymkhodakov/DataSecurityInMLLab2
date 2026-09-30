"""
Робота з набором даних ArTaxOr (Arthropod Taxonomy Orders Object Detection Dataset).

Структура оригінального датасету:
    ArTaxOr/<Клас>/<id>.jpg                      – зображення
    ArTaxOr/<Клас>/annotations/<hash>-asset.json – анотації у форматі VoTT:
        asset.name  – ім'я файлу зображення,
        asset.size  – оригінальні ширина / висота,
        regions[]   – список об'єктів: tags (назва класу + службові теги "_occluded",
                      "_truncated" тощо) та boundingBox (left, top, width, height).

Модуль виконує:
  1) розбір анотацій і відбір «чистих» зображень (усі об'єкти одного класу);
  2) стратифікований відбір підвибірки train/test;
  3) зменшення зображень до 320 px і збереження в компактний кеш (один pickle-файл
     зі стиснутими JPEG-байтами) — це економить і диск, і оперативну пам'ять;
  4) клас Dataset для PyTorch, який читає зображення тільки за переданими індексами
     (саме так реалізується ізоляція даних кожного клієнта).
"""
from __future__ import annotations

import io
import json
import pickle
import random
from collections import Counter
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset

from . import config as C


# ==================================================================================
# 1. Розбір анотацій
# ==================================================================================
def _parse_annotation(json_path: Path, class_name: str) -> dict | None:
    """Читає один JSON-файл VoTT і повертає опис зображення або None,
    якщо зображення не підходить (змішані класи, немає файлу, немає рамок)."""
    with open(json_path, "r", encoding="utf-8") as f:
        ann = json.load(f)

    img_path = json_path.parent.parent / ann["asset"]["name"]
    if not img_path.exists():
        return None

    boxes = []
    for region in ann.get("regions", []):
        # Службові теги починаються з "_" (напр. "_occluded"), їх ігноруємо.
        tags = [t for t in region["tags"] if not t.startswith("_")]
        # Беремо лише зображення, де ВСІ об'єкти належать до класу папки.
        # Це потрібно, щоб у сценарії «клієнт не має двох класів» ці класи
        # гарантовано не потрапили до клієнта через «сторонні» рамки.
        if len(tags) != 1 or tags[0] != class_name:
            return None
        bb = region["boundingBox"]
        boxes.append([bb["left"], bb["top"], bb["left"] + bb["width"], bb["top"] + bb["height"]])

    if not boxes:
        return None
    return {
        "path": str(img_path),
        "class_name": class_name,
        "width": ann["asset"]["size"]["width"],
        "height": ann["asset"]["size"]["height"],
        "boxes": boxes,
    }


def scan_dataset() -> dict[str, list[dict]]:
    """Сканує всі класи і повертає словник {клас: [описи придатних зображень]}."""
    per_class: dict[str, list[dict]] = {}
    for cls in C.CLASSES:
        ann_dir = C.DATASET_DIR / cls / "annotations"
        items = []
        for jp in sorted(ann_dir.glob("*.json")):
            rec = _parse_annotation(jp, cls)
            if rec is not None:
                items.append(rec)
        per_class[cls] = items
        print(f"  {cls:<12s}: придатних зображень {len(items)}")
    return per_class


# ==================================================================================
# 2. Зменшення зображень та побудова кешу
# ==================================================================================
def _load_and_resize(rec: dict) -> tuple[bytes, list[list[float]], tuple[int, int]]:
    """Відкриває велике JPEG-зображення, зменшує довшу сторону до IMAGE_SIZE,
    масштабує рамки та повертає стиснуті JPEG-байти."""
    img = Image.open(rec["path"])
    orig_w, orig_h = rec["width"], rec["height"]
    # Захист від невідповідності розміру файлу та анотації (напр. повернуте фото):
    # такі зображення пропускаємо, щоб не отримати зсунуті рамки.
    if img.size != (orig_w, orig_h):
        img.close()
        return b"", [], (0, 0)
    # draft() дозволяє декодеру JPEG одразу читати зображення у зменшеній роздільності
    # (1/2, 1/4, 1/8) — це в рази швидше та економніше по пам'яті, ніж повне декодування
    # фотографії 3000x2000.
    img.draft("RGB", (C.IMAGE_SIZE * 2, C.IMAGE_SIZE * 2))
    img = img.convert("RGB")

    # Анотації задані в координатах ОРИГІНАЛЬНОГО розміру (asset.size),
    # тому масштаб рахуємо відносно нього.
    scale = C.IMAGE_SIZE / max(orig_w, orig_h)
    new_w, new_h = max(1, round(orig_w * scale)), max(1, round(orig_h * scale))
    img = img.resize((new_w, new_h), Image.BILINEAR)

    boxes = []
    for x1, y1, x2, y2 in rec["boxes"]:
        b = [
            float(np.clip(x1 * scale, 0, new_w)), float(np.clip(y1 * scale, 0, new_h)),
            float(np.clip(x2 * scale, 0, new_w)), float(np.clip(y2 * scale, 0, new_h)),
        ]
        if b[2] - b[0] >= C.MIN_BOX_SIZE and b[3] - b[1] >= C.MIN_BOX_SIZE:
            boxes.append(b)

    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=C.JPEG_QUALITY)
    img.close()
    return buf.getvalue(), boxes, (new_w, new_h)


def build_cache(force: bool = False) -> Path:
    """Формує файл кешу з train/test підвибіркою.

    Структура кешу (dict):
        classes : список назв класів
        train   : список записів {jpeg, boxes, labels, class_idx, size, src}
        test    : аналогічно для спільного тестового набору сервера
    """
    if C.CACHE_FILE.exists() and not force:
        print(f"Кеш уже існує: {C.CACHE_FILE}")
        return C.CACHE_FILE

    C.CACHE_DIR.mkdir(parents=True, exist_ok=True)
    rng = random.Random(C.SEED)

    print("Сканування анотацій...")
    per_class = scan_dataset()

    cache = {"classes": C.CLASSES, "train": [], "test": []}
    for class_idx, cls in enumerate(C.CLASSES):
        items = per_class[cls][:]
        rng.shuffle(items)
        need = C.TRAIN_PER_CLASS + C.TEST_PER_CLASS
        chosen, i = [], 0
        # Беремо зображення по черзі, поки не наберемо потрібну кількість
        # (деякі можуть «відпасти», якщо після зменшення рамки стали занадто малими).
        while len(chosen) < need and i < len(items):
            jpeg, boxes, size = _load_and_resize(items[i])
            i += 1
            if not boxes:
                continue
            chosen.append({
                "jpeg": jpeg,
                "boxes": boxes,
                # мітка у Faster R-CNN = індекс класу + 1 (0 – фон)
                "labels": [class_idx + 1] * len(boxes),
                "class_idx": class_idx,
                "size": size,
                "src": Path(items[i - 1]["path"]).name,
            })
        cache["train"].extend(chosen[:C.TRAIN_PER_CLASS])
        cache["test"].extend(chosen[C.TRAIN_PER_CLASS:])
        print(f"  {cls:<12s}: train={len(chosen[:C.TRAIN_PER_CLASS])}, "
              f"test={len(chosen[C.TRAIN_PER_CLASS:])}")

    with open(C.CACHE_FILE, "wb") as f:
        pickle.dump(cache, f, protocol=pickle.HIGHEST_PROTOCOL)
    size_mb = C.CACHE_FILE.stat().st_size / 2**20
    print(f"Кеш збережено: {C.CACHE_FILE} ({size_mb:.1f} МБ)")
    return C.CACHE_FILE


def load_cache() -> dict:
    """Завантажує кеш (стиснуті JPEG займають лише кілька десятків МБ RAM)."""
    if not C.CACHE_FILE.exists():
        raise FileNotFoundError("Кеш не знайдено. Спочатку запустіть: python prepare_data.py")
    with open(C.CACHE_FILE, "rb") as f:
        return pickle.load(f)


# ==================================================================================
# 3. PyTorch Dataset
# ==================================================================================
class ArthropodDetectionDataset(Dataset):
    """Датасет для детекції. Зберігає ЛИШЕ ті записи, індекси яких йому передали.

    Саме через цей клас реалізовано приватність у симуляції федеративного навчання:
    клієнт отримує екземпляр датасету тільки зі своїми записами і фізично не має
    посилань на дані іншого клієнта. Сервер взагалі не отримує навчальних даних.
    """

    def __init__(self, records: list[dict], indices: list[int] | None = None, train: bool = False):
        if indices is None:
            indices = list(range(len(records)))
        # Копіюємо лише «свої» записи — список інших записів далі не зберігається.
        self.records = [records[i] for i in indices]
        self.train = train

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, idx: int):
        rec = self.records[idx]
        img = Image.open(io.BytesIO(rec["jpeg"])).convert("RGB")
        # PIL (H, W, 3) uint8 -> Tensor (3, H, W) float у діапазоні [0, 1].
        # Нормалізацію (mean/std) виконує сама модель Faster R-CNN (GeneralizedRCNNTransform).
        image = torch.from_numpy(np.asarray(img, dtype=np.uint8).copy()).permute(2, 0, 1).float() / 255.0
        boxes = torch.tensor(rec["boxes"], dtype=torch.float32)
        labels = torch.tensor(rec["labels"], dtype=torch.int64)

        # Аугментація для навчання: випадкове горизонтальне віддзеркалення.
        if self.train and random.random() < 0.5:
            image = image.flip(-1)
            w = image.shape[-1]
            boxes = boxes[:, [2, 1, 0, 3]]
            boxes[:, 0] = w - boxes[:, 0]
            boxes[:, 2] = w - boxes[:, 2]

        target = {"boxes": boxes, "labels": labels}
        return image, target

    def class_histogram(self) -> list[int]:
        """Кількість зображень кожного класу — для звіту про розподіл даних клієнта."""
        cnt = Counter(r["class_idx"] for r in self.records)
        return [cnt.get(i, 0) for i in range(len(C.CLASSES))]


def collate_fn(batch):
    """Детектори torchvision приймають списки зображень різного розміру,
    тому батч формується як кортеж списків, а не як один тензор."""
    return tuple(zip(*batch))
