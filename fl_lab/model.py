"""
Модель детекції об'єктів: Faster R-CNN з backbone MobileNetV3-Large + FPN (320 px).

Чому саме вона:
  * двоетапний детектор Faster R-CNN дає стабільну якість навіть на малій вибірці;
  * версія MobileNetV3-320 — найлегша з детекторів torchvision (~19 млн параметрів),
    її можна навчати на ноутбуці (Apple M1) без дискретної GPU;
  * є ваги, попередньо навчені на COCO, тому виконуємо transfer learning:
    backbone заморожено, навчаються FPN, RPN та ROI-голови, а класифікатор
    (box_predictor) замінено на новий під 7 класів + фон.

Будова Faster R-CNN (у порядку проходження зображення):
  1) backbone (MobileNetV3) – витягує карти ознак із зображення;
  2) FPN (Feature Pyramid Network) – комбінує ознаки різних масштабів,
     щоб знаходити як великі, так і малі об'єкти;
  3) RPN (Region Proposal Network) – пропонує прямокутні області,
     де «ймовірно є якийсь об'єкт» (без урахування класу);
  4) ROI-голови – для кожної пропозиції визначають клас і уточнюють рамку.

Також тут зібрано утиліти для роботи з вагами, які потрібні федеративному
навчанню: отримання / встановлення стану моделі та вибір параметрів,
що реально навчаються (саме їх клієнти пересилають серверу).
"""
from __future__ import annotations

from collections import OrderedDict

import torch
from torchvision.models.detection import fasterrcnn_mobilenet_v3_large_320_fpn
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor

from . import config as C


def get_device() -> torch.device:
    """Apple Silicon → MPS, NVIDIA → CUDA, інакше CPU."""
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def create_model(seed: int = C.SEED) -> torch.nn.Module:
    """Створює модель. Фіксований seed гарантує, що в УСІХ експериментах
    стартова глобальна модель однакова (чесне порівняння стратегій)."""
    # Випадково ініціалізується лише нова голова FastRCNNPredictor (нижче);
    # решта ваг береться з COCO. Seed робить цю ініціалізацію однаковою.
    torch.manual_seed(seed)
    model = fasterrcnn_mobilenet_v3_large_320_fpn(
        weights="DEFAULT",                              # ваги COCO (з локального кешу torch)
        trainable_backbone_layers=C.TRAINABLE_BACKBONE_LAYERS,
        # Зменшуємо кількість RoI-пропозицій під час навчання (за замовчуванням 2000):
        # на фото зазвичай 1–3 об'єкти, а це прискорює навчання приблизно вдвічі.
        #   pre_nms  – скільки найвпевненіших пропозицій RPN береться до NMS;
        #   post_nms – скільки лишається після придушення дублікатів (NMS);
        #   box_batch_size_per_image – скільки RoI на зображення використовується
        #   для обчислення втрат ROI-голови (за замовчуванням 512).
        rpn_pre_nms_top_n_train=300,
        rpn_post_nms_top_n_train=150,
        box_batch_size_per_image=128,
    )
    # Замінюємо «голову» класифікації COCO (91 клас) на нашу (7 класів + фон)
    # in_features — розмір вектора ознак, який ROI-голова подає на класифікатор;
    # FastRCNNPredictor створює два лінійні шари: класифікацію (NUM_CLASSES виходів)
    # і регресію рамок (4 координати на кожен клас).
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, C.NUM_CLASSES)
    return model


def trainable_keys(model: torch.nn.Module) -> list[str]:
    """Імена параметрів, які навчаються (requires_grad=True).
    Лише вони змінюються на клієнтах і лише їх потрібно агрегувати та пересилати.
    Заморожений backbone однаковий у всіх учасників і не передається по мережі."""
    return [name for name, p in model.named_parameters() if p.requires_grad]


def get_weights(model: torch.nn.Module, keys: list[str]) -> "OrderedDict[str, torch.Tensor]":
    """Копія навчуваних ваг на CPU (імітує «відправку» ваг на сервер)."""
    params = dict(model.named_parameters())
    # detach() – відриваємо від графа обчислень (градієнти не потрібні);
    # copy=True – обов'язково створюємо НЕЗАЛЕЖНУ копію: без неї на CPU-пристрої
    # повернувся б той самий тензор, і подальше навчання змінило б «відправлені» ваги.
    return OrderedDict((k, params[k].detach().to("cpu", copy=True)) for k in keys)


def set_weights(model: torch.nn.Module, weights: "OrderedDict[str, torch.Tensor]") -> None:
    """Записує отримані від сервера ваги у модель клієнта."""
    params = dict(model.named_parameters())
    # copy_ змінює параметр «на місці» (in-place), тому оптимізатор і модель
    # продовжують посилатися на ті самі об'єкти-параметри. no_grad потрібен,
    # бо in-place зміна листових тензорів з requires_grad інакше заборонена.
    with torch.no_grad():
        for k, v in weights.items():
            params[k].copy_(v.to(params[k].device))


def weights_nbytes(weights: "OrderedDict[str, torch.Tensor]") -> int:
    """Обсяг переданих ваг у байтах (для оцінки комунікаційних витрат)."""
    # numel() – кількість елементів тензора, element_size() – байт на елемент (4 для float32).
    return sum(v.numel() * v.element_size() for v in weights.values())
