# Лабораторна робота №2
## "Безпека даних в ML"
## Ходаков Максим Олегович ШІ-2

**Тема:** Federated Learning пайплайн навчання моделі детекції об'єктів на наборі
[ArTaxOr](https://www.kaggle.com/datasets/mistag/arthropod-taxonomy-orders-object-detection-dataset).

Модель: Faster R-CNN (MobileNetV3-Large-320 + FPN, ваги COCO). Два клієнти, кожен має
доступ лише до своєї частини даних. Стратегії: FedAvg, FedAdam, FedProx.
Сценарії розбиття: `iid_equal`, `unequal_balanced`, `unequal_imbalanced`, `missing_classes`.

### Структура
```
fl_lab/config.py      – усі гіперпараметри
fl_lab/data.py        – розбір анотацій VoTT, компактний кеш 320 px, Dataset
fl_lab/partition.py   – 4 сценарії розбиття на 2 клієнти
fl_lab/model.py       – модель та утиліти для ваг
fl_lab/client.py      – локальне навчання клієнта (+ проксимальний член FedProx)
fl_lab/strategies.py  – FedAvg / FedAdam / FedProx на сервері
fl_lab/server.py      – цикл раундів FL, оцінка, журнали JSON
fl_lab/evaluate.py    – mAP (COCO) на спільному тестовому наборі
fl_lab/visualize.py   – візуалізація передбачень
prepare_data.py       – крок 1: кеш даних (~18 МБ)
tune_fedadam.py       – підбір серверного lr для FedAdam
run_experiments.py    – крок 2: 12 експериментів + централізована базова лінія
make_plots.py         – крок 3: графіки та зведені таблиці
report/               – звіт (ДСТУ 3008:2015)
```

### Запуск
```bash
pip install -r requirements.txt
python prepare_data.py
python tune_fedadam.py                 # необов'язково
python run_experiments.py --with-centralized
python make_plots.py
```
Датасет розпакувати в `ArTaxOr/` у корені проєкту. Оригінальні файли не змінюються;
на диск пишуться лише кеш (~18 МБ), JSON-журнали та PNG-графіки (ваги моделей не зберігаються).
