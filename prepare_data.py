"""
Крок 1. Підготовка даних.

Сканує анотації ArTaxOr, відбирає стратифіковану підвибірку та зберігає
зменшені (320 px) зображення в компактний кеш data/artaxor_320.pkl.
Оригінальні файли датасету не змінюються і не копіюються.

Запуск:  python prepare_data.py            (або --force для перебудови кешу)
"""
import argparse

from fl_lab.data import build_cache

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Підготовка кешу датасету ArTaxOr")
    parser.add_argument("--force", action="store_true", help="перебудувати кеш")
    args = parser.parse_args()
    build_cache(force=args.force)
