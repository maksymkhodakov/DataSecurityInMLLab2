"""
Крок 2. Запуск експериментів федеративного навчання.

Для кожної стратегії (FedAvg, FedAdam, FedProx) перевіряються 4 сценарії
розбиття даних між двома клієнтами → 12 експериментів + централізована базова лінія.

Результати кожного експерименту: results/logs/<strategy>__<scenario>.json
Завершені експерименти при повторному запуску пропускаються (можна перервати
виконання Ctrl+C і продовжити пізніше без втрати результатів).

Приклади:
    python run_experiments.py                                  # усе
    python run_experiments.py --strategies fedavg --scenarios iid_equal
    python run_experiments.py --centralized                    # лише базова лінія
    python run_experiments.py --device cpu                     # примусово CPU
"""
import argparse
import json

import torch

from fl_lab import config as C
from fl_lab.data import load_cache
from fl_lab.model import get_device
from fl_lab.server import run_centralized, run_federated


def is_finished(path) -> bool:
    if not path.exists():
        return False
    with open(path, encoding="utf-8") as f:
        return json.load(f).get("finished", False)


def main():
    parser = argparse.ArgumentParser(description="Federated Learning на ArTaxOr")
    parser.add_argument("--strategies", nargs="+", default=C.STRATEGIES, choices=C.STRATEGIES)
    parser.add_argument("--scenarios", nargs="+", default=C.SCENARIOS, choices=C.SCENARIOS)
    parser.add_argument("--rounds", type=int, default=C.NUM_ROUNDS)
    parser.add_argument("--device", default=None, help="cpu / mps / cuda (за замовчуванням — автовибір)")
    parser.add_argument("--centralized", action="store_true", help="запустити лише централізовану базову лінію")
    parser.add_argument("--with-centralized", action="store_true", help="додатково запустити базову лінію")
    parser.add_argument("--threads", type=int, default=None, help="кількість потоків CPU для torch")
    args = parser.parse_args()

    if args.threads:
        torch.set_num_threads(args.threads)
    device = torch.device(args.device) if args.device else get_device()
    cache = load_cache()
    C.LOGS_DIR.mkdir(parents=True, exist_ok=True)

    if args.centralized or args.with_centralized:
        path = C.LOGS_DIR / "centralized.json"
        if is_finished(path):
            print("Базова лінія вже виконана — пропуск.")
        else:
            run_centralized(device, epochs=args.rounds, out_path=path, cache=cache)
        if args.centralized:
            return

    for scenario in args.scenarios:
        for strategy in args.strategies:
            path = C.LOGS_DIR / f"{strategy}__{scenario}.json"
            if is_finished(path):
                print(f"[пропуск] {strategy} / {scenario} вже виконано")
                continue
            run_federated(strategy, scenario, device, num_rounds=args.rounds, out_path=path, cache=cache)


if __name__ == "__main__":
    main()
