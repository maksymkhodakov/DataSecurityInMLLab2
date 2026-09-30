"""
Допоміжний експеримент: підбір серверної швидкості навчання η для FedAdam.

FedAdam чутливий до η: занадто мале значення гальмує навчання (перші раунди
додатково «гасяться» моментами Adam), занадто велике — дестабілізує модель.
Порівнюємо кілька значень на 3 раундах у сценарії iid_equal і обираємо найкраще
для основних експериментів (значення потім фіксується в config.FEDADAM_ETA).

Запуск: python tune_fedadam.py
"""
import json

from fl_lab import config as C
from fl_lab import strategies
from fl_lab.data import load_cache
from fl_lab.model import get_device
from fl_lab.server import run_federated

ETAS = [3e-3, 1e-2, 3e-2]
ROUNDS = 3

if __name__ == "__main__":
    device = get_device()
    cache = load_cache()
    out_dir = C.LOGS_DIR / "tuning"
    out_dir.mkdir(parents=True, exist_ok=True)
    summary = {}
    for eta in ETAS:
        path = out_dir / f"fedadam_eta_{eta:g}.json"
        if path.exists() and json.load(open(path, encoding="utf-8")).get("finished"):
            hist = json.load(open(path, encoding="utf-8"))
        else:
            # Тимчасово підміняємо значення η за замовчуванням для FedAdam
            strategies.FedAdam.__init__.__defaults__ = (eta, C.FEDADAM_BETA1, C.FEDADAM_BETA2, C.FEDADAM_TAU)
            hist = run_federated("fedadam", "iid_equal", device, num_rounds=ROUNDS,
                                 out_path=path, cache=cache, visualize=False)
        summary[eta] = [r["map_50"] for r in hist["rounds"]]
        print(f"eta={eta:g}: mAP@0.5 по раундах = {[round(x, 4) for x in summary[eta]]}")
    with open(out_dir / "summary.json", "w", encoding="utf-8") as f:
        json.dump({str(k): v for k, v in summary.items()}, f, indent=1)
