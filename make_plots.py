"""
Крок 3. Побудова графіків прогресу навчання та зведених таблиць.

Читає results/logs/*.json і створює в results/figures/:
  fig_partitions.png       – розподіл класів між клієнтами для 4 сценаріїв;
  fig_map50_progress.png   – mAP@0.5 глобальної моделі по раундах (по сценаріях);
  fig_map_progress.png     – mAP@0.5:0.95 по раундах;
  fig_client_loss.png      – локальні втрати клієнтів по раундах;
  fig_final_comparison.png – підсумкове порівняння стратегій;
  fig_per_class_ap50.png   – AP@0.5 по класах у non-IID сценаріях;
  fig_local_vs_global.png  – локальні моделі клієнтів vs глобальна (сценарій missing_classes);
  fig_fedadam_tuning.png   – підбір η для FedAdam.
Також results/summary.csv та results/summary.json – підсумкові таблиці.
"""
import csv
import json

import matplotlib
matplotlib.use("Agg")  # неінтерактивний бекенд: графіки лише зберігаються у файли
import matplotlib.pyplot as plt
import numpy as np

from fl_lab import config as C
from fl_lab.data import load_cache
from fl_lab.partition import SCENARIO_TITLES, split_indices
from fl_lab.data import ArthropodDetectionDataset

# Фіксована відповідність «стратегія → колір / маркер» в усіх графіках
STRAT_STYLE = {
    "fedavg":  {"color": "#2a78d6", "marker": "o", "label": "FedAvg"},
    "fedadam": {"color": "#eb6834", "marker": "s", "label": "FedAdam"},
    "fedprox": {"color": "#1baf7a", "marker": "^", "label": "FedProx"},
}
# Базова лінія (централізоване навчання) — сірий пунктир, щоб не конкурувати з FL-стратегіями
CENTRAL_STYLE = {"color": "#7a7a7a", "linestyle": "--", "label": "Централізоване"}
# Кольори клієнтів однакові на всіх графіках (розбиття, втрати, локальні моделі)
CLIENT_COLORS = {1: "#2a78d6", 2: "#e87ba4"}

# Спільний мінімалістичний стиль: без верхньої/правої рамки, світла сітка, легенда без рамки
plt.rcParams.update({
    "font.size": 10, "axes.spines.top": False, "axes.spines.right": False,
    "axes.grid": True, "grid.color": "#e6e6e6", "grid.linewidth": 0.8,
    "axes.edgecolor": "#9a9a9a", "legend.frameon": False,
})


def load_logs():
    """Зчитує JSON-логи всіх експериментів.

    Повертає (logs, central), де logs — словник {(стратегія, сценарій): історія},
    а central — історія централізованої базової лінії або None. Відсутні логи
    просто пропускаються, тому графіки можна будувати й за частковими результатами.
    """
    logs = {}
    for s in C.STRATEGIES:
        for sc in C.SCENARIOS:
            p = C.LOGS_DIR / f"{s}__{sc}.json"
            if p.exists():
                logs[(s, sc)] = json.load(open(p, encoding="utf-8"))
    central = None
    p = C.LOGS_DIR / "centralized.json"
    if p.exists():
        central = json.load(open(p, encoding="utf-8"))
    return logs, central


def savefig(fig, name):
    # bbox_inches="tight" обрізає зайві поля і не дає винесеній легенді вийти за межі файлу
    C.FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    fig.savefig(C.FIGURES_DIR / name, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print("  збережено", name)


# ----------------------------------------------------------------------------------
def plot_partitions():
    """Гістограми класів у клієнтів для кожного сценарію (наочно показує non-IID).
    Розбиття перераховується тією ж функцією split_indices з тим самим seed,
    тому воно точно збігається з використаним в експериментах."""
    cache = load_cache()
    fig, axes = plt.subplots(2, 2, figsize=(11, 6.5), sharey=True)
    x = np.arange(len(C.CLASSES))
    for ax, sc in zip(axes.flat, C.SCENARIOS):
        parts = split_indices(cache["train"], sc)
        for k, idxs in enumerate(parts, start=1):
            hist = ArthropodDetectionDataset(cache["train"], idxs).class_histogram()
            # Зсув ±0.2 від центру мітки класу: стовпчики двох клієнтів стоять поруч
            ax.bar(x + (k - 1.5) * 0.4, hist, width=0.38, color=CLIENT_COLORS[k],
                   label=f"Клієнт {k} (n={len(idxs)})")
        ax.set_title(SCENARIO_TITLES[sc], fontsize=10)
        ax.set_xticks(x, C.CLASSES, rotation=30, ha="right", fontsize=8)
        ax.legend(fontsize=8, loc="upper center")
    for ax in axes[:, 0]:
        ax.set_ylabel("Кількість зображень")
    fig.tight_layout()
    savefig(fig, "fig_partitions.png")


def plot_progress(logs, central, key, ylabel, fname):
    """Метрика `key` глобальної моделі по раундах: 4 підграфіки (по сценарію),
    на кожному — три стратегії та централізована базова лінія.
    sharey=True — спільна вісь Y, щоб сценарії можна було порівнювати між собою."""
    fig, axes = plt.subplots(2, 2, figsize=(11, 7), sharey=True)
    for ax, sc in zip(axes.flat, C.SCENARIOS):
        for s in C.STRATEGIES:
            if (s, sc) not in logs:
                continue
            r = logs[(s, sc)]["rounds"]
            st = STRAT_STYLE[s]
            ax.plot([e["round"] for e in r], [e[key] for e in r], color=st["color"],
                    marker=st["marker"], markersize=5, linewidth=2, label=st["label"])
        if central:
            r = central["rounds"]
            ax.plot([e["round"] for e in r], [e[key] for e in r], color=CENTRAL_STYLE["color"],
                    linestyle="--", linewidth=1.5, label=CENTRAL_STYLE["label"])
        ax.set_title(SCENARIO_TITLES[sc], fontsize=10)
        ax.set_xlabel("Раунд федеративного навчання")
        ax.set_xticks(range(0, C.NUM_ROUNDS + 1))
    for ax in axes[:, 0]:
        ax.set_ylabel(ylabel)
    # Одна спільна легенда під усіма підграфіками замість чотирьох однакових
    handles, labels = axes.flat[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=4, bbox_to_anchor=(0.5, -0.02))
    fig.tight_layout(rect=(0, 0.04, 1, 1))
    savefig(fig, fname)


def plot_client_loss(logs):
    """Сітка 3 x 4 (стратегія x сценарій): локальна втрата детекції кожного клієнта.
    Використовується det_loss (без проксимального члена), щоб FedProx можна було
    чесно порівнювати з іншими стратегіями."""
    fig, axes = plt.subplots(3, 4, figsize=(14, 8.5), sharex=True)
    for i, s in enumerate(C.STRATEGIES):
        for j, sc in enumerate(C.SCENARIOS):
            ax = axes[i, j]
            if (s, sc) not in logs:
                ax.set_visible(False)
                continue
            # Раунд 0 — лише оцінка стартової моделі, клієнти ще не навчались → пропускаємо
            rounds = [r for r in logs[(s, sc)]["rounds"] if r["clients"]]
            for cid in (1, 2):
                ax.plot([r["round"] for r in rounds],
                        [next(c for c in r["clients"] if c["client_id"] == cid)["det_loss"] for r in rounds],
                        color=CLIENT_COLORS[cid], marker="o", markersize=4, linewidth=2,
                        label=f"Клієнт {cid}")
            if i == 0:
                ax.set_title(SCENARIO_TITLES[sc], fontsize=9)
            if j == 0:
                ax.set_ylabel(f"{STRAT_STYLE[s]['label']}\nлокальна втрата")
            if i == 2:
                ax.set_xlabel("Раунд")
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=2, bbox_to_anchor=(0.5, -0.01))
    fig.tight_layout(rect=(0, 0.03, 1, 1))
    savefig(fig, "fig_client_loss.png")


def plot_final_comparison(logs, central):
    """Згруповані стовпчики: mAP@0.5 після останнього раунду для кожної
    пари (сценарій, стратегія) + горизонтальна лінія централізованої моделі."""
    fig, ax = plt.subplots(figsize=(10, 4.5))
    x = np.arange(len(C.SCENARIOS))
    w = 0.26  # ширина групи з 3 стовпчиків ≈ 0.78, між групами лишається проміжок
    for k, s in enumerate(C.STRATEGIES):
        # Відсутній експеримент відображаємо як 0, щоб не зсувати групи
        vals = [logs[(s, sc)]["rounds"][-1]["map_50"] if (s, sc) in logs else 0 for sc in C.SCENARIOS]
        bars = ax.bar(x + (k - 1) * w, vals, width=w - 0.02, color=STRAT_STYLE[s]["color"],
                      label=STRAT_STYLE[s]["label"])
        # Точні значення над стовпчиками — зручно для посилань у звіті
        ax.bar_label(bars, fmt="%.3f", fontsize=7, padding=2)
    if central:
        ax.axhline(central["rounds"][-1]["map_50"], color=CENTRAL_STYLE["color"], linestyle="--",
                   linewidth=1.2, label="Централізоване")
    ax.set_xticks(x, [SCENARIO_TITLES[sc].replace(", ", ",\n") for sc in C.SCENARIOS], fontsize=8)
    ax.set_ylabel("mAP@0.5 після останнього раунду")
    ax.legend(ncol=4, loc="upper center", bbox_to_anchor=(0.5, -0.16))
    savefig(fig, "fig_final_comparison.png")


def plot_per_class(logs):
    """AP@0.5 по класах у двох найбільш неоднорідних сценаріях — показує,
    які саме класи «страждають» від дисбалансу чи відсутності у клієнта."""
    scen = ["unequal_imbalanced", "missing_classes"]
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.2), sharey=True)
    x = np.arange(len(C.CLASSES))
    w = 0.26
    for ax, sc in zip(axes, scen):
        for k, s in enumerate(C.STRATEGIES):
            if (s, sc) not in logs:
                continue
            ax.bar(x + (k - 1) * w, logs[(s, sc)]["rounds"][-1]["ap50_per_class"], width=w - 0.02,
                   color=STRAT_STYLE[s]["color"], label=STRAT_STYLE[s]["label"])
        ax.set_title(SCENARIO_TITLES[sc], fontsize=10)
        ax.set_xticks(x, C.CLASSES, rotation=30, ha="right", fontsize=8)
    axes[0].set_ylabel("AP@0.5 глобальної моделі")
    axes[0].legend(fontsize=8)
    fig.tight_layout()
    savefig(fig, "fig_per_class_ap50.png")


def plot_local_vs_global(logs, strategy="fedavg", sc="missing_classes"):
    """Порівнює AP@0.5 по класах для локальних моделей клієнтів (до агрегації)
    і глобальної моделі (після). У сценарії missing_classes локальна модель
    не знає «чужих» класів, а агрегація об'єднує знання обох клієнтів."""
    if (strategy, sc) not in logs:
        return
    last = logs[(strategy, sc)]["rounds"][-1]
    # local_models_eval записується сервером лише в останньому раунді (див. server.py)
    loc = last.get("local_models_eval", {})
    fig, ax = plt.subplots(figsize=(10, 4))
    x = np.arange(len(C.CLASSES))
    w = 0.26
    series = [("Локальна модель клієнта 1", loc.get("1", {}).get("ap50_per_class"), CLIENT_COLORS[1]),
              ("Локальна модель клієнта 2", loc.get("2", {}).get("ap50_per_class"), CLIENT_COLORS[2]),
              ("Глобальна модель (після агрегації)", last["ap50_per_class"], "#1baf7a")]
    for k, (lab, vals, col) in enumerate(series):
        if vals is None:
            continue
        ax.bar(x + (k - 1) * w, vals, width=w - 0.02, color=col, label=lab)
    ax.set_xticks(x, C.CLASSES, rotation=30, ha="right", fontsize=8)
    ax.set_ylabel("AP@0.5")
    ax.set_title(f"{STRAT_STYLE[strategy]['label']}, сценарій «{SCENARIO_TITLES[sc]}», "
                 f"останній раунд", fontsize=10)
    ax.legend(fontsize=8, ncol=3, loc="upper center", bbox_to_anchor=(0.5, -0.25))
    savefig(fig, "fig_local_vs_global.png")


def plot_tuning():
    """Криві mAP@0.5 по раундах для різних η FedAdam (результат tune_fedadam.py)."""
    p = C.LOGS_DIR / "tuning" / "summary.json"
    if not p.exists():
        return
    summ = json.load(open(p, encoding="utf-8"))
    fig, ax = plt.subplots(figsize=(6, 3.8))
    shades = ["#9ec5f4", "#3987e5", "#104281"]  # одна гама: світліший = менший η
    # Ключі JSON — рядки, тому сортуємо за числовим значенням η
    for (eta, vals), col in zip(sorted(summ.items(), key=lambda kv: float(kv[0])), shades):
        ax.plot(range(len(vals)), vals, marker="o", color=col, linewidth=2, label=f"η = {float(eta):g}")
    ax.set_xlabel("Раунд")
    ax.set_ylabel("mAP@0.5")
    ax.set_xticks(range(len(vals)))
    ax.legend()
    savefig(fig, "fig_fedadam_tuning.png")


def write_summary(logs, central):
    """Зведена таблиця результатів (CSV для Excel/звіту і JSON для програмної обробки)."""
    rows = []
    for sc in C.SCENARIOS:
        for s in C.STRATEGIES:
            if (s, sc) not in logs:
                continue
            h = logs[(s, sc)]
            r = h["rounds"]
            best = max(r, key=lambda e: e["map_50"])
            # Швидкість збіжності: перший раунд, на якому mAP@0.5 досяг 0.3 (None — не досяг)
            reach = next((e["round"] for e in r if e["map_50"] >= 0.3), None)
            last_clients = r[-1]["clients"]
            rows.append({
                "scenario": sc, "strategy": s,
                "n_client1": h["clients"][0]["num_examples"], "n_client2": h["clients"][1]["num_examples"],
                "final_map50": round(r[-1]["map_50"], 4), "final_map": round(r[-1]["map"], 4),
                "final_mar100": round(r[-1]["mar_100"], 4),
                # середнє за 3 останні раунди — менш чутливе до коливань на малому тесті
                "last3_map50": round(float(np.mean([e["map_50"] for e in r[-3:]])), 4),
                "best_map50": round(best["map_50"], 4), "best_round": best["round"],
                "round_map50_ge_0.3": reach,
                "final_loss_c1": round(last_clients[0]["det_loss"], 4),
                "final_loss_c2": round(last_clients[1]["det_loss"], 4),
                "time_min": round(h.get("total_time_sec", 0) / 60, 1),
            })
    if central:
        r = central["rounds"]
        rows.append({"scenario": "all_data", "strategy": "centralized",
                     "final_map50": round(r[-1]["map_50"], 4), "final_map": round(r[-1]["map"], 4),
                     "final_mar100": round(r[-1]["mar_100"], 4),
                     "last3_map50": round(float(np.mean([e["map_50"] for e in r[-3:]])), 4),
                     "best_map50": round(max(e["map_50"] for e in r), 4)})
    if not rows:
        return
    # Порядок стовпців береться з першого рядка (рядок базової лінії має менше полів —
    # DictWriter залишить відсутні комірки порожніми)
    keys = list(rows[0].keys())
    with open(C.RESULTS_DIR / "summary.csv", "w", newline="", encoding="utf-8") as f:
        wr = csv.DictWriter(f, fieldnames=keys)
        wr.writeheader()
        wr.writerows(rows)
    json.dump(rows, open(C.RESULTS_DIR / "summary.json", "w", encoding="utf-8"), ensure_ascii=False, indent=1)
    print("  збережено summary.csv / summary.json")


if __name__ == "__main__":
    logs, central = load_logs()
    print(f"Знайдено експериментів: {len(logs)}; базова лінія: {'так' if central else 'ні'}")
    plot_partitions()  # не залежить від логів — лише від кешу даних
    if logs:
        plot_progress(logs, central, "map_50", "mAP@0.5 (тест)", "fig_map50_progress.png")
        plot_progress(logs, central, "map", "mAP@0.5:0.95 (тест)", "fig_map_progress.png")
        plot_client_loss(logs)
        plot_final_comparison(logs, central)
        plot_per_class(logs)
        plot_local_vs_global(logs)
    plot_tuning()
    write_summary(logs, central)
