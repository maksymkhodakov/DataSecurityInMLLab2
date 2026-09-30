"""
Серверні стратегії агрегації федеративного навчання.

Позначення:
    w_t          – глобальні ваги на початку раунду t;
    w_t^k        – ваги клієнта k після локального навчання;
    n_k          – кількість прикладів клієнта k,  n = Σ n_k.

FedAvg  (McMahan et al., 2017):
    w_{t+1} = Σ_k (n_k / n) · w_t^k                 – зважене середнє.

FedProx (Li et al., 2020):
    агрегація така сама, як у FedAvg, але клієнти мінімізують
    F_k(w) + (mu/2)·||w − w_t||²  (див. client.py). Серверна частина = FedAvg.

FedAdam (Reddi et al., 2021, «Adaptive Federated Optimization»):
    Δ_t     = Σ_k (n_k / n) · (w_t^k − w_t)         – «псевдоградієнт»;
    m_t     = β1 · m_{t−1} + (1 − β1) · Δ_t;
    v_t     = β2 · v_{t−1} + (1 − β2) · Δ_t²;
    w_{t+1} = w_t + η · m_t / (√v_t + τ).
    Тобто сервер застосовує оптимізатор Adam до усередненого оновлення клієнтів,
    що дає адаптивний (покоординатний) крок на стороні сервера.
"""
from __future__ import annotations

from collections import OrderedDict

import torch

from . import config as C


def weighted_average(results: list[dict]) -> "OrderedDict[str, torch.Tensor]":
    """Зважене (за кількістю прикладів) середнє ваг клієнтів."""
    total = sum(r["num_examples"] for r in results)
    keys = results[0]["weights"].keys()
    avg = OrderedDict()
    for k in keys:
        avg[k] = sum(r["weights"][k].float() * (r["num_examples"] / total) for r in results)
    return avg


class Strategy:
    """Базовий клас стратегії."""
    name = "base"
    proximal_mu = 0.0  # > 0 лише у FedProx: сервер повідомляє клієнтам mu

    def aggregate(self, global_weights, results):
        raise NotImplementedError


class FedAvg(Strategy):
    name = "fedavg"

    def aggregate(self, global_weights, results):
        return weighted_average(results)


class FedProx(FedAvg):
    """Серверна агрегація як у FedAvg; відмінність — у локальній цілі клієнтів."""
    name = "fedprox"

    def __init__(self, mu: float = C.FEDPROX_MU):
        self.proximal_mu = mu


class FedAdam(Strategy):
    name = "fedadam"

    def __init__(self, eta: float = C.FEDADAM_ETA, beta1: float = C.FEDADAM_BETA1,
                 beta2: float = C.FEDADAM_BETA2, tau: float = C.FEDADAM_TAU):
        self.eta, self.beta1, self.beta2, self.tau = eta, beta1, beta2, tau
        self.m: dict[str, torch.Tensor] | None = None   # перший момент (на сервері)
        self.v: dict[str, torch.Tensor] | None = None   # другий момент (на сервері)

    def aggregate(self, global_weights, results):
        avg = weighted_average(results)
        # Псевдоградієнт: наскільки в середньому клієнти змістили ваги
        delta = {k: avg[k] - global_weights[k] for k in global_weights}

        if self.m is None:
            self.m = {k: torch.zeros_like(v) for k, v in delta.items()}
            self.v = {k: torch.zeros_like(v) for k, v in delta.items()}

        new_weights = OrderedDict()
        for k, d in delta.items():
            self.m[k].mul_(self.beta1).add_(d, alpha=1 - self.beta1)
            self.v[k].mul_(self.beta2).addcmul_(d, d, value=1 - self.beta2)
            new_weights[k] = global_weights[k] + self.eta * self.m[k] / (self.v[k].sqrt() + self.tau)
        return new_weights


def make_strategy(name: str) -> Strategy:
    return {"fedavg": FedAvg, "fedadam": FedAdam, "fedprox": FedProx}[name]()
