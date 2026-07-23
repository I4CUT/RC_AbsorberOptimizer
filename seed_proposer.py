"""
seed_proposer.py — этап 4 дорожной карты (упрощённая, но рабочая версия)

Идея: вместо того чтобы гонять честный DE+TMM (десятки-сотни вызовов TMM)
для КАЖДОЙ последовательности материалов, сначала быстро прикинуть через уже
обученный forward-суррогат (ml_surrogate.SurrogateModel), какие
последовательности вообще перспективны для заданной цели (полоса частот +
порог RC + макс. толщина), и честно оптимизировать толщины (DE+TMM) только
для top-K лучших.

Важно — это НЕ отдельная ML-модель и не требует новых данных сверх того, что
уже есть: используется тот же суррогат, что и обычный режим «Нейросеть
(суррогат)» (см. ml_surrogate.py). Поэтому seed-proposer можно включать уже
сейчас, не дожидаясь роста таблицы design_targets — она остаётся полезной
как лог истории запросов и как будущий источник данных для более сложных
моделей (этап 5), но не блокирует эту, более простую и практичную версию.

Как и everywhere в этом проекте: seed-proposer только выбирает, ДЛЯ КАКИХ
последовательностей тратить дорогой честный DE+TMM. Он не может ухудшить
достоверность итоговых результатов — просто может пропустить хорошую
последовательность, если суррогат ошибся (сейчас это вероятно, пока модель
обучена на малом датасете; см. ml_surrogate.py и предыдущее обсуждение).

Используется ТОЛЬКО в сочетании: вкладка «Нейросеть (суррогат)» +
включённый Inverse Design (см. worker.py). Без suррогата или вне Inverse
Design поведение не меняется — используется обычный честный nsga2_optimise().
"""

from __future__ import annotations

import itertools
import logging
import random

import numpy as np

from physics import optimise_thicknesses, H_MIN

log = logging.getLogger("rc_opt")

# Ограничение на комбинаторный перебор: если материалов и слоёв много,
# полный перебор (num_materials ** n_layers) может быть неподъёмным —
# в этом случае берём случайную выборку вместо полного перебора.
MAX_SEQUENCES_TO_SCORE = 2000


def enumerate_sequences(num_materials: int, n_layers: int,
                        rng: random.Random | None = None) -> list:
    """
    Вернуть список всех последовательностей материалов длины n_layers
    (1-based индексы, как всюду в physics.py), либо случайную выборку из них,
    если полный перебор превышает MAX_SEQUENCES_TO_SCORE.
    """
    total = num_materials ** n_layers
    if total <= MAX_SEQUENCES_TO_SCORE:
        return list(itertools.product(range(1, num_materials + 1), repeat=n_layers))

    rng = rng or random.Random(42)
    seen = set()
    out = []
    while len(out) < MAX_SEQUENCES_TO_SCORE:
        seq = tuple(rng.randint(1, num_materials) for _ in range(n_layers))
        if seq not in seen:
            seen.add(seq)
            out.append(seq)
    log.info(f"  Seed-proposer: {total} возможных последовательностей — "
            f"случайная выборка {MAX_SEQUENCES_TO_SCORE} вместо полного перебора.")
    return out


def _score_sequence(sequence: tuple, model, material_scalar_by_material: dict,
                    h_max_total: float, rc_threshold: float,
                    n_trials: int, rng: random.Random) -> tuple:
    """
    Быстро (без единого вызова TMM) оценить последовательность: пробуем
    n_trials случайных наборов толщин в допустимых границах, для каждого
    предсказываем RC(f) суррогатом и берём лучший по штрафной функции —
    той же, что использует DE внутри physics.py (RC + штраф за превышение
    порога + вклад толщины), чтобы ранжирование было согласовано с тем,
    что будет дальше оптимизировать честный DE.

    Возвращает (score, best_hs_guess) — best_hs_guess пригождается как
    неплохая стартовая точка, но честный DE всё равно переоптимизирует
    толщины заново (см. worker.py) — здесь важна только относительная
    оценка last, чтобы отобрать top-K последовательностей.
    """
    n = len(sequence)
    h_max_per = h_max_total / n
    mat_scalar_layers = np.stack([material_scalar_by_material[mi] for mi in sequence])

    best_score = float("inf")
    best_hs = None
    for _ in range(n_trials):
        hs = np.array([rng.uniform(H_MIN, h_max_per) for _ in range(n)])
        thickness_mm = hs * 1000.0
        rc_curve = model.predict_curve(mat_scalar_layers, thickness_mm)
        max_rc = float(np.max(rc_curve))
        total_mm = float(np.sum(thickness_mm))
        penalty = 5.0 * max(0.0, max_rc - rc_threshold)
        score = max_rc + 0.3 * total_mm / 100.0 + penalty  # веса как в w_rc=1,w_th≈0.3
        if score < best_score:
            best_score = score
            best_hs = hs
    return best_score, best_hs


def propose_and_verify(n_layers: int, num_materials: int,
                       freqs_target: np.ndarray, mat_arrays_target: dict,
                       material_datas: list,
                       h_max_total: float,
                       rc_threshold: float,
                       surrogate_model,
                       rc_evaluator,
                       top_k: int = 15,
                       n_trials_per_sequence: int = 8,
                       w_rc: float = 1.0, w_th: float = 0.3,
                       _cancel_flag=None) -> list:
    """
    Главная функция seed-proposer'а. Заменяет вызов physics.nsga2_optimise()
    для одного значения n_layers, когда одновременно активны суррогат и
    Inverse Design (см. worker.py).

    1. Перебирает (или сэмплирует) последовательности материалов длины n_layers.
    2. Быстро ранжирует их суррогатом (без TMM).
    3. Для top_k лучших запускает ЧЕСТНЫЙ DE (physics.optimise_thicknesses,
       с rc_evaluator — обычно тем же суррогатом, для скорости самого DE —
       но итоговый результат в worker.py всё равно честно перепроверяется
       настоящим TMM, как и всегда).

    Возвращает список (sequence, hs, rc, th) — тот же формат, что и
    physics.nsga2_optimise(), поэтому worker.py не требует изменений в
    дальнейшей обработке результатов.
    """
    from ml_surrogate import material_scalar_features

    rng = random.Random(42)

    # Скалярные признаки каждого материала считаем один раз
    material_scalar_by_material = {
        i + 1: material_scalar_features(m["er"], m["eim"], m["mr"], m["mim"])
        for i, m in enumerate(material_datas)
    }

    sequences = enumerate_sequences(num_materials, n_layers, rng)
    log.info(f"  Seed-proposer: оцениваю {len(sequences)} последовательностей "
            f"суррогатом (n_layers={n_layers})…")

    scored = []
    for seq in sequences:
        if _cancel_flag is not None and _cancel_flag.is_set():
            return []
        score, hs_guess = _score_sequence(
            seq, surrogate_model, material_scalar_by_material,
            h_max_total, rc_threshold, n_trials_per_sequence, rng)
        scored.append((score, seq, hs_guess))

    scored.sort(key=lambda x: x[0])
    top = scored[:top_k]
    log.info(f"  Seed-proposer: отобрано {len(top)} лучших последовательностей "
            f"для честного DE (из {len(scored)} оценённых).")

    h_max_per = h_max_total / n_layers
    candidates = []
    for _, seq, _hs_guess in top:
        if _cancel_flag is not None and _cancel_flag.is_set():
            return candidates
        try:
            hs, rc, th = optimise_thicknesses(
                seq, freqs_target, mat_arrays_target,
                H_MIN, h_max_per, w_rc=w_rc, w_th=w_th,
                rc_threshold=rc_threshold, rc_evaluator=rc_evaluator)
            candidates.append((seq, hs, rc, th))
        except Exception as exc:
            log.warning(f"  Seed-proposer: DE не удался для seq={seq}: {exc}")

    return candidates
