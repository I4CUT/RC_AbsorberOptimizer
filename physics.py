"""
physics.py — физическая модель и алгоритмы оптимизации

Содержит:
  • TMM (Transfer Matrix Method) — расчёт коэффициента отражения
  • build_mat_arrays  — предвычисление γ, ν для всех материалов
  • NSGA-II           — внешний контур (дискретные последовательности)
  • Differential Evolution — внутренний контур (непрерывные толщины)
  • sensitivity_analysis, compute_bandwidth, rank_configurations
  • load_material_data, write_csv
"""

import os
import csv
import random
import logging
import time
import warnings

import numpy as np
import pandas as pd
from scipy.constants import c, mu_0, epsilon_0
from scipy.optimize import differential_evolution

warnings.filterwarnings("ignore")

log = logging.getLogger("rc_opt")

# ─────────────────────────────────────────────────────────────────────────────
# Физические константы
# ─────────────────────────────────────────────────────────────────────────────

Z0     = 377.0    # волновое сопротивление свободного пространства [Ом]
H_MIN  = 2e-4     # минимальная толщина слоя 0.2 мм [м]
H_GRID = 1e-4     # шаг квантования 0.1 мм [м]

RC_THRESHOLD = -10.0   # целевой уровень RC [дБ] (значение по умолчанию)
RC_PENALTY   =  5.0    # штраф за дБ выше порога

# Допустимые множители длины волны λ для ограничения суммарной толщины покрытия.
# UI предоставляет слайдер с этими фиксированными позициями.
LAMBDA_MULTIPLIERS = [1/8, 1/4, 1/2, 1.0, 2.0]
LAMBDA_MULTIPLIER_DEFAULT = 1.0
LAMBDA_MULTIPLIER_LABELS = {
    1/8: "λ/8", 1/4: "λ/4", 1/2: "λ/2", 1.0: "λ", 2.0: "2λ",
}


def validate_rc_threshold(value) -> float:
    """
    Проверить и привести значение порога RC к допустимому диапазону.
    Порог должен быть строго меньше 0 дБ (физически: меньше единичного
    отражения, иначе порог бессмысленен).

    Возвращает float. Вызывает ValueError при некорректном значении.
    """
    try:
        v = float(value)
    except (TypeError, ValueError):
        raise ValueError(f"Порог RC должен быть числом, получено: {value!r}")
    if v >= 0.0:
        raise ValueError(
            f"Порог RC должен быть меньше 0 дБ (получено {v} дБ). "
            f"0 дБ означает полное отражение — порог обязан быть отрицательным.")
    return v


def validate_lambda_multiplier(value) -> float:
    """
    Проверить, что множитель λ — одно из допустимых значений слайдера.
    Возвращает ближайшее допустимое значение, если точного совпадения нет
    (защита от ошибок округления float).
    """
    try:
        v = float(value)
    except (TypeError, ValueError):
        raise ValueError(f"Множитель λ должен быть числом, получено: {value!r}")
    closest = min(LAMBDA_MULTIPLIERS, key=lambda m: abs(m - v))
    if abs(closest - v) > 1e-6:
        log.warning(f"Множитель λ={v} не из допустимого набора, "
                   f"использую ближайший: {closest}")
    return closest

# ─────────────────────────────────────────────────────────────────────────────
# Кеши
# ─────────────────────────────────────────────────────────────────────────────

# Кеш тригонометрических функций: ключ = (material_idx, h_quantised)
_trig_cache: dict = {}

# Кеш оценки конфигураций: ключ = (sequence, hs_quantised)
_fitness_cache: dict = {}


def clear_caches() -> None:
    """Очистить оба кеша перед новым прогоном."""
    _trig_cache.clear()
    _fitness_cache.clear()


# ─────────────────────────────────────────────────────────────────────────────
# TMM — метод матриц передачи
# ─────────────────────────────────────────────────────────────────────────────

def tmm_reflection(freqs: np.ndarray,
                   sequence: tuple,
                   hs: tuple,
                   mat_arrays: dict) -> np.ndarray:
    """
    Векторизованный расчёт RC(f) методом TMM по всем частотным точкам.

    Граничное условие: PEC (Z_load = 0).
    Матрица слоя:
        M = [[cos(γh),       j·ν·sin(γh) ],
             [j/ν·sin(γh),   cos(γh)     ]]
    Zin = M[0,1] / M[1,1] = b / d
    RC  = 20·log10 |( Zin - Z0 ) / ( Zin + Z0 )|

    Параметры
    ---------
    freqs      : (F,) — сетка частот [Гц]
    sequence   : tuple int  — индексы материалов (1-based)
    hs         : tuple float — квантованные толщины [м]
    mat_arrays : dict из build_mat_arrays()

    Возвращает
    ----------
    RC : (F,) ndarray — коэффициент отражения [дБ]
    """
    F = len(freqs)
    a = np.ones(F,  dtype=complex)
    b = np.zeros(F, dtype=complex)
    c_ = np.zeros(F, dtype=complex)
    d = np.ones(F,  dtype=complex)

    for l in range(len(sequence)):
        mi  = sequence[l] - 1
        key = (mat_arrays['_cache_id'], mi, hs[l])
        if key in _trig_cache:
            cos_, sin_, nu = _trig_cache[key]
        else:
            gamma = mat_arrays['gamma'][mi]
            nu    = mat_arrays['nu'][mi]
            gh    = gamma * hs[l]
            cos_  = np.cos(gh)
            sin_  = np.sin(gh)
            _trig_cache[key] = (cos_, sin_, nu)

        nus    = 1j * nu   * sin_
        invnus = 1j / nu   * sin_
        a, b, c_, d = (cos_ * a + nus * c_,   cos_ * b + nus * d,
                       invnus * a + cos_ * c_, invnus * b + cos_ * d)

    safe = np.abs(d) > 1e-30
    Zin  = np.where(safe, b / np.where(safe, d, 1.0), 1e30 + 0j)
    refl = (Zin - Z0) / (Zin + Z0)
    result = 20.0 * np.log10(np.maximum(np.abs(refl), 1e-15))
    # Заменить любые NaN/Inf на 0 дБ (нейтральное значение)
    result = np.where(np.isfinite(result), result, 0.0)
    return result


import itertools as _itertools
_mat_arrays_counter = _itertools.count()


def build_mat_arrays(material_datas: list) -> dict:
    """
    Предвычислить γ и ν для всех материалов и всех частот.

    Возвращает словарь с ключами:
        'er', 'eim', 'mr', 'mim'   — форма (n_mat, n_freq)
        'gamma', 'nu'              — форма (n_mat, n_freq)
        '_cache_id'                — уникальный монотонный штамп для ключей
                                     кеша (_trig_cache/_fitness_cache) вместо
                                     id(mat_arrays). id() python-объекта не
                                     годится: после сборки мусора он может
                                     быть переиспользован для другого,
                                     физически иного mat_arrays — тогда кеш
                                     возвращал бы значения ЧУЖОГО материала.
                                     Актуально для коротко живущих mat_arrays
                                     (например, db_viewer.py строит новый
                                     mat_arrays на каждый выбранный график).
    """
    n_mat  = len(material_datas)
    n_freq = len(material_datas[0]['freq'])
    freqs  = material_datas[0]['freq']
    arrs   = {k: np.zeros((n_mat, n_freq)) for k in ('er', 'eim', 'mr', 'mim')}
    for i, mat in enumerate(material_datas):
        for k in arrs:
            arrs[k][i] = mat[k]

    k0       = 2.0 * np.pi * freqs * np.sqrt(mu_0 * epsilon_0)   # (n_freq,)
    eps_c    = arrs['er']  - 1j * arrs['eim']                     # (n_mat, n_freq)
    mu_c     = arrs['mr']  - 1j * arrs['mim']                     # (n_mat, n_freq)
    arrs['gamma'] = k0[None, :] * np.sqrt(mu_c * eps_c)           # (n_mat, n_freq)
    arrs['nu']    = Z0          * np.sqrt(mu_c / eps_c)            # (n_mat, n_freq)
    arrs['_cache_id'] = next(_mat_arrays_counter)
    return arrs


# ─────────────────────────────────────────────────────────────────────────────
# Кеш и квантование
# ─────────────────────────────────────────────────────────────────────────────

def quantise(hs) -> tuple:
    """Квантовать толщины с шагом H_GRID."""
    return tuple(round(h / H_GRID) * H_GRID for h in hs)


def evaluate_objectives(sequence: tuple, hs,
                        freqs: np.ndarray,
                        mat_arrays: dict,
                        rc_evaluator=None) -> tuple:
    """
    Вернуть (max_RC [дБ], total_thickness [мм]).
    Результат кешируется по квантованным ключам.

    rc_evaluator : callable(freqs, sequence, hs, mat_arrays) -> RC array
        По умолчанию — честный tmm_reflection. Может быть заменён на
        суррогатную модель (см. worker.py::_make_rc_evaluator) для ускорения
        ПОИСКА внутри NSGA-II/DE. Обратите внимание: это влияет только на
        то, какие кандидаты оптимизатор считает перспективными во время
        поиска — worker.py всегда пересчитывает max_rc честным TMM для
        итоговых конфигураций перед сохранением в БД/показом пользователю,
        независимо от того, что использовалось здесь.
    """
    if rc_evaluator is None:
        rc_evaluator = tmm_reflection
    hs_q = quantise(hs)
    key  = (mat_arrays['_cache_id'], id(rc_evaluator), sequence, hs_q)
    if key in _fitness_cache:
        return _fitness_cache[key]
    RC     = rc_evaluator(freqs, sequence, hs_q, mat_arrays)
    result = (float(np.max(RC)), float(np.sum(hs_q) * 1000.0))
    _fitness_cache[key] = result
    return result


# ─────────────────────────────────────────────────────────────────────────────
# Дифференциальная эволюция — оптимизация толщин
# ─────────────────────────────────────────────────────────────────────────────

def _scalar_obj(hs_arr, sequence, freqs, mat_arrays, w_rc, w_th, rc_threshold,
                rc_evaluator):
    hs     = quantise(np.clip(hs_arr, H_MIN, None))
    rc, th = evaluate_objectives(sequence, hs, freqs, mat_arrays, rc_evaluator)
    penalty = RC_PENALTY * max(0.0, rc - rc_threshold)
    return w_rc * rc + w_th * th + penalty


def optimise_thicknesses(sequence, freqs, mat_arrays,
                         h_min, h_max_per,
                         w_rc=1.0, w_th=0.0,
                         rc_threshold=RC_THRESHOLD,
                         rc_evaluator=None):
    """Оптимизировать толщины слоёв методом Differential Evolution.

    Параметры DE адаптируются к числу слоёв:
      • maxiter убывает с ростом n (константное число вычислений TMM)
      • popsize убывает, чтобы внутренняя популяция DE не росла квадратично
      • polish=False — L-BFGS-B на разрывной (квантованной) функции не помогает
        и добавляет сотни лишних вызовов TMM
      • tol=1e-4 вместо 1e-5 — для квантованного ландшафта нет смысла в более
        точном сходимости

    rc_threshold : float
        Порог RC [дБ], используемый только для штрафа в целевой функции
        (значение должно быть < 0; за валидацию отвечает вызывающий код).
    rc_evaluator : callable(freqs, sequence, hs, mat_arrays) -> RC array
        См. evaluate_objectives(). По умолчанию — честный TMM.
    """
    n         = len(sequence)
    bounds    = [(h_min, h_max_per)] * n
    # Адаптивные параметры: суммарное число вычислений TMM ≈ const независимо от n
    maxiter_  = max(80, 300 // n)
    popsize_  = max(5, 12 - n)
    res       = differential_evolution(
        _scalar_obj, bounds,
        args=(sequence, freqs, mat_arrays, w_rc, w_th, rc_threshold, rc_evaluator),
        maxiter=maxiter_, popsize=popsize_, tol=1e-4,
        mutation=(0.5, 1.5), recombination=0.9,
        seed=42, polish=False, workers=1, updating='deferred'
    )
    hs     = quantise(np.clip(res.x, h_min, h_max_per))
    rc, th = evaluate_objectives(sequence, hs, freqs, mat_arrays, rc_evaluator)
    return hs, rc, th


# ─────────────────────────────────────────────────────────────────────────────
# NSGA-II утилиты
# ─────────────────────────────────────────────────────────────────────────────

def non_dominated_sort(objs: list) -> list:
    """
    Векторизованная недоминируемая сортировка (NumPy).
    Возвращает список фронтов (каждый фронт — список индексов).
    """
    arr = np.array(objs, dtype=float)
    N   = len(arr)
    dom_by  = np.zeros(N, dtype=int)
    dom_set = [[] for _ in range(N)]

    for i in range(N):
        leq = np.all(arr[i] <= arr, axis=1)
        lt  = np.any(arr[i] <  arr, axis=1)
        i_dominates = leq & lt
        i_dominates[i] = False
        dom_set[i] = np.where(i_dominates)[0].tolist()

        leq_rev = np.all(arr <= arr[i], axis=1)
        lt_rev  = np.any(arr <  arr[i], axis=1)
        dom_by[i] = int(np.sum(leq_rev & lt_rev))

    fronts = [list(np.where(dom_by == 0)[0])]
    k = 0
    while fronts[k]:
        nxt = []
        for i in fronts[k]:
            for j in dom_set[i]:
                dom_by[j] -= 1
                if dom_by[j] == 0:
                    nxt.append(j)
        k += 1
        fronts.append(nxt)
    return [f for f in fronts if f]


def crowding_distance(front: list, objs: list) -> dict:
    """Вычислить crowding distance для указанного фронта."""
    dist = {i: 0.0 for i in front}
    for m in range(2):
        srt  = sorted(front, key=lambda i: objs[i][m])
        dist[srt[0]] = dist[srt[-1]] = float('inf')
        span = (objs[srt[-1]][m] - objs[srt[0]][m]) or 1e-15
        for k in range(1, len(srt) - 1):
            dist[srt[k]] += (objs[srt[k + 1]][m] - objs[srt[k - 1]][m]) / span
    return dist


# ─────────────────────────────────────────────────────────────────────────────
# NSGA-II — главный цикл
# ─────────────────────────────────────────────────────────────────────────────

def nsga2_optimise(n_layers: int, num_materials: int,
                   freqs: np.ndarray, mat_arrays: dict,
                   h_max_total: float,
                   w_rc: float = 1.0, w_th: float = 0.3,
                   pop_size: int = 50, generations: int = 60,
                   rc_threshold: float = RC_THRESHOLD,
                   rc_evaluator=None,
                   _cancel_flag=None, _progress_cb=None) -> list:
    """
    NSGA-II для поиска Парето-оптимальных последовательностей материалов.

    pop_size и generations адаптируются к n_layers, чтобы суммарное время
    оставалось примерно постоянным:
      generations = max(30, 80 - n_layers * 5)
      pop_size    = max(30, 60 - n_layers * 3)

    rc_threshold : float
        Порог RC [дБ] для штрафа в целевой функции DE (см. optimise_thicknesses).
    rc_evaluator : callable(freqs, sequence, hs, mat_arrays) -> RC array, optional
        Функция расчёта RC(f), используемая ВНУТРИ поиска (для отбора
        перспективных кандидатов). По умолчанию — честный TMM. Может быть
        заменена на суррогатную ML-модель для ускорения поиска — см.
        worker.py. Итоговый результат при этом ВСЕГДА перепроверяется
        честным TMM в worker.py перед сохранением, так что ошибки суррогата
        не попадают в БД — они лишь могут ухудшить качество/скорость поиска.

    Возвращает список кортежей (sequence, hs, rc, th).
    """
    # Адаптивные параметры — игнорируем внешние значения по умолчанию
    generations = max(30, 80 - n_layers * 5)
    pop_size    = max(30, 60 - n_layers * 3)
    h_max_per   = h_max_total / n_layers

    def rnd_seq():
        return tuple(random.randint(1, num_materials) for _ in range(n_layers))

    def eval_seq(seq):
        try:
            hs, rc, th = optimise_thicknesses(seq, freqs, mat_arrays,
                                              H_MIN, h_max_per, w_rc, w_th,
                                              rc_threshold=rc_threshold,
                                              rc_evaluator=rc_evaluator)
            # Защита от NaN/Inf в целевых значениях
            if not (np.isfinite(rc) and np.isfinite(th)):
                rc = 0.0; th = float(np.sum([H_MIN]*len(seq))*1000)
        except Exception as e:
            log.warning(f"  eval_seq ошибка {seq}: {e}")
            hs = tuple([H_MIN] * len(seq))
            rc = 0.0; th = float(np.sum([H_MIN]*len(seq))*1000)
        return hs, (rc, th)

    pop      = [rnd_seq() for _ in range(pop_size)]
    pop_hs   = [None] * pop_size
    pop_objs = [None] * pop_size
    log.info(f"  [n={n_layers}] Инициализация {pop_size} особей, "
             f"{generations} поколений (адаптивно)…")
    for i, seq in enumerate(pop):
        pop_hs[i], pop_objs[i] = eval_seq(seq)

    def make_tournament(objs_list):
        fronts = non_dominated_sort(objs_list)
        rank   = {idx: r for r, fr in enumerate(fronts) for idx in fr}
        cd     = {}
        for fr in fronts:
            cd.update(crowding_distance(fr, objs_list))
        def select(k=2):
            cands = random.sample(range(len(objs_list)), k)
            return min(cands, key=lambda i: (rank.get(i, 0), -cd.get(i, 0)))
        return select

    for gen in range(generations):
        if _cancel_flag is not None and _cancel_flag.is_set():
            break
        t0 = time.time()
        mr = 0.20 * (1 - gen / generations) + 0.05
        tournament = make_tournament(pop_objs)

        offspring = []
        for _ in range(pop_size):
            p1 = tournament(); p2 = tournament()
            s1, s2 = list(pop[p1]), list(pop[p2])
            if n_layers > 1 and random.random() < 0.85:
                cp    = random.randint(1, n_layers - 1)
                child = s1[:cp] + s2[cp:]
            else:
                child = s1[:]
            for k in range(n_layers):
                if random.random() < mr:
                    child[k] = random.randint(1, num_materials)
            offspring.append(tuple(child))

        off_hs   = [None] * pop_size
        off_objs = [None] * pop_size
        for i, seq in enumerate(offspring):
            off_hs[i], off_objs[i] = eval_seq(seq)

        comb_seqs = pop + offspring
        comb_hs   = pop_hs + off_hs
        comb_objs = pop_objs + off_objs

        fronts = non_dominated_sort(comb_objs)
        new_pop, new_hs, new_objs = [], [], []
        for front in fronts:
            if len(new_pop) + len(front) <= pop_size:
                for idx in front:
                    new_pop.append(comb_seqs[idx])
                    new_hs.append(comb_hs[idx])
                    new_objs.append(comb_objs[idx])
            else:
                need = pop_size - len(new_pop)
                cd   = crowding_distance(front, comb_objs)
                for idx in sorted(front, key=lambda i: -cd[i])[:need]:
                    new_pop.append(comb_seqs[idx])
                    new_hs.append(comb_hs[idx])
                    new_objs.append(comb_objs[idx])
                break
        pop, pop_hs, pop_objs = new_pop, new_hs, new_objs

        # Защита: если популяция опустела (крайний случай) — прерваться
        if not pop:
            log.warning(f"  [n={n_layers}] Популяция опустела на поколении {gen+1}, прерываем.")
            break

        best_rc = min(o[0] for o in pop_objs)
        best_th = min(o[1] for o in pop_objs)
        log.info(f"  [n={n_layers}] Поколение {gen+1:>3}/{generations}  "
                 f"лучший RC={best_rc:+.2f} дБ  тонкий={best_th:.1f} мм  "
                 f"кеш={len(_fitness_cache)}  ({time.time()-t0:.1f}с)")
        if _progress_cb is not None:
            _progress_cb()

    fronts     = non_dominated_sort(pop_objs)
    pareto_idx = fronts[0]
    results, seen = [], set()

    for idx in pareto_idx:
        seq = pop[idx]; hs = pop_hs[idx]; rc, th = pop_objs[idx]
        key = (seq, quantise(hs))
        if key not in seen:
            seen.add(key)
            results.append((seq, hs, rc, th))

    for seq in list({r[0] for r in results}):
        hs2, rc2, th2 = optimise_thicknesses(seq, freqs, mat_arrays,
                                             H_MIN, h_max_per,
                                             w_rc=1.0, w_th=0.0,
                                             rc_threshold=rc_threshold)
        key = (seq, quantise(hs2))
        if key not in seen:
            seen.add(key)
            results.append((seq, hs2, rc2, th2))

    log.info(f"  [n={n_layers}] Парето-решений: {len(results)}")
    return results


# ─────────────────────────────────────────────────────────────────────────────
# Постобработка
# ─────────────────────────────────────────────────────────────────────────────

def sensitivity_analysis(sequence: tuple, hs: tuple,
                         freqs: np.ndarray, mat_arrays: dict,
                         perturb: float = 0.05) -> dict:
    """Анализ чувствительности RC к отклонению толщин ±5%."""
    rc_base, _ = evaluate_objectives(sequence, hs, freqs, mat_arrays)
    deltas = []
    for i in range(len(hs)):
        hs_up   = list(hs); hs_up[i]   = max(H_MIN, hs_up[i]   * (1 + perturb))
        hs_down = list(hs); hs_down[i] = max(H_MIN, hs_down[i] * (1 - perturb))
        rc_up,  _ = evaluate_objectives(sequence, tuple(hs_up),   freqs, mat_arrays)
        rc_down,_ = evaluate_objectives(sequence, tuple(hs_down), freqs, mat_arrays)
        deltas.append(max(abs(rc_up - rc_base), abs(rc_down - rc_base)))
    return {
        'per_layer_dRC':    deltas,
        'max_sensitivity':  float(max(deltas)),
        'mean_sensitivity': float(np.mean(deltas)),
        'robust_score':     sum(1 for d in deltas if d < 2.0),
    }


def compute_bandwidth(RC: np.ndarray, freqs: np.ndarray,
                      threshold: float = -10.0) -> float:
    """Вернуть ширину полосы (ГГц), в которой RC <= threshold."""
    mask = RC <= threshold
    if not np.any(mask):
        return 0.0
    idx = np.where(mask)[0]
    return float(freqs[idx[-1]] - freqs[idx[0]]) / 1e9


def rank_configurations(configs: list, rc_threshold: float = RC_THRESHOLD) -> list:
    """
    Разделить конфигурации на валидные (RC <= rc_threshold) и невалидные,
    вычислить нормированную оценку, отсортировать.
    """
    if not configs:
        return configs
    valid   = [c for c in configs if c['max_rc'] <= rc_threshold]
    invalid = [c for c in configs if c['max_rc'] >  rc_threshold]
    log.info(f"  Валидных конфигураций (RC <= {rc_threshold} дБ): "
             f"{len(valid)} / {len(configs)}")
    if invalid:
        log.info(f"  Отброшено (RC > {rc_threshold} дБ): {len(invalid)}")
    if valid:
        rc_vals = [c['max_rc']   for c in valid]
        th_vals = [c['total_mm'] for c in valid]
        rc_span = (max(rc_vals) - min(rc_vals)) or 1.0
        th_span = (max(th_vals) - min(th_vals)) or 1.0
        rc_min  = min(rc_vals)
        th_min  = min(th_vals)
        for cfg in valid:
            cfg['score'] = (0.7 * (cfg['max_rc']  - rc_min) / rc_span +
                            0.3 * (cfg['total_mm'] - th_min) / th_span)
        valid.sort(key=lambda x: x['score'])
        for rank, cfg in enumerate(valid, 1):
            cfg['rank'] = rank
    for cfg in invalid:
        cfg['score'] = float('inf')
        cfg['rank']  = None
    return valid + invalid


# ─────────────────────────────────────────────────────────────────────────────
# Ввод/вывод
# ─────────────────────────────────────────────────────────────────────────────

def load_material_data(folder_path: str):
    """
    Загрузить все .xlsx-файлы из папки как данные материалов.

    Возвращает (freqs, material_datas, file_names).
    """
    files = sorted([f for f in os.listdir(folder_path) if f.endswith('.xlsx')])
    if not files:
        raise FileNotFoundError(f"Нет .xlsx файлов в '{folder_path}'")
    mats = []
    for fname in files:
        df = pd.read_excel(os.path.join(folder_path, fname))
        mats.append({'freq': df['f'].values,
                     'er':   df['er'].values, 'eim': df['eim'].values,
                     'mr':   df['mr'].values, 'mim': df['mim'].values})
    freqs = mats[0]['freq']
    for m in mats[1:]:
        if not np.array_equal(freqs, m['freq']):
            raise ValueError("Сетки частот не совпадают между файлами материалов!")
    log.info(f"Загружено {len(files)} материалов, {len(freqs)} частотных точек "
             f"({freqs[0]/1e9:.2f}–{freqs[-1]/1e9:.2f} ГГц)")
    return freqs, mats, files


def write_csv(output_file: str, configs: list, N_max_layers: int) -> None:
    """Сохранить все конфигурации в CSV с разделителями ; и BOM для Excel."""
    header = ['Ранг', 'Оценка', 'Число слоёв']
    for i in range(N_max_layers):
        header += [f'Слой {i+1} материал', f'Слой {i+1} h (мм)']
    header += ['Max RC (дБ)', 'Суммарная h (мм)',
               'Полоса <-10дБ (ГГц)', 'Max чувств. dRC (дБ)', 'Робастных слоёв']
    
    with open(output_file, 'w', newline='', encoding='utf-8-sig') as f:
        w = csv.writer(f, delimiter=';')  # ← точка с запятой для Excel
        w.writerow(header)
        for cfg in configs:
            seq  = cfg['sequence']; hs = cfg['hs']; n = len(seq)
            sens = cfg.get('sensitivity', {})
            
            rank = cfg.get('rank', '')
            score = round(cfg.get('score', 0), 4) if cfg.get('rank') is not None else ''
            
            row  = [rank, score, n]
            for i in range(N_max_layers):
                if i < n:
                    row += [seq[i], round(hs[i] * 1000, 1)]
                else:
                    row += ['', '']
                    
            row += [
                round(cfg['max_rc'], 2),
                round(cfg['total_mm'], 1),
                round(cfg.get('bandwidth_ghz', 0), 3),
                round(sens.get('max_sensitivity', 0), 3) if sens and sens.get('max_sensitivity') else '',
                sens.get('robust_score', '') if sens else ''
            ]
            w.writerow(row)
    log.info(f"CSV сохранён: {output_file}  ({len(configs)} строк)")