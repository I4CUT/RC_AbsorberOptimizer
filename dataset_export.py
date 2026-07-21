"""
dataset_export.py — экспорт накопленной БД в обучающую выборку для ML

Берёт всё, что накопилось в rc_absorber.db (materials + coatings из всех
прогонов), и превращает в готовые для обучения тензоры/таблицы.

Главная сложность: разные прогоны могут использовать материалы с разными
частотными сетками (разный диапазон, разное число точек). Чтобы данные
из разных прогонов были сопоставимы для сети, спектры материалов и кривые
RC(f) ресемплируются на единую каноническую сетку частот.

Выходные файлы (в output_dir):
  dataset_scalar.csv   — по одной строке на покрытие: material_id и толщина
                          каждого слоя (с паддингом), скалярные метрики
                          (max_rc, толщина, полоса), метаданные прогона.
                          Годится для быстрого старта: sklearn/pandas,
                          без матриц толщиной в частоты.
  dataset_tensors.npz  — то же самое + сами ресемплированные спектры
                          материалов и кривые RC(f) на канонической сетке.
                          Нужен для sequence/CNN-архитектур (см. roadmap,
                          этап 2 — forward-суррогат).
  dataset_meta.json    — параметры экспорта (сетка, число сэмплов, отсечки)
                          для воспроизводимости и логов.

Запуск как скрипт:
    python dataset_export.py --output ./ml_dataset --grid-points 201

Запуск как модуль:
    from dataset_export import export_dataset
    summary = export_dataset(output_dir="./ml_dataset")
"""

from __future__ import annotations

import os
import json
import argparse
import logging

import numpy as np
import pandas as pd

from database import Database, DEFAULT_DB_PATH

log = logging.getLogger("rc_opt")


# ─────────────────────────────────────────────────────────────────────────────
# Ресемплинг на каноническую сетку
# ─────────────────────────────────────────────────────────────────────────────

def _resample(freq_src, values_src, freq_canon) -> np.ndarray:
    """Линейная интерполяция значений с сетки измерения на каноническую сетку."""
    freq_src = np.asarray(freq_src, dtype=float)
    values_src = np.asarray(values_src, dtype=float)
    order = np.argsort(freq_src)
    return np.interp(freq_canon, freq_src[order], values_src[order])


def _build_canonical_grid(materials: list, n_points: int,
                          f_min: float | None, f_max: float | None) -> np.ndarray:
    """
    Построить каноническую сетку частот.

    Если f_min/f_max не заданы явно — берётся ПЕРЕСЕЧЕНИЕ диапазонов всех
    материалов в БД (max из f_min, min из f_max), чтобы ресемплинг никогда
    не экстраполировал за пределы измеренных данных.
    """
    if f_min is None:
        f_min = max(m["f_min_hz"] for m in materials)
    if f_max is None:
        f_max = min(m["f_max_hz"] for m in materials)
    if f_max <= f_min:
        raise ValueError(
            f"Диапазоны частот материалов в БД не пересекаются "
            f"(получено f_min={f_min/1e9:.2f} ГГц >= f_max={f_max/1e9:.2f} ГГц). "
            f"Укажите --f-min/--f-max вручную, или экспортируйте подмножество "
            f"материалов с совпадающими диапазонами.")
    return np.linspace(f_min, f_max, n_points)


# ─────────────────────────────────────────────────────────────────────────────
# Скалярные дескрипторы материала (для быстрого baseline без полных кривых)
# ─────────────────────────────────────────────────────────────────────────────

def _material_scalar_features(er, eim, mr, mim) -> dict:
    er, eim, mr, mim = (np.asarray(x, dtype=float) for x in (er, eim, mr, mim))
    with np.errstate(divide="ignore", invalid="ignore"):
        loss_tangent = np.where(er != 0, eim / er, 0.0)
    return {
        "er_mean": float(np.mean(er)), "er_std": float(np.std(er)),
        "eim_mean": float(np.mean(eim)), "eim_std": float(np.std(eim)),
        "mr_mean": float(np.mean(mr)), "mr_std": float(np.std(mr)),
        "mim_mean": float(np.mean(mim)), "mim_std": float(np.std(mim)),
        "loss_tangent_mean": float(np.mean(loss_tangent)),
        # огрублённый наклон дисперсии er по диапазону (начало → конец)
        "er_slope": float((er[-1] - er[0]) / max(len(er) - 1, 1)),
    }


_SCALAR_FEATURE_NAMES = [
    "er_mean", "er_std", "eim_mean", "eim_std",
    "mr_mean", "mr_std", "mim_mean", "mim_std",
    "loss_tangent_mean", "er_slope",
]


# ─────────────────────────────────────────────────────────────────────────────
# Основная функция экспорта
# ─────────────────────────────────────────────────────────────────────────────

def export_dataset(db_path: str = DEFAULT_DB_PATH,
                   output_dir: str = "./ml_dataset",
                   grid_points: int = 201,
                   f_min: float | None = None,
                   f_max: float | None = None,
                   only_valid: bool = False,
                   include_curves: bool = True,
                   max_layers: int | None = None,
                   run_ids: list[int] | None = None,
                   progress_cb=None) -> dict:
    """
    Экспортировать накопленные в БД покрытия в обучающую выборку.

    run_ids          — если задан, экспортировать только покрытия из ЭТИХ
                       прогонов (а не всей БД). Каноническая сетка частот
                       при этом строится только по материалам, реально
                       использованным в выбранных прогонах — так добавление
                       в БД материала с совсем другим диапазоном (в другом,
                       не выбранном прогоне) не сузит сетку.
    only_valid      — экспортировать только покрытия с RC <= порога их прогона.
                       По умолчанию False: невалидные конфигурации — тоже
                       ценные данные (сеть должна учиться отличать плохие
                       решения от хороших, а не видеть только хорошие).
                       На практике, если в worker.py включено «сохранять
                       только валидные», это в любом случае всё, что есть в БД.
    include_curves  — сохранять ли ресемплированные кривые RC(f) и спектры
                       материалов в dataset_tensors.npz. Если материалов много
                       и точек в сетке много, файл может быть заметного
                       размера — при желании можно выключить и работать
                       только со скалярными признаками.
    max_layers       — на сколько слоёв паддить тензоры. По умолчанию —
                       максимум по всем экспортируемым покрытиям.
    progress_cb       — необязательный callback(done: int, total: int),
                       вызывается по ходу обработки покрытий (для GUI).

    Возвращает словарь со сводкой экспорта (тоже пишется в dataset_meta.json).
    """
    os.makedirs(output_dir, exist_ok=True)
    db = Database(db_path)

    if run_ids:
        material_id_set = set()
        for rid in run_ids:
            run = db.get_run(rid)
            if run is None:
                db.close()
                raise RuntimeError(f"Прогон run_id={rid} не найден в БД.")
            material_id_set.update(run["material_ids"])
        all_materials = db.list_materials()
        materials = [m for m in all_materials if m["material_id"] in material_id_set]
        coatings = []
        for rid in run_ids:
            coatings.extend(db.list_coatings(run_id=rid, only_valid=only_valid, limit=None))
    else:
        materials = db.list_materials()
        coatings = db.list_coatings(only_valid=only_valid, limit=None)

    if not materials:
        db.close()
        raise RuntimeError("Нет материалов, подходящих под выбор "
                           "(проверьте выбранные прогоны).")
    if not coatings:
        db.close()
        raise RuntimeError("Нет покрытий, подходящих под фильтр "
                           f"(run_ids={run_ids}, only_valid={only_valid}).")

    canon_freq = _build_canonical_grid(materials, grid_points, f_min, f_max)
    log.info(f"Каноническая сетка: {canon_freq[0]/1e9:.2f}–{canon_freq[-1]/1e9:.2f} ГГц, "
            f"{grid_points} точек")

    n_layers_cap = max_layers or max(c["n_layers"] for c in coatings)

    # ── Кэш материалов: полные данные + скалярные признаки + ресемпл. спектр ──
    mat_cache: dict[int, dict] = {}
    skipped_out_of_range = 0

    def _get_material_cached(mid: int) -> dict | None:
        if mid in mat_cache:
            return mat_cache[mid]
        m = db.get_material(mid)
        if m is None:
            return None
        freq = np.asarray(m["freq_hz"], dtype=float)
        if freq.min() > canon_freq.min() or freq.max() < canon_freq.max():
            # материал не покрывает всю каноническую сетку — пропускаем при
            # использовании (такое бывает, если диапазоны материалов сильно
            # различаются; --f-min/--f-max можно сузить, чтобы избежать этого)
            mat_cache[mid] = None
            return None
        entry = {
            "name": m["name"],
            "scalar": _material_scalar_features(m["er"], m["eim"], m["mr"], m["mim"]),
            "er_r":  _resample(freq, m["er"],  canon_freq),
            "eim_r": _resample(freq, m["eim"], canon_freq),
            "mr_r":  _resample(freq, m["mr"],  canon_freq),
            "mim_r": _resample(freq, m["mim"], canon_freq),
        }
        mat_cache[mid] = entry
        return entry

    # ── Проход по покрытиям ────────────────────────────────────────────────
    rows = []
    tens_mat_scalar = []   # (N, n_layers_cap, n_scalar_features)
    tens_mat_curve  = []   # (N, n_layers_cap, grid_points, 4)  [er,eim,mr,mim]
    tens_thickness  = []   # (N, n_layers_cap)
    tens_mask       = []   # (N, n_layers_cap)  1 = слой существует
    tens_material_id = []  # (N, n_layers_cap)  для трассируемости (0 = пусто)
    tens_rc_curve   = []   # (N, grid_points)  или NaN, если кривой не было
    y_max_rc, y_thickness, y_bandwidth, y_valid = [], [], [], []
    meta_run_id, meta_coating_id, meta_rank, meta_score = [], [], [], []

    n_scalar = len(_SCALAR_FEATURE_NAMES)
    n_coatings_total = len(coatings)

    for i, c in enumerate(coatings):
        if progress_cb is not None and i % 20 == 0:
            progress_cb(i, n_coatings_total)
        seq_ids = c["sequence_material_ids"]
        if len(seq_ids) > n_layers_cap:
            continue  # покрытие толще, чем текущий паддинг — пропускаем

        entries = [_get_material_cached(mid) for mid in seq_ids]
        if any(e is None for e in entries):
            skipped_out_of_range += 1
            continue

        full = db.get_coating(c["coating_id"])

        mat_scalar_row = np.zeros((n_layers_cap, n_scalar), dtype=np.float32)
        mat_curve_row  = np.zeros((n_layers_cap, grid_points, 4), dtype=np.float32) \
                        if include_curves else None
        thickness_row  = np.zeros(n_layers_cap, dtype=np.float32)
        mask_row       = np.zeros(n_layers_cap, dtype=np.float32)
        matid_row      = np.zeros(n_layers_cap, dtype=np.int32)

        for i, (mid, e, h_mm) in enumerate(zip(seq_ids, entries, full["thicknesses_mm"])):
            mat_scalar_row[i] = [e["scalar"][k] for k in _SCALAR_FEATURE_NAMES]
            if include_curves:
                mat_curve_row[i, :, 0] = e["er_r"]
                mat_curve_row[i, :, 1] = e["eim_r"]
                mat_curve_row[i, :, 2] = e["mr_r"]
                mat_curve_row[i, :, 3] = e["mim_r"]
            thickness_row[i] = h_mm
            mask_row[i] = 1.0
            matid_row[i] = mid

        if include_curves:
            if full.get("rc_curve_db") and full.get("rc_curve_freq_hz"):
                rc_r = _resample(full["rc_curve_freq_hz"], full["rc_curve_db"], canon_freq)
            else:
                rc_r = np.full(grid_points, np.nan, dtype=np.float32)
            tens_rc_curve.append(rc_r)

        tens_mat_scalar.append(mat_scalar_row)
        if include_curves:
            tens_mat_curve.append(mat_curve_row)
        tens_thickness.append(thickness_row)
        tens_mask.append(mask_row)
        tens_material_id.append(matid_row)

        y_max_rc.append(c["max_rc_db"])
        y_thickness.append(c["total_thickness_mm"])
        y_bandwidth.append(c.get("bandwidth_ghz") if c.get("bandwidth_ghz") is not None else np.nan)
        y_valid.append(c["is_valid"])
        meta_run_id.append(c["run_id"])
        meta_coating_id.append(c["coating_id"])
        meta_rank.append(c["rank"] if c["rank"] is not None else -1)
        meta_score.append(full.get("score") if full.get("score") is not None else np.nan)

        row = {
            "coating_id": c["coating_id"], "run_id": c["run_id"],
            "n_layers": c["n_layers"], "max_rc_db": c["max_rc_db"],
            "total_thickness_mm": c["total_thickness_mm"],
            "bandwidth_ghz": c.get("bandwidth_ghz"),
            "is_valid": c["is_valid"], "rank": c["rank"],
        }
        for i in range(n_layers_cap):
            row[f"layer{i+1}_material_id"] = int(matid_row[i]) if mask_row[i] else None
            row[f"layer{i+1}_thickness_mm"] = float(thickness_row[i]) if mask_row[i] else None
        rows.append(row)

    if progress_cb is not None:
        progress_cb(n_coatings_total, n_coatings_total)

    if skipped_out_of_range:
        log.warning(f"Пропущено {skipped_out_of_range} покрытий: материалы не "
                    f"покрывают выбранную каноническую сетку частот.")

    if not rows:
        db.close()
        raise RuntimeError("После фильтрации не осталось ни одного покрытия для "
                           "экспорта — проверьте каноническую сетку частот "
                           "(--f-min/--f-max) и параметр only_valid.")

    # ── Скалярный CSV ──────────────────────────────────────────────────────
    df = pd.DataFrame(rows)
    csv_path = os.path.join(output_dir, "dataset_scalar.csv")
    df.to_csv(csv_path, index=False, encoding="utf-8-sig")

    # ── Тензоры NPZ ─────────────────────────────────────────────────────────
    npz_path = os.path.join(output_dir, "dataset_tensors.npz")
    npz_data = dict(
        canon_freq_hz=canon_freq.astype(np.float64),
        material_scalar=np.stack(tens_mat_scalar),
        thickness_mm=np.stack(tens_thickness),
        layer_mask=np.stack(tens_mask),
        material_id=np.stack(tens_material_id),
        max_rc_db=np.array(y_max_rc, dtype=np.float32),
        total_thickness_mm=np.array(y_thickness, dtype=np.float32),
        bandwidth_ghz=np.array(y_bandwidth, dtype=np.float32),
        is_valid=np.array(y_valid, dtype=np.int8),
        run_id=np.array(meta_run_id, dtype=np.int32),
        coating_id=np.array(meta_coating_id, dtype=np.int32),
        rank=np.array(meta_rank, dtype=np.int32),
        score=np.array(meta_score, dtype=np.float32),
        scalar_feature_names=np.array(_SCALAR_FEATURE_NAMES),
    )
    if include_curves:
        npz_data["material_curve"] = np.stack(tens_mat_curve)
        npz_data["rc_curve_db"] = np.stack(tens_rc_curve).astype(np.float32)

    np.savez_compressed(npz_path, **npz_data)

    # ── Метаданные экспорта ──────────────────────────────────────────────────
    summary = {
        "n_samples": len(rows),
        "n_layers_cap": n_layers_cap,
        "n_materials_used": len({mid for row in tens_material_id for mid in row if mid > 0}),
        "n_skipped_out_of_range": skipped_out_of_range,
        "grid_points": grid_points,
        "f_min_hz": float(canon_freq[0]), "f_max_hz": float(canon_freq[-1]),
        "only_valid": only_valid,
        "include_curves": include_curves,
        "run_ids": run_ids,
        "n_valid": int(sum(y_valid)),
        "csv_path": csv_path, "npz_path": npz_path,
    }
    with open(os.path.join(output_dir, "dataset_meta.json"), "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    db.close()
    log.info(f"Экспортировано {summary['n_samples']} покрытий "
            f"({summary['n_valid']} валидных) → {output_dir}")
    return summary


def preview_selection(db_path: str, run_ids: list[int] | None,
                      only_valid: bool = False) -> dict:
    """
    Быстрое превью без полного экспорта — сколько наберётся сэмплов и
    материалов, и какой получится диапазон канонической сетки частот.
    Используется GUI, чтобы показать это сразу при изменении выбора прогонов,
    не дожидаясь полного экспорта.
    """
    db = Database(db_path)
    try:
        if run_ids:
            material_id_set = set()
            n_coatings = 0
            for rid in run_ids:
                run = db.get_run(rid)
                if run is None:
                    continue
                material_id_set.update(run["material_ids"])
                n_coatings += len(db.list_coatings(run_id=rid, only_valid=only_valid,
                                                    limit=None))
            all_materials = db.list_materials()
            materials = [m for m in all_materials if m["material_id"] in material_id_set]
        else:
            materials = db.list_materials()
            n_coatings = len(db.list_coatings(only_valid=only_valid, limit=None))

        if not materials:
            return {"ok": False, "message": "Нет материалов в выбранных прогонах."}

        f_min = max(m["f_min_hz"] for m in materials)
        f_max = min(m["f_max_hz"] for m in materials)
        if f_max <= f_min:
            return {"ok": False,
                    "message": "Диапазоны частот выбранных материалов не "
                              "пересекаются — сузьте выбор прогонов."}
        return {
            "ok": True, "n_materials": len(materials), "n_coatings": n_coatings,
            "f_min_ghz": f_min / 1e9, "f_max_ghz": f_max / 1e9,
        }
    finally:
        db.close()


# ─────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────

def _main():
    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s  %(levelname)-8s  %(message)s")
    p = argparse.ArgumentParser(description="Экспорт БД в обучающую выборку для ML")
    p.add_argument("--db", default=DEFAULT_DB_PATH, help="Путь к rc_absorber.db")
    p.add_argument("--output", default="./ml_dataset", help="Папка для выходных файлов")
    p.add_argument("--grid-points", type=int, default=201,
                  help="Число точек канонической частотной сетки")
    p.add_argument("--f-min", type=float, default=None, help="Мин. частота, ГГц")
    p.add_argument("--f-max", type=float, default=None, help="Макс. частота, ГГц")
    p.add_argument("--only-valid", action="store_true",
                  help="Экспортировать только валидные покрытия (RC <= порога)")
    p.add_argument("--no-curves", action="store_true",
                  help="Не сохранять полные кривые (только скалярные признаки)")
    p.add_argument("--max-layers", type=int, default=None,
                  help="Паддинг по числу слоёв (по умолчанию — максимум в данных)")
    p.add_argument("--runs", default=None,
                  help="Список run_id через запятую (по умолчанию — все прогоны)")
    args = p.parse_args()

    run_ids = [int(x) for x in args.runs.split(",")] if args.runs else None

    summary = export_dataset(
        db_path=args.db, output_dir=args.output, grid_points=args.grid_points,
        f_min=(args.f_min * 1e9 if args.f_min is not None else None),
        f_max=(args.f_max * 1e9 if args.f_max is not None else None),
        only_valid=args.only_valid, include_curves=not args.no_curves,
        max_layers=args.max_layers, run_ids=run_ids,
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    _main()
