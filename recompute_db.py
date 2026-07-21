"""
recompute_db.py — разовая миграция: пересчитать max_rc_db / bandwidth_ghz /
is_valid / rank / score для всех уже сохранённых покрытий.

Зачем: до фикса в worker.py в БД писалось значение 'rc' из оптимизатора
(NSGA-II/DE), а не честный np.max(RC(f)) от итоговых квантованных толщин.
У части записей эти значения расходятся (см. обсуждение), поэтому вся
накопленная история требует ревизии.

Скрипт для каждого покрытия:
  1. восстанавливает материалы/толщины (как db_viewer.build_plot_context),
  2. честно пересчитывает RC(f) через physics.tmm_reflection,
  3. берёт max_rc = np.max(RC), bandwidth = compute_bandwidth(...),
  4. пере-ранжирует покрытия ЗАНОВО в пределах каждого прогона
     (physics.rank_configurations — та же логика, что и в живом прогоне),
  5. записывает обновлённые значения обратно в БД.

Ничего не удаляет — только обновляет числа. Перед массовым обновлением
делает резервную копию файла БД (.bak).

Запуск:
    python recompute_db.py                      # database.py::DEFAULT_DB_PATH
    python recompute_db.py --db path/to/rc.db
    python recompute_db.py --dry-run             # только показать расхождения
"""

from __future__ import annotations

import os
import shutil
import argparse
import logging

import numpy as np

from database import Database, DEFAULT_DB_PATH
from db_viewer import build_plot_context
from physics import tmm_reflection, compute_bandwidth, rank_configurations

log = logging.getLogger("rc_opt")


def recompute_all(db_path: str = DEFAULT_DB_PATH, dry_run: bool = False) -> dict:
    if not dry_run and os.path.exists(db_path):
        backup_path = db_path + ".bak"
        shutil.copy2(db_path, backup_path)
        log.info(f"Резервная копия БД: {backup_path}")

    db = Database(db_path)
    runs = db.list_runs()

    n_total = 0
    n_changed = 0
    n_skipped = 0
    max_diff_seen = 0.0

    for run in runs:
        run_id = run["run_id"]
        run_full = db.get_run(run_id)
        rc_threshold = run_full["rc_threshold_db"]

        coatings = db.list_coatings(run_id=run_id)
        recomputed = []

        for c in coatings:
            n_total += 1
            full = db.get_coating(c["coating_id"])
            try:
                cfg, freqs, mat_arrays, names = build_plot_context(db, full)
            except ValueError as e:
                log.warning(f"  Покрытие #{c['coating_id']}: пропущено "
                           f"(не удалось восстановить материалы: {e})")
                n_skipped += 1
                continue

            RC = tmm_reflection(freqs, cfg["sequence"], cfg["hs"], mat_arrays)
            new_max_rc = float(np.max(RC))
            new_bw = compute_bandwidth(RC, freqs, threshold=rc_threshold)

            old_max_rc = full["max_rc_db"]
            diff = abs(new_max_rc - old_max_rc)
            max_diff_seen = max(max_diff_seen, diff)
            if diff > 0.05:
                n_changed += 1
                log.info(f"  Покрытие #{c['coating_id']} (прогон #{run_id}): "
                        f"max_rc {old_max_rc:+.2f} → {new_max_rc:+.2f} дБ "
                        f"(Δ={diff:.2f}), полоса {full['bandwidth_ghz']:.2f} → "
                        f"{new_bw:.2f} ГГц")

            recomputed.append({
                "coating_id": c["coating_id"],
                "max_rc": new_max_rc,
                "total_mm": full["total_thickness_mm"],
                "bandwidth_ghz": new_bw,
            })

        # Пере-ранжировать заново в пределах прогона той же логикой, что и worker.py
        ranked = rank_configurations(recomputed, rc_threshold=rc_threshold)

        if not dry_run:
            for cfg in ranked:
                is_valid = 1 if cfg["max_rc"] <= rc_threshold else 0
                db.update_coating_metrics(
                    cfg["coating_id"], max_rc_db=cfg["max_rc"],
                    bandwidth_ghz=cfg["bandwidth_ghz"], is_valid=is_valid,
                    rank=cfg.get("rank"), score=cfg.get("score"),
                )

    db.close()
    summary = {
        "n_total": n_total, "n_changed": n_changed, "n_skipped": n_skipped,
        "max_diff_db": round(max_diff_seen, 3), "dry_run": dry_run,
    }
    log.info(f"Готово: проверено {n_total}, изменено {n_changed}, "
            f"пропущено {n_skipped}, макс. расхождение {summary['max_diff_db']} дБ"
            + (" (dry-run, БД не изменена)" if dry_run else ""))
    return summary


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s  %(levelname)-8s  %(message)s")
    p = argparse.ArgumentParser(description="Пересчитать max_rc/bandwidth/rank в БД")
    p.add_argument("--db", default=DEFAULT_DB_PATH)
    p.add_argument("--dry-run", action="store_true",
                  help="Только показать расхождения, не менять БД")
    args = p.parse_args()
    recompute_all(db_path=args.db, dry_run=args.dry_run)
