"""
database.py — локальная база данных истории вычислений (SQLite, только stdlib)

Три уровня данных:
  1. materials           — материалы (частотные спектры er/eim/mr/mim), с дедупликацией
  2. optimisation_runs    — метаданные каждого прогона оптимизатора
  3. coatings             — готовые покрытия (Парето-кандидаты), привязанные к прогону
                            и к конкретным материалам (по material_id)

Дополнительно:
  4. design_targets       — заготовка на будущее (этап "inverse design"): лог запросов
                            вида "найди покрытие под такую-то полосу" и итоговый результат.
                            Пока не используется активно, но таблица уже создаётся, чтобы
                            не делать миграцию схемы позже.

Дизайн-решения (сознательно простые для этапа 0):
  • Числовые массивы (спектры, кривая RC(f)) хранятся как JSON-текст, а не BLOB.
    Для объёмов в тысячи записей по несколько сотен точек это не создаёт проблем
    с размером БД, но зато данные читаемы и легко экспортируются без numpy.
  • Материалы дедуплицируются по хэшу содержимого (freq+er+eim+mr+mim), а не по
    имени файла — если один и тот же материал измерили и назвали по-другому,
    он не расплодится в БД. Имя, под которым материал впервые встретился,
    сохраняется как основное; остальные имена копятся в поле aliases.
  • Покрытия дедуплицируются в пределах одного прогона по хэшу
    (материалы + квантованные толщины) — чтобы одна и та же конфигурация,
    случайно попавшая и в топ-3, и в "особые" (min RC / min толщина) списки
    plotting'а, не создавала две строки.
  • Единица хранения толщины в БД — миллиметры (как и в остальном проекте),
    единица частоты — герцы (как в physics.py).

Использование:
    from database import Database

    db = Database()  # по умолчанию rc_absorber.db рядом со скриптом
    material_ids = db.register_materials(material_datas, files)
    run_id = db.create_run(params, material_ids)
    coating_ids = db.add_coatings(run_id, configs, material_ids, freqs,
                                   rc_threshold=used_rc_threshold)
    db.close()
"""

from __future__ import annotations

import os
import json
import sqlite3
import hashlib
import logging
from datetime import datetime, timezone
from contextlib import closing

import numpy as np

log = logging.getLogger("rc_opt")

import sys

def _default_app_dir() -> str:
    """
    Папка, рядом с которой по умолчанию хранится файл БД.

    При обычном запуске (`python main.py`) — папка со скриптами.
    При сборке в исполняемый файл (PyInstaller --onefile/--onedir) —
    папка, где лежит сам .exe, а НЕ временная папка распаковки
    (sys._MEIPASS): она удаляется при каждом закрытии программы, и без
    этой проверки вся накопленная история терялась бы при каждом перезапуске.
    """
    if getattr(sys, "frozen", False):
        return os.path.dirname(sys.executable)
    return os.path.dirname(os.path.abspath(__file__))


DEFAULT_DB_PATH = os.path.join(_default_app_dir(), "rc_absorber.db")

_SCHEMA = """
PRAGMA foreign_keys = ON;

CREATE TABLE IF NOT EXISTS materials (
    material_id     INTEGER PRIMARY KEY AUTOINCREMENT,
    name            TEXT NOT NULL,
    aliases         TEXT NOT NULL DEFAULT '[]',   -- JSON-список альтернативных имён
    content_hash    TEXT NOT NULL UNIQUE,          -- sha256(freq+er+eim+mr+mim)
    freq_hz         TEXT NOT NULL,                 -- JSON-массив, форма (F,)
    er              TEXT NOT NULL,
    eim             TEXT NOT NULL,
    mr              TEXT NOT NULL,
    mim             TEXT NOT NULL,
    n_points        INTEGER NOT NULL,
    f_min_hz        REAL NOT NULL,
    f_max_hz        REAL NOT NULL,
    added_at        TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS optimisation_runs (
    run_id             INTEGER PRIMARY KEY AUTOINCREMENT,
    timestamp          TEXT NOT NULL,
    mode               TEXT NOT NULL,
    n_max_layers       INTEGER NOT NULL,
    rc_threshold_db    REAL NOT NULL,
    lambda_multiplier  REAL NOT NULL,
    w_rc               REAL,
    w_th               REAL,
    folder_path        TEXT,
    output_dir         TEXT,
    material_ids       TEXT NOT NULL,   -- JSON-список material_id, в порядке индексов 1..N прогона
    comment            TEXT DEFAULT ''
);

CREATE TABLE IF NOT EXISTS coatings (
    coating_id         INTEGER PRIMARY KEY AUTOINCREMENT,
    run_id             INTEGER NOT NULL REFERENCES optimisation_runs(run_id) ON DELETE CASCADE,
    config_hash        TEXT NOT NULL,
    n_layers           INTEGER NOT NULL,
    sequence_material_ids TEXT NOT NULL,  -- JSON-список material_id, воздух → металл
    thicknesses_mm     TEXT NOT NULL,     -- JSON-список float, мм
    max_rc_db          REAL NOT NULL,
    total_thickness_mm REAL NOT NULL,
    bandwidth_ghz      REAL,
    rank               INTEGER,
    score              REAL,
    sensitivity_json    TEXT,             -- JSON: per_layer_dRC, max/mean_sensitivity, robust_score
    rc_curve_freq_hz   TEXT,              -- JSON-массив частот, на которых считалась кривая
    rc_curve_db        TEXT,              -- JSON-массив RC(f), дБ (может быть NULL, если не сохраняли)
    is_valid           INTEGER NOT NULL,  -- 1, если max_rc_db <= rc_threshold_db прогона
    created_at         TEXT NOT NULL,
    UNIQUE(run_id, config_hash)
);

CREATE INDEX IF NOT EXISTS idx_coatings_run       ON coatings(run_id);
CREATE INDEX IF NOT EXISTS idx_coatings_valid      ON coatings(is_valid);
CREATE INDEX IF NOT EXISTS idx_coatings_max_rc     ON coatings(max_rc_db);
CREATE INDEX IF NOT EXISTS idx_coatings_thickness  ON coatings(total_thickness_mm);

-- Задел на этап inverse-design (пока не используется активно)
CREATE TABLE IF NOT EXISTS design_targets (
    target_id          INTEGER PRIMARY KEY AUTOINCREMENT,
    created_at         TEXT NOT NULL,
    f_lo_hz            REAL NOT NULL,
    f_hi_hz            REAL NOT NULL,
    rc_target_db       REAL NOT NULL,
    max_thickness_mm   REAL,
    allowed_material_ids TEXT,           -- JSON-список material_id, NULL = любые
    resulting_coating_id INTEGER REFERENCES coatings(coating_id),
    comment            TEXT DEFAULT ''
);
"""


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def _round_for_hash(arr, ndigits: int = 6) -> list:
    """Округлить массив перед хэшированием — защита от различий float в последнем разряде."""
    return [round(float(x), ndigits) for x in arr]


def _hash_material(freq, er, eim, mr, mim) -> str:
    payload = json.dumps({
        "freq": _round_for_hash(freq, 2),   # частоты — целые Гц, округление грубее
        "er":   _round_for_hash(er),
        "eim":  _round_for_hash(eim),
        "mr":   _round_for_hash(mr),
        "mim":  _round_for_hash(mim),
    }, sort_keys=True)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _hash_coating(sequence_material_ids, thicknesses_mm, ndigits: int = 2) -> str:
    payload = json.dumps({
        "seq": list(sequence_material_ids),
        "hs":  [round(float(h), ndigits) for h in thicknesses_mm],
    }, sort_keys=True)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


class Database:
    """Обёртка над SQLite для хранения материалов, прогонов и покрытий."""

    def __init__(self, db_path: str = DEFAULT_DB_PATH):
        self.db_path = db_path
        self._conn = sqlite3.connect(db_path, check_same_thread=False)
        self._conn.row_factory = sqlite3.Row
        self._conn.execute("PRAGMA journal_mode = WAL;")
        self._conn.executescript(_SCHEMA)
        self._conn.commit()
        log.info(f"БД инициализирована: {db_path}")

    def close(self) -> None:
        self._conn.close()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        self.close()

    # ─────────────────────────────────────────────────────────────────────
    # Материалы
    # ─────────────────────────────────────────────────────────────────────

    def register_material(self, name: str, freq, er, eim, mr, mim) -> int:
        """
        Зарегистрировать один материал. Если материал с таким же содержимым
        спектра уже есть в БД — вернуть существующий material_id (и добавить
        имя в aliases, если оно новое). Иначе — создать новую запись.
        """
        freq = list(np.asarray(freq, dtype=float))
        er, eim, mr, mim = (list(np.asarray(x, dtype=float))
                            for x in (er, eim, mr, mim))
        h = _hash_material(freq, er, eim, mr, mim)

        with closing(self._conn.cursor()) as cur:
            cur.execute("SELECT material_id, name, aliases FROM materials "
                       "WHERE content_hash = ?", (h,))
            row = cur.fetchone()
            if row is not None:
                material_id = row["material_id"]
                aliases = json.loads(row["aliases"])
                if name != row["name"] and name not in aliases:
                    aliases.append(name)
                    cur.execute("UPDATE materials SET aliases = ? WHERE material_id = ?",
                               (json.dumps(aliases, ensure_ascii=False), material_id))
                    self._conn.commit()
                return material_id

            cur.execute("""
                INSERT INTO materials
                    (name, aliases, content_hash, freq_hz, er, eim, mr, mim,
                     n_points, f_min_hz, f_max_hz, added_at)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                name, "[]", h,
                json.dumps(freq), json.dumps(er), json.dumps(eim),
                json.dumps(mr), json.dumps(mim),
                len(freq), min(freq), max(freq), _now_iso(),
            ))
            self._conn.commit()
            log.info(f"  БД: новый материал «{name}» (id={cur.lastrowid})")
            return cur.lastrowid

    def register_materials(self, material_datas: list, file_names: list) -> list:
        """
        Массовая регистрация — как использует worker.py после load_material_data().
        material_datas[i] — словарь {'freq','er','eim','mr','mim'} (как из physics.py).
        file_names[i]      — имя файла материала (используется как человекочитаемое имя).

        Возвращает список material_id в ТОМ ЖЕ ПОРЯДКЕ, что и материалы прогона
        (важно: индекс i в sequence конфигураций — это 1-based индекс в этом списке).
        """
        ids = []
        for mat, fname in zip(material_datas, file_names):
            name = os.path.splitext(fname)[0]
            mid = self.register_material(name, mat["freq"], mat["er"],
                                         mat["eim"], mat["mr"], mat["mim"])
            ids.append(mid)
        return ids

    def get_material(self, material_id: int) -> dict | None:
        cur = self._conn.execute("SELECT * FROM materials WHERE material_id = ?",
                                 (material_id,))
        row = cur.fetchone()
        if row is None:
            return None
        d = dict(row)
        for k in ("aliases", "freq_hz", "er", "eim", "mr", "mim"):
            d[k] = json.loads(d[k])
        return d

    def list_materials(self) -> list:
        cur = self._conn.execute(
            "SELECT material_id, name, aliases, n_points, f_min_hz, f_max_hz, added_at "
            "FROM materials ORDER BY material_id")
        return [dict(r) for r in cur.fetchall()]

    # ─────────────────────────────────────────────────────────────────────
    # Прогоны
    # ─────────────────────────────────────────────────────────────────────

    def create_run(self, params: dict, material_ids: list,
                   w_rc: float | None = None, w_th: float | None = None,
                   comment: str = "") -> int:
        """
        Создать запись о прогоне оптимизатора.

        params — тот же словарь, что worker.run_optimisation() получает от gui.py
                 (folder_path, output_dir, mode, N_max_layers, rc_threshold,
                 lambda_multiplier).
        material_ids — список material_id из register_materials(), в порядке,
                       соответствующем индексам 1..N в sequence конфигураций.
        """
        cur = self._conn.execute("""
            INSERT INTO optimisation_runs
                (timestamp, mode, n_max_layers, rc_threshold_db, lambda_multiplier,
                 w_rc, w_th, folder_path, output_dir, material_ids, comment)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            _now_iso(), str(params.get("mode", "")),
            int(params.get("N_max_layers", 0)),
            float(params.get("rc_threshold", 0.0)),
            float(params.get("lambda_multiplier", 1.0)),
            w_rc, w_th,
            params.get("folder_path"), params.get("output_dir"),
            json.dumps(material_ids), comment,
        ))
        self._conn.commit()
        run_id = cur.lastrowid
        log.info(f"  БД: создан прогон run_id={run_id}")
        return run_id

    def get_run(self, run_id: int) -> dict | None:
        cur = self._conn.execute("SELECT * FROM optimisation_runs WHERE run_id = ?",
                                 (run_id,))
        row = cur.fetchone()
        if row is None:
            return None
        d = dict(row)
        d["material_ids"] = json.loads(d["material_ids"])
        return d

    def list_runs(self) -> list:
        cur = self._conn.execute(
            "SELECT run_id, timestamp, mode, n_max_layers, rc_threshold_db, "
            "lambda_multiplier, folder_path, output_dir, comment "
            "FROM optimisation_runs ORDER BY run_id DESC")
        return [dict(r) for r in cur.fetchall()]

    # ─────────────────────────────────────────────────────────────────────
    # Покрытия
    # ─────────────────────────────────────────────────────────────────────

    def add_coating(self, run_id: int, cfg: dict, material_ids: list,
                    rc_threshold: float,
                    freqs=None, rc_curve=None) -> int | None:
        """
        Сохранить одно покрытие (одну конфигурацию из configs, как её собирает
        worker.py). material_ids — список из create_run()/register_materials(),
        cfg['sequence'] — 1-based индексы в этом списке (как в physics.py/plotting.py).

        freqs, rc_curve — если переданы, кривая RC(f) сохраняется целиком
        (см. решение "хранить полную кривую" для будущего ML). Можно не передавать —
        тогда сохранятся только скалярные метрики (быстрее, компактнее).

        Возвращает coating_id, либо None, если такая же конфигурация уже
        сохранена для этого прогона (дубликат тихо пропускается).
        """
        sequence = cfg["sequence"]
        hs_mm    = [h * 1000.0 for h in cfg["hs"]]
        seq_material_ids = [material_ids[i - 1] for i in sequence]

        h = _hash_coating(seq_material_ids, hs_mm)

        sens = cfg.get("sensitivity") or {}
        rc_curve_freq_json = json.dumps(list(np.asarray(freqs, dtype=float))) \
            if freqs is not None else None
        rc_curve_json = json.dumps(list(np.asarray(rc_curve, dtype=float))) \
            if rc_curve is not None else None

        try:
            cur = self._conn.execute("""
                INSERT INTO coatings
                    (run_id, config_hash, n_layers, sequence_material_ids,
                     thicknesses_mm, max_rc_db, total_thickness_mm, bandwidth_ghz,
                     rank, score, sensitivity_json,
                     rc_curve_freq_hz, rc_curve_db, is_valid, created_at)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                run_id, h, cfg["n_layers"], json.dumps(seq_material_ids),
                json.dumps(hs_mm), cfg["max_rc"], cfg["total_mm"],
                cfg.get("bandwidth_ghz"),
                cfg.get("rank"), cfg.get("score"),
                json.dumps(sens, ensure_ascii=False) if sens else None,
                rc_curve_freq_json, rc_curve_json,
                1 if cfg["max_rc"] <= rc_threshold else 0,
                _now_iso(),
            ))
            self._conn.commit()
            return cur.lastrowid
        except sqlite3.IntegrityError:
            # Дубликат в пределах этого прогона — уже сохранён, это нормально
            return None

    def add_coatings(self, run_id: int, configs: list, material_ids: list,
                     rc_threshold: float,
                     freqs=None, mat_arrays=None,
                     only_valid: bool = False,
                     save_curves: bool = True) -> list:
        """
        Массовое сохранение списка конфигураций (то, что worker.py называет
        `pool` / `all_configs`).

        only_valid   — если True, сохранять только конфигурации с
                       max_rc <= rc_threshold (пропускать явно невалидные).
        save_curves  — если True и передан mat_arrays, для каждой конфигурации
                       пересчитывается и сохраняется полная кривая RC(f)
                       (через physics.tmm_reflection). Это дороже по месту в БД,
                       но даёт данные для будущих сеток, предсказывающих кривую,
                       а не только скаляры.

        Возвращает список coating_id (без учёта пропущенных дубликатов/невалидных).
        """
        ids = []
        tmm_reflection = None
        if save_curves and mat_arrays is not None and freqs is not None:
            from physics import tmm_reflection  # локальный импорт — без цикличности

        for cfg in configs:
            if only_valid and cfg["max_rc"] > rc_threshold:
                continue
            rc_curve = None
            curve_freqs = None
            if save_curves and tmm_reflection is not None:
                rc_curve = tmm_reflection(freqs, cfg["sequence"], cfg["hs"], mat_arrays)
                curve_freqs = freqs
            cid = self.add_coating(run_id, cfg, material_ids, rc_threshold,
                                   freqs=curve_freqs, rc_curve=rc_curve)
            if cid is not None:
                ids.append(cid)

        log.info(f"  БД: сохранено {len(ids)} / {len(configs)} покрытий "
                 f"для run_id={run_id}")
        return ids

    def get_coating(self, coating_id: int) -> dict | None:
        cur = self._conn.execute("SELECT * FROM coatings WHERE coating_id = ?",
                                 (coating_id,))
        row = cur.fetchone()
        if row is None:
            return None
        d = dict(row)
        for k in ("sequence_material_ids", "thicknesses_mm"):
            d[k] = json.loads(d[k])
        d["sensitivity"] = json.loads(d["sensitivity_json"]) if d["sensitivity_json"] else {}
        d["rc_curve_freq_hz"] = json.loads(d["rc_curve_freq_hz"]) if d["rc_curve_freq_hz"] else None
        d["rc_curve_db"]      = json.loads(d["rc_curve_db"])      if d["rc_curve_db"]      else None
        return d

    def list_coatings(self, run_id: int | None = None,
                      material_id: int | None = None,
                      only_valid: bool = False,
                      limit: int | None = None) -> list:
        """Найти покрытия с фильтрами. material_id ищет по вхождению в состав слоёв."""
        q = ("SELECT coating_id, run_id, n_layers, sequence_material_ids, "
             "thicknesses_mm, max_rc_db, total_thickness_mm, bandwidth_ghz, "
             "rank, score, is_valid, created_at FROM coatings WHERE 1=1")
        args: list = []
        if run_id is not None:
            q += " AND run_id = ?"
            args.append(run_id)
        if only_valid:
            q += " AND is_valid = 1"
        q += " ORDER BY coating_id DESC"
        if limit:
            q += " LIMIT ?"
            args.append(limit)
        cur = self._conn.execute(q, args)
        rows = [dict(r) for r in cur.fetchall()]
        for r in rows:
            r["sequence_material_ids"] = json.loads(r["sequence_material_ids"])
            r["thicknesses_mm"] = json.loads(r["thicknesses_mm"])
        if material_id is not None:
            rows = [r for r in rows if material_id in r["sequence_material_ids"]]
        return rows

    # ─────────────────────────────────────────────────────────────────────
    # Сводная статистика (для отображения в GUI)
    # ─────────────────────────────────────────────────────────────────────

    def update_coating_metrics(self, coating_id: int, max_rc_db: float,
                               bandwidth_ghz: float | None,
                               is_valid: int, rank: int | None,
                               score: float | None) -> None:
        """
        Обновить пересчитанные метрики покрытия (используется скриптом
        recompute_db.py для исправления исторических записей после фикса
        бага рассинхронизации max_rc — см. recompute_db.py).
        """
        self._conn.execute("""
            UPDATE coatings
            SET max_rc_db = ?, bandwidth_ghz = ?, is_valid = ?, rank = ?, score = ?
            WHERE coating_id = ?
        """, (max_rc_db, bandwidth_ghz, is_valid, rank, score, coating_id))
        self._conn.commit()

    # ─────────────────────────────────────────────────────────────────────
    # Inverse Design — лог запросов "цель → результат"
    # ─────────────────────────────────────────────────────────────────────

    def create_design_target(self, f_lo_hz: float, f_hi_hz: float,
                             rc_target_db: float,
                             max_thickness_mm: float | None = None,
                             allowed_material_ids: list | None = None,
                             comment: str = "") -> int:
        """
        Зафиксировать запрос Inverse Design ("нужна полоса такая-то с таким-то
        порогом RC из вот этих материалов"). Не завязан на конкретный run_id —
        один запрос может привести к нескольким прогонам/попыткам.
        """
        cur = self._conn.execute("""
            INSERT INTO design_targets
                (created_at, f_lo_hz, f_hi_hz, rc_target_db, max_thickness_mm,
                 allowed_material_ids, comment)
            VALUES (?, ?, ?, ?, ?, ?, ?)
        """, (
            _now_iso(), f_lo_hz, f_hi_hz, rc_target_db, max_thickness_mm,
            json.dumps(allowed_material_ids) if allowed_material_ids else None,
            comment,
        ))
        self._conn.commit()
        target_id = cur.lastrowid
        log.info(f"  БД: зафиксирована цель Inverse Design target_id={target_id} "
                f"({f_lo_hz/1e9:.2f}-{f_hi_hz/1e9:.2f} ГГц, RC≤{rc_target_db:+.1f} дБ)")
        return target_id

    def set_design_target_result(self, target_id: int, coating_id: int | None) -> None:
        """Привязать результат (лучшее найденное покрытие) к запросу цели."""
        self._conn.execute(
            "UPDATE design_targets SET resulting_coating_id = ? WHERE target_id = ?",
            (coating_id, target_id))
        self._conn.commit()

    def get_design_target(self, target_id: int) -> dict | None:
        cur = self._conn.execute(
            "SELECT * FROM design_targets WHERE target_id = ?", (target_id,))
        row = cur.fetchone()
        if row is None:
            return None
        d = dict(row)
        d["allowed_material_ids"] = (json.loads(d["allowed_material_ids"])
                                     if d["allowed_material_ids"] else None)
        return d

    def list_design_targets(self) -> list:
        cur = self._conn.execute(
            "SELECT * FROM design_targets ORDER BY target_id DESC")
        rows = [dict(r) for r in cur.fetchall()]
        for r in rows:
            r["allowed_material_ids"] = (json.loads(r["allowed_material_ids"])
                                        if r["allowed_material_ids"] else None)
        return rows

    def set_material_aliases_text(self, material_id: int, aliases_text: str) -> None:
        """
        Установить «Другие имена» материала из свободного текста, введённого
        пользователем в окне БД (через запятую — несколько записей).
        Полностью заменяет прежний список aliases (а не дописывает к нему).
        """
        aliases = [a.strip() for a in aliases_text.split(",") if a.strip()]
        self._conn.execute(
            "UPDATE materials SET aliases = ? WHERE material_id = ?",
            (json.dumps(aliases, ensure_ascii=False), material_id))
        self._conn.commit()

    def set_run_comment(self, run_id: int, comment: str) -> None:
        """Установить комментарий прогона (редактируется в окне БД)."""
        self._conn.execute(
            "UPDATE optimisation_runs SET comment = ? WHERE run_id = ?",
            (comment, run_id))
        self._conn.commit()

    def stats(self) -> dict:
        cur = self._conn.cursor()
        n_materials = cur.execute("SELECT COUNT(*) FROM materials").fetchone()[0]
        n_runs      = cur.execute("SELECT COUNT(*) FROM optimisation_runs").fetchone()[0]
        n_coatings  = cur.execute("SELECT COUNT(*) FROM coatings").fetchone()[0]
        n_valid     = cur.execute("SELECT COUNT(*) FROM coatings WHERE is_valid = 1").fetchone()[0]
        best_rc     = cur.execute("SELECT MIN(max_rc_db) FROM coatings").fetchone()[0]
        thinnest    = cur.execute("SELECT MIN(total_thickness_mm) FROM coatings "
                                  "WHERE is_valid = 1").fetchone()[0]
        return {
            "n_materials": n_materials,
            "n_runs": n_runs,
            "n_coatings": n_coatings,
            "n_valid_coatings": n_valid,
            "best_rc_db": best_rc,
            "thinnest_valid_mm": thinnest,
            "db_path": self.db_path,
            "db_size_mb": round(os.path.getsize(self.db_path) / 1e6, 2)
                         if os.path.exists(self.db_path) else 0.0,
        }


if __name__ == "__main__":
    # Небольшая самопроверка модуля на синтетических данных
    logging.basicConfig(level=logging.INFO)
    import tempfile

    tmp_path = os.path.join(tempfile.gettempdir(), "rc_absorber_selftest.db")
    if os.path.exists(tmp_path):
        os.remove(tmp_path)

    db = Database(tmp_path)

    freq = np.linspace(8e9, 12e9, 50)
    mat1 = {"freq": freq, "er": np.full(50, 10.0), "eim": np.full(50, 1.0),
           "mr": np.full(50, 1.0), "mim": np.full(50, 0.0)}
    mat2 = {"freq": freq, "er": np.full(50, 5.0), "eim": np.full(50, 0.5),
           "mr": np.full(50, 1.0), "mim": np.full(50, 0.0)}

    mat_ids = db.register_materials([mat1, mat2], ["CNT_composite_A.xlsx",
                                                   "CNT_composite_B.xlsx"])
    print("material_ids:", mat_ids)

    params = {"folder_path": "/tmp/mats", "output_dir": "/tmp/out",
             "mode": "3", "N_max_layers": 3,
             "rc_threshold": -10.0, "lambda_multiplier": 1.0}
    run_id = db.create_run(params, mat_ids, w_rc=1.0, w_th=0.3)
    print("run_id:", run_id)

    cfg = {"sequence": (1, 2), "hs": (0.001, 0.002), "n_layers": 2,
          "max_rc": -14.3, "total_mm": 3.0, "bandwidth_ghz": 1.8,
          "rank": 1, "score": 0.12,
          "sensitivity": {"per_layer_dRC": [0.5, 1.2], "max_sensitivity": 1.2,
                         "mean_sensitivity": 0.85, "robust_score": 2}}
    cid = db.add_coating(run_id, cfg, mat_ids, rc_threshold=-10.0)
    print("coating_id:", cid)
    dup = db.add_coating(run_id, cfg, mat_ids, rc_threshold=-10.0)
    print("dup (должно быть None):", dup)

    print("stats:", db.stats())
    print("materials:", db.list_materials())
    print("coatings:", db.list_coatings(run_id=run_id))

    db.close()
    os.remove(tmp_path)
    print("Самопроверка OK.")
