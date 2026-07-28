"""
db_viewer.py — окно просмотра всей локальной базы данных

В отличие от preview.py (который показывает результаты ОДНОГО прогона,
ещё не сохранённые на диск), это окно показывает ВСЮ накопленную историю
из rc_absorber.db — по всем прогонам за всё время.

Четыре вкладки:
  • Материалы — все материалы, когда-либо загруженные (с дедупликацией)
  • Прогоны   — все прогоны оптимизатора с их параметрами
  • Покрытия  — все посчитанные конфигурации с фильтрами по прогону/материалу/
                валидности и графиком RC(f) при выборе строки
                (график восстанавливается из сохранённых спектров материалов —
                это одновременно проверка, что раунд-трип «БД → график» работает)
  • Экспорт датасета — выбор конкретных прогонов и формирование обучающей
                выборки (dataset_export.py) без командной строки
"""

import os
import json
import threading
import logging

import tkinter as tk
from tkinter import ttk, filedialog, messagebox

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg

from database import Database
from physics import build_mat_arrays
from plotting import make_config_figure
from dataset_export import export_dataset, preview_selection

try:
    from ml_surrogate import SurrogateModel, load_dataset as load_ml_dataset
    _ML_AVAILABLE = True
    _ML_IMPORT_ERROR = None
except ImportError as e:
    _ML_AVAILABLE = False
    _ML_IMPORT_ERROR = str(e)

log = logging.getLogger("rc_opt")

# ─────────────────────────────────────────────────────────────────────────────
# Цветовая схема (та же, что в gui.py / preview.py)
# ─────────────────────────────────────────────────────────────────────────────
BG       = "#ece9d8"
BG_DARK  = "#d4d0c8"
TITLE_BG = "#0a246a"
TITLE_FG = "#ffffff"
FG       = "#000000"
FG_DIM   = "#444444"
SEL_BG   = "#316ac5"
FONT     = ("Tahoma", 9)
FONT_B   = ("Tahoma", 9, "bold")
FONT_SM  = ("Tahoma", 8)


# ─────────────────────────────────────────────────────────────────────────────
# Восстановление данных для графика по записи из БД (без зависимости от Tk —
# можно тестировать отдельно)
# ─────────────────────────────────────────────────────────────────────────────

def build_plot_context(db: Database, coating: dict):
    """
    По записи покрытия (из db.get_coating()) восстановить всё, что нужно
    plotting.make_config_figure(): cfg, freqs, mat_arrays, material_names.

    Работает независимо от того, из какого прогона взято покрытие — тянет
    материалы напрямую по их material_id, сохранённым в самом покрытии.
    Не требует, чтобы состав материалов прогона совпадал с текущей папкой
    на диске: все данные уже лежат в БД.

    Вызывает ValueError, если материалы покрытия имеют несовпадающие
    частотные сетки (в норме такого не бывает — все материалы одного
    прогона проходят проверку сеток в load_material_data при загрузке).
    """
    seq_ids = coating["sequence_material_ids"]
    unique_ids = list(dict.fromkeys(seq_ids))  # уникальные, сохраняя порядок

    materials = [db.get_material(mid) for mid in unique_ids]
    if any(m is None for m in materials):
        raise ValueError("Один из материалов покрытия отсутствует в БД.")

    freq0 = materials[0]["freq_hz"]
    for m in materials[1:]:
        if len(m["freq_hz"]) != len(freq0):
            raise ValueError(
                "Материалы этого покрытия измерены на разных частотных "
                "сетках — построить график нельзя.")

    material_datas = [
        {"freq": np.array(m["freq_hz"]), "er": np.array(m["er"]),
         "eim": np.array(m["eim"]), "mr": np.array(m["mr"]),
         "mim": np.array(m["mim"])}
        for m in materials
    ]
    mat_arrays = build_mat_arrays(material_datas)
    material_names = [m["name"] for m in materials]

    id_to_local = {mid: i + 1 for i, mid in enumerate(unique_ids)}
    local_sequence = tuple(id_to_local[mid] for mid in seq_ids)
    hs = tuple(t / 1000.0 for t in coating["thicknesses_mm"])

    cfg = {
        "sequence": local_sequence,
        "hs": hs,
        "n_layers": coating["n_layers"],
        "max_rc": coating["max_rc_db"],
        "total_mm": coating["total_thickness_mm"],
        "bandwidth_ghz": coating.get("bandwidth_ghz"),
        "sensitivity": coating.get("sensitivity") or {},
        "rank": coating.get("rank") if coating.get("rank") is not None else "?",
        "score": coating.get("score") if coating.get("score") is not None else float("inf"),
    }
    freqs = np.array(freq0)
    return cfg, freqs, mat_arrays, material_names


# ─────────────────────────────────────────────────────────────────────────────
# Окно
# ─────────────────────────────────────────────────────────────────────────────

class DatabaseWindow(tk.Toplevel):
    """Окно просмотра всей накопленной базы данных."""

    def __init__(self, parent, db_path: str | None = None):
        super().__init__(parent)
        self.title("База данных — история вычислений")
        self.configure(bg=BG)
        self.resizable(True, True)
        self.geometry("1280x760")
        self.minsize(900, 560)

        self._db = Database(db_path) if db_path else Database()
        self._current_fig = None
        self._mpl_canvas = None

        self._build_ui()
        self.refresh()

        self.protocol("WM_DELETE_WINDOW", self._on_close)

    # ── Каркас окна ──────────────────────────────────────────────────────────

    def _build_ui(self) -> None:
        title_bar = tk.Frame(self, bg=TITLE_BG, height=26)
        title_bar.pack(side="top", fill="x")
        tk.Label(title_bar, text="  База данных — история вычислений",
                 bg=TITLE_BG, fg=TITLE_FG, font=FONT_B, anchor="w"
                 ).pack(side="left", padx=4, pady=3)

        # ── Сводная строка статистики ────────────────────────────────────────
        stats_bar = tk.Frame(self, bg=BG_DARK, bd=1, relief="groove")
        stats_bar.pack(side="top", fill="x", padx=4, pady=(4, 0))
        self._var_stats = tk.StringVar(value="")
        tk.Label(stats_bar, textvariable=self._var_stats,
                 bg=BG_DARK, fg=FG, font=FONT_SM, anchor="w", padx=6, pady=3
                 ).pack(side="left")
        ttk.Button(stats_bar, text="Обновить", command=self.refresh
                   ).pack(side="right", padx=4, pady=2)

        # ── Вкладки ───────────────────────────────────────────────────────────
        style = ttk.Style()
        style.configure("TNotebook", background=BG)
        style.configure("TNotebook.Tab", font=FONT, padding=(10, 4))

        nb = ttk.Notebook(self)
        nb.pack(fill="both", expand=True, padx=4, pady=4)
        self._nb = nb

        self._tab_materials = tk.Frame(nb, bg=BG)
        self._tab_runs       = tk.Frame(nb, bg=BG)
        self._tab_coatings   = tk.Frame(nb, bg=BG)
        self._tab_export     = tk.Frame(nb, bg=BG)
        self._tab_surrogate  = tk.Frame(nb, bg=BG)
        self._tab_targets    = tk.Frame(nb, bg=BG)
        nb.add(self._tab_materials, text="Материалы")
        nb.add(self._tab_runs,      text="Прогоны")
        nb.add(self._tab_coatings,  text="Покрытия")
        nb.add(self._tab_export,    text="Экспорт датасета")
        nb.add(self._tab_surrogate, text="ML-суррогат")
        nb.add(self._tab_targets,   text="История целей")

        self._build_materials_tab()
        self._build_runs_tab()
        self._build_coatings_tab()
        self._build_export_tab()
        self._build_surrogate_tab()
        self._build_targets_tab()

    # ── Вкладка «Материалы» ──────────────────────────────────────────────────

    def _build_materials_tab(self) -> None:
        frame = self._tab_materials
        frame.rowconfigure(0, weight=1)
        frame.columnconfigure(0, weight=1)

        cols = [("id", "ID", 40), ("name", "Название", 220),
                ("aliases", "Другие имена", 220), ("n_points", "Точек", 60),
                ("f_min", "f мин (ГГц)", 90), ("f_max", "f макс (ГГц)", 90),
                ("added_at", "Добавлен", 160)]
        tree = ttk.Treeview(frame, columns=[c[0] for c in cols], show="headings")
        for cid, cname, cw in cols:
            tree.heading(cid, text=cname)
            tree.column(cid, width=cw, anchor="center", stretch=(cid == "name"))
        vsb = ttk.Scrollbar(frame, orient="vertical", command=tree.yview)
        tree.configure(yscrollcommand=vsb.set)
        tree.grid(row=0, column=0, sticky="nsew", padx=(4, 0), pady=4)
        vsb.grid(row=0, column=1, sticky="ns", pady=4)
        tree.bind("<Double-1>", self._on_material_cell_double_click)
        self._tree_materials = tree

        tk.Label(frame, text="Двойной клик по ячейке «Другие имена» — "
                            "редактирование (через запятую — несколько записей).",
                bg=BG, fg=FG_DIM, font=FONT_SM, anchor="w"
                ).grid(row=1, column=0, columnspan=2, sticky="w", padx=4)

    def _on_material_cell_double_click(self, event) -> None:
        tree = self._tree_materials
        if tree.identify_region(event.x, event.y) != "cell":
            return
        row_id = tree.identify_row(event.y)
        col_id = tree.identify_column(event.x)  # например '#3'
        if not row_id:
            return
        col_index = int(col_id.replace("#", "")) - 1
        columns = tree["columns"]
        if columns[col_index] != "aliases":
            return  # редактируется только столбец «Другие имена»

        bbox = tree.bbox(row_id, col_id)
        if not bbox:
            return
        x, y, w, h = bbox
        current_value = tree.set(row_id, "aliases")

        entry = tk.Entry(tree, font=FONT)
        entry.insert(0, current_value)
        entry.select_range(0, "end")
        entry.place(x=x, y=y, width=w, height=h)
        entry.focus_set()

        material_id = int(tree.set(row_id, "id"))

        def save(event=None):
            new_value = entry.get()
            entry.destroy()
            self._db.set_material_aliases_text(material_id, new_value)
            self._refresh_materials()

        def cancel(event=None):
            entry.destroy()

        entry.bind("<Return>", save)
        entry.bind("<FocusOut>", save)
        entry.bind("<Escape>", cancel)
    def _refresh_materials(self) -> None:
        tree = self._tree_materials
        for row in tree.get_children():
            tree.delete(row)
        for m in self._db.list_materials():
            aliases = ", ".join(json.loads(m["aliases"])) if m["aliases"] else ""
            tree.insert("", "end", iid=str(m["material_id"]), values=(
                m["material_id"], m["name"], aliases, m["n_points"],
                f"{m['f_min_hz']/1e9:.2f}", f"{m['f_max_hz']/1e9:.2f}",
                m["added_at"][:19],
            ))

    # ── Вкладка «Прогоны» ────────────────────────────────────────────────────

    def _build_runs_tab(self) -> None:
        frame = self._tab_runs
        frame.rowconfigure(0, weight=1)
        frame.columnconfigure(0, weight=1)

        cols = [("id", "ID", 40), ("ts", "Время", 160), ("mode", "Режим", 60),
                ("n", "N слоёв", 70), ("rc", "Порог RC", 80),
                ("lam", "×λ", 60), ("folder", "Папка материалов", 260),
                ("comment", "Комментарий", 160)]
        tree = ttk.Treeview(frame, columns=[c[0] for c in cols], show="headings")
        for cid, cname, cw in cols:
            tree.heading(cid, text=cname)
            tree.column(cid, width=cw, anchor="center",
                       stretch=(cid in ("folder", "comment")))
        vsb = ttk.Scrollbar(frame, orient="vertical", command=tree.yview)
        tree.configure(yscrollcommand=vsb.set)
        tree.grid(row=0, column=0, sticky="nsew", padx=(4, 0), pady=4)
        vsb.grid(row=0, column=1, sticky="ns", pady=4)
        tree.bind("<Double-1>", self._on_run_cell_double_click)
        self._tree_runs = tree

        tk.Label(frame, text="Двойной клик по ячейке «Комментарий» — редактирование.",
                bg=BG, fg=FG_DIM, font=FONT_SM, anchor="w"
                ).grid(row=1, column=0, columnspan=2, sticky="w", padx=4)

    def _on_run_cell_double_click(self, event) -> None:
        tree = self._tree_runs
        if tree.identify_region(event.x, event.y) != "cell":
            return
        row_id = tree.identify_row(event.y)
        col_id = tree.identify_column(event.x)
        if not row_id:
            return
        col_index = int(col_id.replace("#", "")) - 1
        columns = tree["columns"]
        if columns[col_index] != "comment":
            return  # редактируется только столбец «Комментарий»

        bbox = tree.bbox(row_id, col_id)
        if not bbox:
            return
        x, y, w, h = bbox
        current_value = tree.set(row_id, "comment")

        entry = tk.Entry(tree, font=FONT)
        entry.insert(0, current_value)
        entry.select_range(0, "end")
        entry.place(x=x, y=y, width=w, height=h)
        entry.focus_set()

        run_id = int(tree.set(row_id, "id"))

        def save(event=None):
            new_value = entry.get()
            entry.destroy()
            self._db.set_run_comment(run_id, new_value)
            self._refresh_runs()

        def cancel(event=None):
            entry.destroy()

        entry.bind("<Return>", save)
        entry.bind("<FocusOut>", save)
        entry.bind("<Escape>", cancel)
    def _refresh_runs(self) -> None:
        tree = self._tree_runs
        for row in tree.get_children():
            tree.delete(row)
        for r in self._db.list_runs():
            tree.insert("", "end", iid=str(r["run_id"]), values=(
                r["run_id"], r["timestamp"][:19], r["mode"], r["n_max_layers"],
                f"{r['rc_threshold_db']:+.1f}", f"{r['lambda_multiplier']:.3g}",
                r.get("folder_path") or "", r.get("comment") or "",
            ))

    # ── Вкладка «Покрытия» ───────────────────────────────────────────────────

    def _build_coatings_tab(self) -> None:
        frame = self._tab_coatings
        frame.columnconfigure(0, weight=0)
        frame.columnconfigure(1, weight=1)
        frame.rowconfigure(1, weight=1)

        # Фильтры
        filt = tk.Frame(frame, bg=BG_DARK, bd=1, relief="groove")
        filt.grid(row=0, column=0, columnspan=2, sticky="ew", padx=4, pady=(4, 0))

        tk.Label(filt, text="Прогон:", bg=BG_DARK, fg=FG, font=FONT_SM
                 ).pack(side="left", padx=(6, 2), pady=4)
        self._var_filter_run = tk.StringVar(value="Все")
        self._cb_filter_run = ttk.Combobox(filt, textvariable=self._var_filter_run,
                                           state="readonly", width=28)
        self._cb_filter_run.pack(side="left", padx=(0, 10))
        self._cb_filter_run.bind("<<ComboboxSelected>>", lambda e: self._refresh_coatings())

        tk.Label(filt, text="Материал:", bg=BG_DARK, fg=FG, font=FONT_SM
                 ).pack(side="left", padx=(6, 2))
        self._var_filter_mat = tk.StringVar(value="Все")
        self._cb_filter_mat = ttk.Combobox(filt, textvariable=self._var_filter_mat,
                                           state="readonly", width=24)
        self._cb_filter_mat.pack(side="left", padx=(0, 10))
        self._cb_filter_mat.bind("<<ComboboxSelected>>", lambda e: self._refresh_coatings())

        self._var_only_valid = tk.BooleanVar(value=False)
        ttk.Checkbutton(filt, text="Только валидные (RC ≤ порога)",
                       variable=self._var_only_valid,
                       command=self._refresh_coatings
                       ).pack(side="left", padx=(6, 6))

        # Левая колонка — список покрытий
        left = tk.Frame(frame, bg=BG, bd=1, relief="groove")
        left.grid(row=1, column=0, sticky="ns", padx=(4, 6), pady=4)
        left.rowconfigure(1, weight=1)

        tk.Label(left, text="Покрытия", bg=BG_DARK, fg=FG, font=FONT_B,
                 anchor="w", padx=6, pady=3).grid(row=0, column=0, sticky="ew")

        cols = [("id", "ID", 45), ("run", "Прогон", 55), ("n", "Сл.", 40),
                ("rc", "RC (дБ)", 70), ("th", "h (мм)", 65),
                ("bw", "Полоса", 65), ("valid", "✓", 30)]
        tree = ttk.Treeview(left, columns=[c[0] for c in cols], show="headings",
                            height=22)
        for cid, cname, cw in cols:
            tree.heading(cid, text=cname)
            tree.column(cid, width=cw, anchor="center", stretch=False)
        vsb = ttk.Scrollbar(left, orient="vertical", command=tree.yview)
        tree.configure(yscrollcommand=vsb.set)
        tree.grid(row=1, column=0, sticky="nsew", padx=(2, 0), pady=2)
        vsb.grid(row=1, column=1, sticky="ns", pady=2)
        tree.bind("<<TreeviewSelect>>", self._on_coating_select)
        self._tree_coatings = tree

        # Правая колонка — график + детали
        right = tk.Frame(frame, bg=BG, bd=2, relief="sunken")
        right.grid(row=1, column=1, sticky="nsew", pady=4)
        right.rowconfigure(0, weight=1)
        right.rowconfigure(1, weight=0)
        right.columnconfigure(0, weight=1)

        self._canvas_frame = tk.Frame(right, bg=BG)
        self._canvas_frame.grid(row=0, column=0, sticky="nsew")
        self._canvas_frame.rowconfigure(0, weight=1)
        self._canvas_frame.columnconfigure(0, weight=1)

        self._placeholder = tk.Label(
            self._canvas_frame, text="← Выберите покрытие в списке",
            bg=BG, fg=FG_DIM, font=FONT)
        self._placeholder.grid(row=0, column=0, sticky="nsew")

        self._var_detail = tk.StringVar(value="")
        tk.Label(right, textvariable=self._var_detail, bg=BG_DARK, fg=FG_DIM,
                 font=FONT_SM, anchor="w", justify="left", padx=6, pady=3
                 ).grid(row=1, column=0, sticky="ew")

    def _refresh_filter_options(self) -> None:
        runs = self._db.list_runs()
        run_labels = ["Все"] + [
            f"{r['run_id']} — {r['timestamp'][:16]} (реж.{r['mode']})" for r in runs
        ]
        self._cb_filter_run["values"] = run_labels
        if self._var_filter_run.get() not in run_labels:
            self._var_filter_run.set("Все")

        mats = self._db.list_materials()
        mat_labels = ["Все"] + [f"{m['material_id']} — {m['name']}" for m in mats]
        self._cb_filter_mat["values"] = mat_labels
        if self._var_filter_mat.get() not in mat_labels:
            self._var_filter_mat.set("Все")

    def _refresh_coatings(self) -> None:
        tree = self._tree_coatings
        for row in tree.get_children():
            tree.delete(row)

        run_sel = self._var_filter_run.get()
        run_id = int(run_sel.split(" — ")[0]) if run_sel != "Все" else None
        mat_sel = self._var_filter_mat.get()
        material_id = int(mat_sel.split(" — ")[0]) if mat_sel != "Все" else None
        only_valid = self._var_only_valid.get()

        coatings = self._db.list_coatings(run_id=run_id, material_id=material_id,
                                          only_valid=only_valid, limit=2000)
        for c in coatings:
            tree.insert("", "end", iid=str(c["coating_id"]), values=(
                c["coating_id"], c["run_id"], c["n_layers"],
                f"{c['max_rc_db']:+.1f}", f"{c['total_thickness_mm']:.1f}",
                f"{(c.get('bandwidth_ghz') or 0):.2f}",
                "✓" if c["is_valid"] else "",
            ))

    def _on_coating_select(self, event) -> None:
        sel = self._tree_coatings.selection()
        if not sel:
            return
        coating_id = int(sel[0])
        coating = self._db.get_coating(coating_id)
        if coating is None:
            return
        self._show_coating(coating)

    def _show_coating(self, coating: dict) -> None:
        if self._mpl_canvas is not None:
            self._mpl_canvas.get_tk_widget().destroy()
            self._mpl_canvas = None
        if self._current_fig is not None:
            plt.close(self._current_fig)
            self._current_fig = None

        run = self._db.get_run(coating["run_id"])
        rc_threshold = run["rc_threshold_db"] if run else -10.0

        try:
            cfg, freqs, mat_arrays, names = build_plot_context(self._db, coating)
        except ValueError as e:
            self._placeholder.config(text=f"Не удалось построить график:\n{e}")
            self._placeholder.grid()
            self._var_detail.set("")
            return

        self._placeholder.grid_remove()
        fig = make_config_figure(cfg, freqs, mat_arrays, names,
                                 rc_threshold=rc_threshold)
        self._current_fig = fig
        canvas = FigureCanvasTkAgg(fig, master=self._canvas_frame)
        canvas.draw()
        canvas.get_tk_widget().grid(row=0, column=0, sticky="nsew")
        self._mpl_canvas = canvas

        mats_str = ", ".join(names)
        self._var_detail.set(
            f"Покрытие #{coating['coating_id']}  |  Прогон #{coating['run_id']}  |  "
            f"Материалы: {mats_str}  |  "
            f"Сохранена кривая RC(f): {'да' if coating.get('rc_curve_db') else 'нет'}"
        )

    # ── Вкладка «Экспорт датасета» ───────────────────────────────────────────

    def _build_export_tab(self) -> None:
        frame = self._tab_export
        frame.columnconfigure(0, weight=1)
        frame.columnconfigure(1, weight=1)
        frame.rowconfigure(0, weight=1)

        self._export_state = {"running": False}

        # Левая колонка — выбор прогонов
        left = tk.Frame(frame, bg=BG, bd=1, relief="groove")
        left.grid(row=0, column=0, sticky="nsew", padx=(4, 2), pady=4)
        left.rowconfigure(1, weight=1)
        left.columnconfigure(0, weight=1)

        tk.Label(left, text="Прогоны (Ctrl/Shift — несколько; ничего не "
                            "выбрано = вся БД)",
                 bg=BG_DARK, fg=FG, font=FONT_B, anchor="w", padx=6, pady=3,
                 wraplength=280, justify="left").grid(row=0, column=0, sticky="ew")

        cols = [("id", "ID", 40), ("ts", "Время", 130), ("mode", "Реж.", 40),
                ("n", "N сл.", 50), ("rc", "Порог", 60),
                ("band", "Диапазон (ГГц)", 120), ("coatings", "Покрытий", 70)]
        tree = ttk.Treeview(left, columns=[c[0] for c in cols], show="headings",
                            selectmode="extended", height=18)
        for cid, cname, cw in cols:
            tree.heading(cid, text=cname)
            tree.column(cid, width=cw, anchor="center", stretch=False)
        vsb = ttk.Scrollbar(left, orient="vertical", command=tree.yview)
        tree.configure(yscrollcommand=vsb.set)
        tree.grid(row=1, column=0, sticky="nsew", padx=(2, 0), pady=2)
        vsb.grid(row=1, column=1, sticky="ns", pady=2)
        tree.bind("<<TreeviewSelect>>", lambda e: self._update_export_preview())
        self._tree_export_runs = tree

        # Правая колонка — параметры и запуск
        right = tk.Frame(frame, bg=BG, bd=1, relief="groove")
        right.grid(row=0, column=1, sticky="nsew", padx=(2, 4), pady=4)
        right.columnconfigure(0, weight=1)

        tk.Label(right, text="Параметры экспорта", bg=BG_DARK, fg=FG, font=FONT_B,
                 anchor="w", padx=6, pady=3).grid(row=0, column=0, sticky="ew")

        opts = tk.Frame(right, bg=BG, padx=8, pady=8)
        opts.grid(row=1, column=0, sticky="ew")
        opts.columnconfigure(1, weight=1)

        self._var_exp_only_valid = tk.BooleanVar(value=True)
        ttk.Checkbutton(opts, text="Только валидные покрытия",
                       variable=self._var_exp_only_valid,
                       command=self._update_export_preview
                       ).grid(row=0, column=0, columnspan=2, sticky="w", pady=2)

        self._var_exp_curves = tk.BooleanVar(value=True)
        ttk.Checkbutton(opts, text="Включить полные кривые (спектры + RC(f))",
                       variable=self._var_exp_curves
                       ).grid(row=1, column=0, columnspan=2, sticky="w", pady=2)

        tk.Label(opts, text="Точек в сетке частот:", bg=BG, fg=FG, font=FONT_SM
                 ).grid(row=2, column=0, sticky="w", pady=(6, 2))
        self._var_exp_grid_points = tk.IntVar(value=201)
        ttk.Spinbox(opts, from_=10, to=2000, increment=1,
                   textvariable=self._var_exp_grid_points, width=8
                   ).grid(row=2, column=1, sticky="w", pady=(6, 2))

        tk.Label(opts, text="Папка для экспорта:", bg=BG, fg=FG, font=FONT_SM
                 ).grid(row=3, column=0, columnspan=2, sticky="w", pady=(6, 2))
        out_frame = tk.Frame(opts, bg=BG)
        out_frame.grid(row=4, column=0, columnspan=2, sticky="ew")
        out_frame.columnconfigure(0, weight=1)
        default_outdir = os.path.join(os.path.dirname(self._db.db_path) or ".",
                                      "ml_dataset")
        self._var_exp_outdir = tk.StringVar(value=default_outdir)
        ttk.Entry(out_frame, textvariable=self._var_exp_outdir
                 ).grid(row=0, column=0, sticky="ew")
        ttk.Button(out_frame, text="Обзор…", command=self._browse_export_dir
                  ).grid(row=0, column=1, padx=(4, 0))

        self._var_exp_preview = tk.StringVar(
            value="Выберите прогоны слева (или оставьте пусто — тогда вся БД).")
        tk.Label(right, textvariable=self._var_exp_preview, bg=BG, fg=FG_DIM,
                 font=FONT_SM, anchor="w", justify="left", wraplength=320,
                 padx=8, pady=6).grid(row=2, column=0, sticky="ew")

        self._btn_export = ttk.Button(right, text="Сформировать датасет",
                                      command=self._start_export)
        self._btn_export.grid(row=3, column=0, sticky="ew", padx=8, pady=(4, 2),
                              ipady=4)

        self._pb_export = ttk.Progressbar(right, mode="determinate")
        self._pb_export.grid(row=4, column=0, sticky="ew", padx=8, pady=(0, 4))

        self._var_exp_status = tk.StringVar(value="")
        tk.Label(right, textvariable=self._var_exp_status, bg=BG, fg=FG_DIM,
                 font=FONT_SM, anchor="w", justify="left", wraplength=320,
                 padx=8, pady=2).grid(row=5, column=0, sticky="ew")

    def _run_frequency_band_str(self, run_id: int) -> str:
        """
        Пересечение диапазонов частот материалов, использованных в прогоне
        (та же логика, что dataset_export.preview_selection). Кэшируется на
        время жизни окна — материалы прогона не меняются задним числом.
        """
        if not hasattr(self, "_run_band_cache"):
            self._run_band_cache = {}
        if run_id in self._run_band_cache:
            return self._run_band_cache[run_id]

        run = self._db.get_run(run_id)
        band_str = "—"
        if run and run.get("material_ids"):
            f_los, f_his = [], []
            for mid in run["material_ids"]:
                m = self._db.get_material(mid)
                if m:
                    f_los.append(m["f_min_hz"])
                    f_his.append(m["f_max_hz"])
            if f_los:
                f_lo, f_hi = max(f_los), min(f_his)
                band_str = (f"{f_lo/1e9:.2f}–{f_hi/1e9:.2f}"
                           if f_hi > f_lo else "не пересек.")
        self._run_band_cache[run_id] = band_str
        return band_str

    def _refresh_export_runs(self) -> None:
        tree = self._tree_export_runs
        prev_selection = tree.selection()
        for row in tree.get_children():
            tree.delete(row)
        self._run_band_cache = {}  # состав материалов прогонов мог обновиться
        for r in self._db.list_runs():
            n_coatings = len(self._db.list_coatings(run_id=r["run_id"]))
            band_str = self._run_frequency_band_str(r["run_id"])
            tree.insert("", "end", iid=str(r["run_id"]), values=(
                r["run_id"], r["timestamp"][:16], r["mode"], r["n_max_layers"],
                f"{r['rc_threshold_db']:+.1f}", band_str, n_coatings,
            ))
        for iid in prev_selection:
            if tree.exists(iid):
                tree.selection_add(iid)
        self._update_export_preview()

    def _get_selected_run_ids(self) -> list | None:
        sel = self._tree_export_runs.selection()
        return [int(s) for s in sel] if sel else None

    def _update_export_preview(self) -> None:
        run_ids = self._get_selected_run_ids()
        only_valid = self._var_exp_only_valid.get()
        prev = preview_selection(self._db.db_path, run_ids, only_valid)
        scope = "всех прогонов" if run_ids is None else f"{len(run_ids)} выбранных прогонов"
        if prev.get("ok"):
            self._var_exp_preview.set(
                f"Выбрано: {scope}.\n"
                f"Материалов: {prev['n_materials']}   Покрытий: {prev['n_coatings']}\n"
                f"Каноническая сетка частот: "
                f"{prev['f_min_ghz']:.2f}–{prev['f_max_ghz']:.2f} ГГц "
                f"(пересечение диапазонов задействованных материалов)")
        else:
            self._var_exp_preview.set(f"Выбрано: {scope}.\n"
                                      f"{prev.get('message', 'Нет данных.')}")

    def _browse_export_dir(self) -> None:
        d = filedialog.askdirectory(
            initialdir=self._var_exp_outdir.get() or os.getcwd(), parent=self)
        if d:
            self._var_exp_outdir.set(d)

    def _start_export(self) -> None:
        if self._export_state.get("running"):
            return

        run_ids = self._get_selected_run_ids()
        only_valid = self._var_exp_only_valid.get()
        include_curves = self._var_exp_curves.get()
        grid_points = self._var_exp_grid_points.get()
        output_dir = self._var_exp_outdir.get().strip()

        if not output_dir:
            messagebox.showerror("Экспорт датасета",
                                 "Укажите папку для экспорта.", parent=self)
            return

        prev = preview_selection(self._db.db_path, run_ids, only_valid)
        if not prev.get("ok"):
            messagebox.showerror("Экспорт датасета",
                                 prev.get("message", "Нет данных для экспорта."),
                                 parent=self)
            return

        self._export_state = {"running": True, "done": 0,
                              "total": max(prev["n_coatings"], 1),
                              "summary": None, "error": None}
        self._pb_export.configure(mode="determinate",
                                  maximum=self._export_state["total"], value=0)
        self._btn_export.configure(state="disabled")
        self._var_exp_status.set(f"Экспорт запущен: {prev['n_coatings']} покрытий, "
                                 f"{prev['n_materials']} материалов…")

        t = threading.Thread(
            target=self._export_worker,
            args=(run_ids, only_valid, include_curves, grid_points, output_dir),
            daemon=True)
        t.start()
        self.after(150, self._poll_export)

    def _export_worker(self, run_ids, only_valid, include_curves,
                       grid_points, output_dir) -> None:
        def progress_cb(done, total):
            self._export_state["done"] = done
            self._export_state["total"] = max(total, 1)

        try:
            summary = export_dataset(
                db_path=self._db.db_path, output_dir=output_dir,
                grid_points=grid_points, only_valid=only_valid,
                include_curves=include_curves, run_ids=run_ids,
                progress_cb=progress_cb,
            )
            self._export_state["summary"] = summary
        except Exception as e:
            log.exception("Ошибка при формировании датасета")
            self._export_state["error"] = str(e)
        finally:
            self._export_state["running"] = False

    def _poll_export(self) -> None:
        st = self._export_state
        self._pb_export.configure(maximum=max(st.get("total", 1), 1))
        self._pb_export.configure(value=st.get("done", 0))
        if st.get("running"):
            self.after(150, self._poll_export)
            return

        self._btn_export.configure(state="normal")
        if st.get("error"):
            self._var_exp_status.set(f"Ошибка: {st['error']}")
            messagebox.showerror("Экспорт датасета", st["error"], parent=self)
            return

        summary = st.get("summary")
        if not summary:
            return
        self._var_exp_status.set(
            f"Готово: {summary['n_samples']} сэмплов "
            f"({summary['n_valid']} валидных), {summary['grid_points']} точек, "
            f"{summary['f_min_hz']/1e9:.2f}–{summary['f_max_hz']/1e9:.2f} ГГц\n"
            f"→ {summary['csv_path']}"
        )
        if messagebox.askyesno(
                "Экспорт датасета",
                f"Готово: {summary['n_samples']} сэмплов сохранено.\n\n"
                f"Открыть папку с результатом?", parent=self):
            self._open_folder(os.path.dirname(summary["csv_path"]))

    def _open_folder(self, path: str) -> None:
        try:
            if os.name == "nt":
                os.startfile(path)  # noqa: on Windows only
            else:
                import subprocess
                subprocess.Popen(["xdg-open", path])
        except Exception as e:
            log.warning(f"Не удалось открыть папку {path}: {e}")

    # ── Вкладка «ML-суррогат» ────────────────────────────────────────────────

    def _build_surrogate_tab(self) -> None:
        frame = self._tab_surrogate

        if not _ML_AVAILABLE:
            tk.Label(
                frame, bg=BG, fg=FG, font=FONT, justify="left", anchor="nw",
                padx=12, pady=12,
                text=("Обучение суррогата недоступно: не найден модуль "
                     "ml_surrogate.py или его зависимости "
                     "(scikit-learn, joblib).\n\n"
                     f"Подробности: {_ML_IMPORT_ERROR}\n\n"
                     "Установите:  pip install scikit-learn joblib")
            ).pack(fill="both", expand=True)
            return

        self._surrogate_state = {"running": False}

        frame.columnconfigure(0, weight=1)

        box = tk.Frame(frame, bg=BG, bd=1, relief="groove")
        box.grid(row=0, column=0, sticky="new", padx=4, pady=4)
        box.columnconfigure(1, weight=1)

        tk.Label(box, text="Обучение forward-суррогата (предсказание RC(f))",
                 bg=BG_DARK, fg=FG, font=FONT_B, anchor="w", padx=6, pady=3
                 ).grid(row=0, column=0, columnspan=3, sticky="ew")

        pad = dict(padx=8, pady=4)

        tk.Label(box, text="Папка датасета (после «Экспорт датасета»):",
                 bg=BG, fg=FG, font=FONT_SM
                 ).grid(row=1, column=0, sticky="w", **pad)
        default_dataset_dir = os.path.join(
            os.path.dirname(self._db.db_path) or ".", "ml_dataset")
        self._var_sur_dataset_dir = tk.StringVar(value=default_dataset_dir)
        ttk.Entry(box, textvariable=self._var_sur_dataset_dir
                 ).grid(row=1, column=1, sticky="ew", **pad)
        ttk.Button(box, text="Обзор…", command=self._browse_surrogate_dataset_dir
                  ).grid(row=1, column=2, **pad)

        tk.Label(box, text="Папка для сохранения модели:", bg=BG, fg=FG, font=FONT_SM
                 ).grid(row=2, column=0, sticky="w", **pad)
        default_model_dir = os.path.join(
            os.path.dirname(self._db.db_path) or ".", "surrogate_model")
        self._var_sur_model_dir = tk.StringVar(value=default_model_dir)
        ttk.Entry(box, textvariable=self._var_sur_model_dir
                 ).grid(row=2, column=1, sticky="ew", **pad)
        ttk.Button(box, text="Обзор…", command=self._browse_surrogate_model_dir
                  ).grid(row=2, column=2, **pad)

        tk.Label(box, text="Число деревьев (n_estimators):", bg=BG, fg=FG, font=FONT_SM
                 ).grid(row=3, column=0, sticky="w", **pad)
        self._var_sur_n_estimators = tk.IntVar(value=200)
        ttk.Spinbox(box, from_=10, to=2000, increment=10,
                   textvariable=self._var_sur_n_estimators, width=8
                   ).grid(row=3, column=1, sticky="w", **pad)

        tk.Label(box, text="Доля тестовой выборки:", bg=BG, fg=FG, font=FONT_SM
                 ).grid(row=4, column=0, sticky="w", **pad)
        self._var_sur_test_size = tk.DoubleVar(value=0.2)
        ttk.Spinbox(box, from_=0.05, to=0.5, increment=0.05,
                   textvariable=self._var_sur_test_size, width=8
                   ).grid(row=4, column=1, sticky="w", **pad)

        self._btn_train_surrogate = ttk.Button(
            box, text="Обучить суррогат-модель", command=self._start_surrogate_train)
        self._btn_train_surrogate.grid(row=5, column=0, columnspan=3, sticky="ew",
                                       padx=8, pady=(6, 2), ipady=4)

        self._pb_surrogate = ttk.Progressbar(box, mode="indeterminate")
        self._pb_surrogate.grid(row=6, column=0, columnspan=3, sticky="ew",
                                padx=8, pady=(0, 4))

        # ── Результаты обучения ──────────────────────────────────────────────
        res_box = tk.Frame(frame, bg=BG, bd=1, relief="groove")
        res_box.grid(row=1, column=0, sticky="new", padx=4, pady=(0, 4))
        res_box.columnconfigure(0, weight=1)

        tk.Label(res_box, text="Результат", bg=BG_DARK, fg=FG, font=FONT_B,
                 anchor="w", padx=6, pady=3).grid(row=0, column=0, sticky="ew")

        self._var_sur_status = tk.StringVar(
            value="Модель ещё не обучалась в этой сессии.")
        tk.Label(res_box, textvariable=self._var_sur_status, bg=BG, fg=FG_DIM,
                 font=FONT_SM, anchor="w", justify="left", wraplength=560,
                 padx=8, pady=8).grid(row=1, column=0, sticky="ew")

    def _browse_surrogate_dataset_dir(self) -> None:
        d = filedialog.askdirectory(
            initialdir=self._var_sur_dataset_dir.get() or os.getcwd(), parent=self)
        if d:
            self._var_sur_dataset_dir.set(d)

    def _browse_surrogate_model_dir(self) -> None:
        d = filedialog.askdirectory(
            initialdir=self._var_sur_model_dir.get() or os.getcwd(), parent=self)
        if d:
            self._var_sur_model_dir.set(d)

    def _start_surrogate_train(self) -> None:
        if self._surrogate_state.get("running"):
            return

        dataset_dir = self._var_sur_dataset_dir.get().strip()
        model_dir = self._var_sur_model_dir.get().strip()
        if not dataset_dir or not os.path.isdir(dataset_dir):
            messagebox.showerror(
                "Обучение суррогата",
                f"Папка датасета не найдена:\n{dataset_dir}\n\n"
                f"Сначала сформируйте датасет во вкладке «Экспорт датасета».",
                parent=self)
            return
        if not model_dir:
            messagebox.showerror("Обучение суррогата",
                                 "Укажите папку для сохранения модели.", parent=self)
            return

        self._surrogate_state = {"running": True, "error": None, "metrics": None}
        self._btn_train_surrogate.configure(state="disabled")
        self._pb_surrogate.start(12)
        self._var_sur_status.set("Обучение запущено — это может занять некоторое "
                                 "время в зависимости от размера датасета…")

        t = threading.Thread(
            target=self._surrogate_train_worker,
            args=(dataset_dir, model_dir, self._var_sur_n_estimators.get(),
                 self._var_sur_test_size.get()),
            daemon=True)
        t.start()
        self.after(150, self._poll_surrogate_train)

    def _surrogate_train_worker(self, dataset_dir, model_dir,
                                n_estimators, test_size) -> None:
        try:
            dataset = load_ml_dataset(dataset_dir)
            model, metrics = SurrogateModel.train(
                dataset, n_estimators=n_estimators, test_size=test_size)
            model.save(model_dir)
            metrics["model_dir"] = model_dir
            self._surrogate_state["metrics"] = metrics
        except Exception as e:
            log.exception("Ошибка при обучении суррогата")
            self._surrogate_state["error"] = str(e)
        finally:
            self._surrogate_state["running"] = False

    def _poll_surrogate_train(self) -> None:
        st = self._surrogate_state
        if st.get("running"):
            self.after(150, self._poll_surrogate_train)
            return

        self._pb_surrogate.stop()
        self._btn_train_surrogate.configure(state="normal")

        if st.get("error"):
            self._var_sur_status.set(f"Ошибка: {st['error']}")
            messagebox.showerror("Обучение суррогата", st["error"], parent=self)
            return

        m = st.get("metrics")
        if not m:
            return
        warn = ""
        if m["n_samples"] < 100:
            warn = ("\n⚠ Датасет маленький (< 100 сэмплов) — метрики ниже "
                   "показательны только как проверка пайплайна, не как "
                   "оценка реального качества модели.")
        self._var_sur_status.set(
            f"Готово. Сэмплов: {m['n_samples']} (train {m['n_train']} / "
            f"test {m['n_test']})\n"
            f"MAE по кривой RC(f): {m['mae_curve_db_mean']} дБ   |   "
            f"R² по кривой: {m['r2_curve']}\n"
            f"MAE по max RC: {m['mae_max_rc_db']} дБ\n"
            f"Модель сохранена: {m['model_dir']}{warn}"
        )

    # ── Вкладка «История целей» (Inverse Design) ─────────────────────────────

    def _build_targets_tab(self) -> None:
        frame = self._tab_targets
        frame.columnconfigure(0, weight=0)
        frame.columnconfigure(1, weight=1)
        frame.rowconfigure(0, weight=1)

        self._current_fig_target = None
        self._mpl_canvas_target = None

        # Левая колонка — список запросов
        left = tk.Frame(frame, bg=BG, bd=1, relief="groove")
        left.grid(row=0, column=0, sticky="ns", padx=(4, 6), pady=4)
        left.rowconfigure(1, weight=1)

        tk.Label(left, text="Запросы Inverse Design", bg=BG_DARK, fg=FG,
                 font=FONT_B, anchor="w", padx=6, pady=3
                 ).grid(row=0, column=0, sticky="ew")

        cols = [("id", "ID", 40), ("created", "Создан", 130),
                ("band", "Полоса (ГГц)", 110), ("rc", "RC цель", 60),
                ("th", "Толщ. (мм)", 75), ("result", "Результат", 90)]
        tree = ttk.Treeview(left, columns=[c[0] for c in cols], show="headings",
                            height=22)
        for cid, cname, cw in cols:
            tree.heading(cid, text=cname)
            tree.column(cid, width=cw, anchor="center", stretch=False)
        vsb = ttk.Scrollbar(left, orient="vertical", command=tree.yview)
        tree.configure(yscrollcommand=vsb.set)
        tree.grid(row=1, column=0, sticky="nsew", padx=(2, 0), pady=2)
        vsb.grid(row=1, column=1, sticky="ns", pady=2)
        tree.bind("<<TreeviewSelect>>", self._on_target_select)
        self._tree_targets = tree

        # Правая колонка — график найденного покрытия + детали цели
        right = tk.Frame(frame, bg=BG, bd=2, relief="sunken")
        right.grid(row=0, column=1, sticky="nsew", pady=4)
        right.rowconfigure(0, weight=1)
        right.rowconfigure(1, weight=0)
        right.columnconfigure(0, weight=1)

        self._canvas_frame_target = tk.Frame(right, bg=BG)
        self._canvas_frame_target.grid(row=0, column=0, sticky="nsew")
        self._canvas_frame_target.rowconfigure(0, weight=1)
        self._canvas_frame_target.columnconfigure(0, weight=1)

        self._placeholder_target = tk.Label(
            self._canvas_frame_target, text="← Выберите запрос в списке",
            bg=BG, fg=FG_DIM, font=FONT)
        self._placeholder_target.grid(row=0, column=0, sticky="nsew")

        self._var_detail_target = tk.StringVar(value="")
        tk.Label(right, textvariable=self._var_detail_target, bg=BG_DARK,
                 fg=FG_DIM, font=FONT_SM, anchor="w", justify="left",
                 padx=6, pady=3).grid(row=1, column=0, sticky="ew")

    def _refresh_targets(self) -> None:
        tree = self._tree_targets
        for row in tree.get_children():
            tree.delete(row)
        for t in self._db.list_design_targets():
            result_text = (f"#{t['resulting_coating_id']}"
                          if t.get("resulting_coating_id") else "не найдено")
            tree.insert("", "end", iid=str(t["target_id"]), values=(
                t["target_id"], t["created_at"][:16],
                f"{t['f_lo_hz']/1e9:.2f}–{t['f_hi_hz']/1e9:.2f}",
                f"{t['rc_target_db']:+.1f}",
                f"{t['max_thickness_mm']:.1f}" if t.get("max_thickness_mm") else "—",
                result_text,
            ))

    def _on_target_select(self, event) -> None:
        sel = self._tree_targets.selection()
        if not sel:
            return
        target = self._db.get_design_target(int(sel[0]))
        if target is None:
            return
        self._show_target(target)

    def _show_target(self, target: dict) -> None:
        if self._mpl_canvas_target is not None:
            self._mpl_canvas_target.get_tk_widget().destroy()
            self._mpl_canvas_target = None
        if self._current_fig_target is not None:
            plt.close(self._current_fig_target)
            self._current_fig_target = None

        band_str = f"{target['f_lo_hz']/1e9:.2f}–{target['f_hi_hz']/1e9:.2f} ГГц"
        th_str = (f"{target['max_thickness_mm']:.1f} мм"
                 if target.get("max_thickness_mm") else "не ограничена")

        coating_id = target.get("resulting_coating_id")
        if not coating_id:
            self._placeholder_target.config(
                text=f"Для этой цели ({band_str}, RC≤{target['rc_target_db']:+.1f} дБ) "
                    f"подходящее покрытие не найдено.")
            self._placeholder_target.grid()
            self._var_detail_target.set(
                f"Цель #{target['target_id']}: {band_str}, "
                f"RC≤{target['rc_target_db']:+.1f} дБ, макс. толщина {th_str} — "
                f"результата нет")
            return

        coating = self._db.get_coating(coating_id)
        if coating is None:
            self._placeholder_target.config(
                text="Связанное покрытие было удалено из БД.")
            self._placeholder_target.grid()
            self._var_detail_target.set("")
            return

        try:
            cfg, freqs, mat_arrays, names = build_plot_context(self._db, coating)
        except ValueError as e:
            self._placeholder_target.config(text=f"Не удалось построить график:\n{e}")
            self._placeholder_target.grid()
            self._var_detail_target.set("")
            return

        self._placeholder_target.grid_remove()
        # График строится на полном измеренном диапазоне материалов (как
        # сохранено в БД), а не только на целевой полосе — см. пояснение в
        # detail-строке ниже: honest max_rc относится именно к целевой полосе.
        fig = make_config_figure(cfg, freqs, mat_arrays, names,
                                 rc_threshold=target["rc_target_db"])
        self._current_fig_target = fig
        canvas = FigureCanvasTkAgg(fig, master=self._canvas_frame_target)
        canvas.draw()
        canvas.get_tk_widget().grid(row=0, column=0, sticky="nsew")
        self._mpl_canvas_target = canvas

        self._var_detail_target.set(
            f"Цель #{target['target_id']}: полоса {band_str}, "
            f"RC≤{target['rc_target_db']:+.1f} дБ, макс. толщина {th_str}  |  "
            f"Покрытие #{coating_id} (прогон #{coating['run_id']}): "
            f"max RC в целевой полосе = {coating['max_rc_db']:+.1f} дБ  |  "
            f"Материалы: {', '.join(names)}  |  "
            f"График показывает ПОЛНЫЙ измеренный диапазон материалов, "
            f"не только целевую полосу."
        )

    # ── Обновление всего окна ────────────────────────────────────────────────

    def refresh(self) -> None:
        st = self._db.stats()
        if st["n_coatings"] == 0:
            text = (f"Материалов: {st['n_materials']}   |   "
                    f"Прогонов: {st['n_runs']}   |   Покрытий: 0   |   "
                    f"Файл БД: {st['db_path']}")
        else:
            best_rc = f"{st['best_rc_db']:+.1f} дБ" if st["best_rc_db"] is not None else "—"
            thinnest = (f"{st['thinnest_valid_mm']:.1f} мм"
                       if st["thinnest_valid_mm"] is not None else "—")
            text = (
                f"Материалов: {st['n_materials']}   |   "
                f"Прогонов: {st['n_runs']}   |   "
                f"Покрытий: {st['n_coatings']} (валидных: {st['n_valid_coatings']})   |   "
                f"Лучший RC: {best_rc}   |   "
                f"Мин. толщина (валидные): {thinnest}   |   "
                f"Файл БД: {st['db_path']}  ({st['db_size_mb']} МБ)"
            )
        self._var_stats.set(text)
        self._refresh_materials()
        self._refresh_runs()
        self._refresh_filter_options()
        self._refresh_coatings()
        self._refresh_export_runs()
        self._refresh_targets()

    def _on_close(self) -> None:
        if self._current_fig is not None:
            plt.close(self._current_fig)
        if self._current_fig_target is not None:
            plt.close(self._current_fig_target)
        self._db.close()
        self.destroy()
