"""
preview.py — окно предпросмотра результатов оптимизации

Показывает все конфигурации, прошедшие порог RC, в прокручиваемом списке.
При выборе конфигурации отображает:
  • график RC(f) с подсветкой полосы
  • стек слоёв (AIR → METAL)
  • гистограмму чувствительности
Дополнительно: итоговая таблица с ключевыми параметрами всех конфигураций.
"""

import io
import tkinter as tk
from tkinter import ttk
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg

from plotting import make_config_figure


# ─────────────────────────────────────────────────────────────────────────────
# Цветовая схема (повторяет стиль главного окна)
# ─────────────────────────────────────────────────────────────────────────────
BG       = "#ece9d8"
BG_DARK  = "#d4d0c8"
BTN_FACE = "#ece9d8"
TITLE_BG = "#0a246a"
TITLE_FG = "#ffffff"
FG       = "#000000"
FG_DIM   = "#444444"
SEL_BG   = "#316ac5"
FONT     = ("Tahoma", 9)
FONT_B   = ("Tahoma", 9, "bold")
FONT_SM  = ("Tahoma", 8)
MONO     = ("Courier New", 9)


class PreviewWindow(tk.Toplevel):
    """
    Окно предпросмотра всех вычисленных конфигураций.

    Параметры
    ---------
    parent          — родительское окно (главный tk.Tk)
    configs         — список словарей конфигураций (уже отранжированных)
    freqs           — numpy-массив частот
    mat_arrays      — словарь с предвычисленными массивами материалов
    material_names  — список имён материалов
    rc_threshold    — порог для выделения валидных конфигураций (дБ)
    """

    def __init__(self, parent, configs, freqs, mat_arrays,
                 material_names, rc_threshold=-10.0):
        super().__init__(parent)
        self.title("Предпросмотр результатов")
        self.configure(bg=BG)
        self.resizable(True, True)
        self.geometry("1280x760")
        self.minsize(900, 560)

        self._configs        = configs
        self._freqs          = freqs
        self._mat_arrays     = mat_arrays
        self._material_names = material_names
        self._rc_threshold   = rc_threshold
        self._current_fig    = None   # текущая Figure matplotlib

        self._build_ui()
        # Выбрать первую конфигурацию сразу
        if self._configs:
            self._listbox.selection_set(0)
            self._on_select(None)

    # ─────────────────────────────────────────────────────────────────────────
    # Построение UI
    # ─────────────────────────────────────────────────────────────────────────

    def _build_ui(self) -> None:
        self.columnconfigure(0, weight=1)
        self.rowconfigure(0, weight=1)

        # ── Псевдо-заголовок в стиле XP ──────────────────────────────────────
        title_bar = tk.Frame(self, bg=TITLE_BG, height=26)
        title_bar.pack(side="top", fill="x")
        tk.Label(
            title_bar,
            text="  Предпросмотр результатов — RC Absorber Optimiser",
            bg=TITLE_BG, fg=TITLE_FG, font=FONT_B, anchor="w"
        ).pack(side="left", padx=4, pady=3)

        # ── Основной контейнер ────────────────────────────────────────────────
        main = ttk.Frame(self, style="TFrame", padding=6)
        main.pack(fill="both", expand=True)
        main.columnconfigure(0, weight=0)   # колонка списка — фиксированная
        main.columnconfigure(1, weight=1)   # колонка графика — растягивается
        main.rowconfigure(0, weight=1)

        # ── Левая панель: список конфигураций ─────────────────────────────────
        left_frame = tk.Frame(main, bg=BG, bd=1, relief="groove")
        left_frame.grid(row=0, column=0, sticky="ns", padx=(0, 6))
        left_frame.rowconfigure(1, weight=1)

        tk.Label(left_frame,
                 text="Конфигурации", bg=BG_DARK, fg=FG,
                 font=FONT_B, anchor="w", padx=6, pady=3
                 ).grid(row=0, column=0, columnspan=2, sticky="ew")

        lb_scroll = tk.Scrollbar(left_frame, orient="vertical", bg=BG_DARK)
        self._listbox = tk.Listbox(
            left_frame,
            yscrollcommand=lb_scroll.set,
            selectbackground=SEL_BG, selectforeground="#fff",
            bg="#fff", fg=FG, font=FONT_SM,
            activestyle="dotbox",
            width=36, relief="flat", bd=0,
        )
        lb_scroll.config(command=self._listbox.yview)
        self._listbox.grid(row=1, column=0, sticky="nsew", padx=(4, 0), pady=4)
        lb_scroll.grid(row=1, column=1, sticky="ns", pady=4)
        left_frame.columnconfigure(0, weight=1)

        self._listbox.bind("<<ListboxSelect>>", self._on_select)
        self._populate_list()

        # ── Правая панель ─────────────────────────────────────────────────────
        right_frame = ttk.Frame(main, style="TFrame")
        right_frame.grid(row=0, column=1, sticky="nsew")
        right_frame.rowconfigure(0, weight=1)
        right_frame.rowconfigure(1, weight=0)
        right_frame.columnconfigure(0, weight=1)

        # Canvas для matplotlib-графика
        self._canvas_frame = tk.Frame(right_frame, bg=BG, bd=2, relief="sunken")
        self._canvas_frame.grid(row=0, column=0, sticky="nsew")
        self._canvas_frame.rowconfigure(0, weight=1)
        self._canvas_frame.columnconfigure(0, weight=1)

        # Заглушка до выбора конфигурации
        self._placeholder = tk.Label(
            self._canvas_frame,
            text="← Выберите конфигурацию в списке",
            bg=BG, fg=FG_DIM, font=FONT
        )
        self._placeholder.grid(row=0, column=0, sticky="nsew")

        self._mpl_canvas: FigureCanvasTkAgg | None = None

        # ── Нижняя панель: сводная таблица ────────────────────────────────────
        tbl_frame = tk.Frame(right_frame, bg=BG_DARK, bd=1, relief="groove")
        tbl_frame.grid(row=1, column=0, sticky="ew", pady=(6, 0))
        tbl_frame.columnconfigure(0, weight=1)

        tk.Label(tbl_frame, text="Сводная таблица",
                 bg=BG_DARK, fg=FG, font=FONT_B,
                 anchor="w", padx=6, pady=2
                 ).grid(row=0, column=0, sticky="ew")

        self._table = ttk.Treeview(
            tbl_frame,
            columns=("rank", "layers", "max_rc", "total_mm", "bw", "robust"),
            show="headings", height=5,
        )
        _cols = [
            ("rank",     "Ранг",         50),
            ("layers",   "Слоёв",        50),
            ("max_rc",   "Max RC (дБ)",  90),
            ("total_mm", "Толщина (мм)", 90),
            ("bw",       "Полоса (ГГц)", 95),
            ("robust",   "Робастных",    75),
        ]
        for cid, cname, cw in _cols:
            self._table.heading(cid, text=cname)
            self._table.column(cid, width=cw, anchor="center", stretch=False)
        tbl_scroll = tk.Scrollbar(tbl_frame, orient="vertical",
                                  command=self._table.yview)
        self._table.configure(yscrollcommand=tbl_scroll.set)
        self._table.grid(row=1, column=0, sticky="ew", padx=(4, 0))
        tbl_scroll.grid(row=1, column=1, sticky="ns")
        tbl_frame.columnconfigure(0, weight=1)

        # Клик по строке таблицы → выбрать ту же конфигурацию в списке
        self._table.bind("<<TreeviewSelect>>", self._on_table_select)

        self._populate_table()

        # ── Статус-бар ────────────────────────────────────────────────────────
        sb = tk.Frame(self, bg=BG_DARK, relief="sunken", bd=1)
        sb.pack(side="bottom", fill="x")
        self._status_var = tk.StringVar(value="")
        tk.Label(sb, textvariable=self._status_var, bg=BG_DARK, fg=FG_DIM,
                 font=FONT_SM, anchor="w", padx=6, pady=2).pack(side="left")

    # ─────────────────────────────────────────────────────────────────────────
    # Заполнение списка
    # ─────────────────────────────────────────────────────────────────────────

    def _populate_list(self) -> None:
        self._listbox.delete(0, "end")
        for cfg in self._configs:
            rank   = cfg.get('rank', '—')
            n      = cfg['n_layers']
            rc     = cfg['max_rc']
            th     = cfg['total_mm']
            bw     = cfg.get('bandwidth_ghz', 0)
            valid  = rc <= self._rc_threshold
            prefix = "✓" if valid else "✗"
            label  = (f"{prefix} #{rank}  {n}сл  "
                      f"RC={rc:+.1f}дБ  {th:.1f}мм  {bw:.2f}ГГц")
            self._listbox.insert("end", label)
            # Цветовая подсветка: зелёный для валидных, красный — нет
            color = "#d6f0d6" if valid else "#fde8e8"
            self._listbox.itemconfig("end", bg=color)

    # ─────────────────────────────────────────────────────────────────────────
    # Заполнение таблицы
    # ─────────────────────────────────────────────────────────────────────────

    def _populate_table(self) -> None:
        for row in self._table.get_children():
            self._table.delete(row)
        for i, cfg in enumerate(self._configs):
            rank   = cfg.get('rank', '—')
            n      = cfg['n_layers']
            rc     = cfg['max_rc']
            th     = cfg['total_mm']
            bw     = cfg.get('bandwidth_ghz', 0)
            sens   = cfg.get('sensitivity', {})
            robust = sens.get('robust_score', '—') if sens else '—'
            tag    = "valid" if rc <= self._rc_threshold else "invalid"
            self._table.insert("", "end", iid=str(i),
                               values=(rank, n,
                                       f"{rc:+.2f}", f"{th:.1f}",
                                       f"{bw:.3f}", robust),
                               tags=(tag,))
        self._table.tag_configure("valid",   background="#d6f0d6")
        self._table.tag_configure("invalid", background="#fde8e8")

    # ─────────────────────────────────────────────────────────────────────────
    # Обработчики событий
    # ─────────────────────────────────────────────────────────────────────────

    def _on_select(self, event) -> None:
        sel = self._listbox.curselection()
        if not sel:
            return
        idx = sel[0]
        cfg = self._configs[idx]
        self._show_config(cfg, idx)
        # Синхронизировать с таблицей
        self._table.selection_set(str(idx))
        self._table.see(str(idx))

    def _on_table_select(self, event) -> None:
        sel = self._table.selection()
        if not sel:
            return
        idx = int(sel[0])
        self._listbox.selection_clear(0, "end")
        self._listbox.selection_set(idx)
        self._listbox.see(idx)
        self._show_config(self._configs[idx], idx)

    def _show_config(self, cfg: dict, idx: int) -> None:
        """Отрендерить график для выбранной конфигурации."""
        # Закрыть предыдущую Figure, чтобы не копить память
        if self._mpl_canvas is not None:
            self._mpl_canvas.get_tk_widget().destroy()
            self._mpl_canvas = None
        if self._current_fig is not None:
            plt.close(self._current_fig)
            self._current_fig = None

        self._placeholder.grid_remove()

        # Построить новую Figure
        fig = make_config_figure(
            cfg, self._freqs, self._mat_arrays, self._material_names,
            rc_threshold=self._rc_threshold
        )
        self._current_fig = fig

        canvas = FigureCanvasTkAgg(fig, master=self._canvas_frame)
        canvas.draw()
        widget = canvas.get_tk_widget()
        widget.grid(row=0, column=0, sticky="nsew")
        self._mpl_canvas = canvas

        rank = cfg.get('rank', '—')
        self._status_var.set(
            f"Конфигурация #{idx+1}  |  Ранг {rank}  |  "
            f"{cfg['n_layers']} сл.  |  "
            f"RC={cfg['max_rc']:+.2f} дБ  |  "
            f"{cfg['total_mm']:.1f} мм  |  "
            f"BW={cfg.get('bandwidth_ghz',0):.3f} ГГц"
        )

    # ─────────────────────────────────────────────────────────────────────────
    # Очистка ресурсов при закрытии
    # ─────────────────────────────────────────────────────────────────────────

    def destroy(self) -> None:
        if self._current_fig is not None:
            plt.close(self._current_fig)
        super().destroy()
