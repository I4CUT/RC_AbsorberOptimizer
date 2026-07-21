"""
gui.py — главное окно приложения

Классический стиль Windows XP (серо-бежевая палитра).
Содержит:
  • Панель настроек (папки, режим, число слоёв)
  • Прогресс-бар и журнал выполнения
  • Кнопки «Запустить» / «Отмена» / «Предпросмотр»
"""

import os
import queue
import threading
import tkinter as tk
from tkinter import ttk, filedialog, messagebox

import worker
from logger import log_queue
from preview import PreviewWindow
from db_viewer import DatabaseWindow
from physics import (
    RC_THRESHOLD, LAMBDA_MULTIPLIERS, LAMBDA_MULTIPLIER_DEFAULT,
    LAMBDA_MULTIPLIER_LABELS, validate_rc_threshold,
)

# ─────────────────────────────────────────────────────────────────────────────
# Цветовая схема XP
# ─────────────────────────────────────────────────────────────────────────────
BG       = "#ece9d8"
BG_DARK  = "#d4d0c8"
BTN_FACE = "#ece9d8"
TITLE_BG = "#0a246a"
TITLE_FG = "#ffffff"
FG       = "#000000"
FG_DIM   = "#444444"
ENTRY_BG = "#ffffff"
LOG_BG   = "#ffffff"
LOG_FG   = "#000000"
SEL_BG   = "#316ac5"
MONO     = "Courier New"
FONT     = ("Tahoma", 9)
FONT_B   = ("Tahoma", 9, "bold")
FONT_SM  = ("Tahoma", 8)


class App:
    """Главное окно приложения."""

    def __init__(self, root: tk.Tk) -> None:
        self._root            = root
        self._preview_win     = None   # ссылка на PreviewWindow (если открыто)
        self._db_viewer_win   = None   # ссылка на DatabaseWindow (если открыто)
        self._last_results    = None   # кортеж (configs, freqs, mat_arrays, names)

        root.title("RC Absorber Optimiser")
        root.resizable(True, True)
        root.configure(bg=BG)
        root.columnconfigure(0, weight=1)
        root.rowconfigure(1, weight=1)

        self._apply_styles()
        self._build_ui()
        root.update_idletasks()
        root.minsize(820, 500)
        root.geometry("1020x620")

    # ─────────────────────────────────────────────────────────────────────────
    # ttk-стили
    # ─────────────────────────────────────────────────────────────────────────

    def _apply_styles(self) -> None:
        s = ttk.Style()
        s.theme_use("classic")
        s.configure("TFrame",       background=BG)
        s.configure("Dark.TFrame",  background=BG_DARK)
        s.configure("TLabel",       background=BG,      foreground=FG, font=FONT)
        s.configure("Dark.TLabel",  background=BG_DARK, foreground=FG, font=FONT)
        s.configure("Bold.TLabel",  background=BG,      foreground=FG, font=FONT_B)
        s.configure("DBold.TLabel", background=BG_DARK, foreground=FG, font=FONT_B)
        s.configure("TButton", font=FONT, relief="raised",
                    background=BTN_FACE, foreground=FG, padding=(6, 3))
        s.map("TButton",
              relief=[("pressed", "sunken"), ("active", "raised")],
              background=[("active", "#c8c4bc")])
        s.configure("TRadiobutton", background=BG_DARK, foreground=FG, font=FONT)
        s.map("TRadiobutton", background=[("active", BG_DARK)])
        s.configure("TCheckbutton", background=BG, foreground=FG, font=FONT)
        s.configure("Horizontal.TProgressbar",
                    troughcolor="#b0b0a8", background=SEL_BG,
                    bordercolor=BG_DARK, darkcolor=SEL_BG, lightcolor="#6b9de8",
                    thickness=16)

    # ─────────────────────────────────────────────────────────────────────────
    # Построение интерфейса
    # ─────────────────────────────────────────────────────────────────────────

    def _build_ui(self) -> None:
        root = self._root

        # ── Псевдо-заголовок ─────────────────────────────────────────────────
        title_bar = tk.Frame(root, bg=TITLE_BG, height=28)
        title_bar.grid(row=0, column=0, sticky="ew")
        title_bar.columnconfigure(0, weight=1)
        tk.Label(
            title_bar,
            text="  RC Absorber Optimiser  |  TMM + NSGA-II + DE",
            bg=TITLE_BG, fg=TITLE_FG, font=FONT_B, anchor="w"
        ).grid(row=0, column=0, sticky="ew", padx=4, pady=4)

        # ── Тело ─────────────────────────────────────────────────────────────
        body = ttk.Frame(root, style="TFrame", padding=8)
        body.grid(row=1, column=0, sticky="nsew")
        body.columnconfigure(0, weight=0)
        body.columnconfigure(1, weight=1)
        body.rowconfigure(0, weight=1)

        def groupbox(parent, title, row, col, rowspan=1, colspan=1,
                     padx=4, pady=4, sticky="nsew"):
            outer = ttk.LabelFrame(parent, text=title,
                                   style="Dark.TFrame",
                                   labelanchor="nw",
                                   padding=(8, 4, 8, 8))
            outer.configure(relief="groove")
            outer.grid(row=row, column=col, rowspan=rowspan, columnspan=colspan,
                       sticky=sticky, padx=padx, pady=pady)
            outer.columnconfigure(0, weight=1)
            return outer

        # ══════════════════════════════════════════════════════════════════════
        # Левая колонка — настройки (в прокручиваемом контейнере: групп много,
        # и на невысоких экранах/маленьких окнах низ панели иначе обрезается)
        # ══════════════════════════════════════════════════════════════════════
        left_container = ttk.Frame(body, style="TFrame")
        left_container.grid(row=0, column=0, sticky="ns", padx=(0, 6))
        left_container.rowconfigure(0, weight=1)

        left_canvas = tk.Canvas(left_container, bg=BG, highlightthickness=0,
                                width=372)
        left_scroll = ttk.Scrollbar(left_container, orient="vertical",
                                    command=left_canvas.yview)
        left_canvas.configure(yscrollcommand=left_scroll.set)
        left_canvas.grid(row=0, column=0, sticky="ns")
        left_scroll.grid(row=0, column=1, sticky="ns")

        left = ttk.Frame(left_canvas, style="TFrame")
        left_window = left_canvas.create_window((0, 0), window=left, anchor="nw")

        def _on_left_configure(event=None):
            left_canvas.configure(scrollregion=left_canvas.bbox("all"))

        def _on_canvas_configure(event):
            # Внутренняя панель растягивается по ширине канваса, но не по высоте —
            # так работает вертикальная прокрутка при переполнении.
            left_canvas.itemconfigure(left_window, width=event.width)

        left.bind("<Configure>", _on_left_configure)
        left_canvas.bind("<Configure>", _on_canvas_configure)

        def _on_mousewheel(event):
            left_canvas.yview_scroll(int(-1 * (event.delta / 120)), "units")

        # Прокрутка колесом мыши работает, только когда курсор над панелью —
        # иначе он перехватывал бы прокрутку у журнала выполнения справа.
        left_canvas.bind("<Enter>", lambda e: left_canvas.bind_all("<MouseWheel>", _on_mousewheel))
        left_canvas.bind("<Leave>", lambda e: left_canvas.unbind_all("<MouseWheel>"))

        # ── Папки ─────────────────────────────────────────────────────────────
        grp_folders = groupbox(left, "Папки", row=0, col=0, sticky="ew")

        self._var_folder = tk.StringVar()
        self._var_outdir = tk.StringVar(value=os.path.expanduser("~"))

        ttk.Label(grp_folders, text="Материалы (.xlsx):",
                  style="Dark.TLabel").grid(row=0, column=0, columnspan=2,
                                            sticky="w", pady=(2, 0))
        tk.Entry(grp_folders, textvariable=self._var_folder, width=34,
                 bg=ENTRY_BG, fg=FG, font=FONT, relief="sunken", bd=2
                 ).grid(row=1, column=0, sticky="ew", padx=(0, 4), pady=(2, 6))
        ttk.Button(grp_folders, text="Обзор…",
                   command=self._browse_folder
                   ).grid(row=1, column=1, pady=(2, 6))

        ttk.Label(grp_folders, text="Сохранить результаты в:",
                  style="Dark.TLabel").grid(row=2, column=0, columnspan=2,
                                            sticky="w", pady=(0, 0))
        tk.Entry(grp_folders, textvariable=self._var_outdir, width=34,
                 bg=ENTRY_BG, fg=FG, font=FONT, relief="sunken", bd=2
                 ).grid(row=3, column=0, sticky="ew", padx=(0, 4), pady=(2, 2))
        ttk.Button(grp_folders, text="Обзор…",
                   command=self._browse_outdir
                   ).grid(row=3, column=1, pady=(2, 2))

        # ── Режим ─────────────────────────────────────────────────────────────
        grp_mode = groupbox(left, "Режим оптимизации", row=1, col=0, sticky="ew")
        self._var_mode = tk.StringVar(value="3")
        modes = [
            ("Лучший RC      (w_rc=1.0, w_th=0.0)", "1"),
            ("Тонкий          (w_rc=0.1, w_th=0.9)", "2"),
            ("Компромисс   (w_rc=1.0, w_th=0.3)", "3"),
        ]
        for i, (txt, val) in enumerate(modes):
            ttk.Radiobutton(grp_mode, text=txt,
                            variable=self._var_mode, value=val,
                            style="TRadiobutton"
                            ).grid(row=i, column=0, sticky="w", pady=2)

        # ── Параметры ─────────────────────────────────────────────────────────
        grp_layers = groupbox(left, "Параметры", row=2, col=0, sticky="ew")
        grp_layers.columnconfigure(1, weight=1)
        self._var_layers = tk.IntVar(value=3)
        ttk.Label(grp_layers, text="Макс. число слоёв:",
                  style="Dark.TLabel").grid(row=0, column=0, sticky="w", padx=(0, 8))
        tk.Spinbox(grp_layers, from_=1, to=10,
                   textvariable=self._var_layers, width=5,
                   bg=ENTRY_BG, fg=FG, font=FONT,
                   relief="sunken", bd=2, buttonbackground=BG_DARK
                   ).grid(row=0, column=1, sticky="w", pady=2)

        # ── Порог RC и ограничение толщины ──────────────────────────────────────
        grp_thresh = groupbox(left, "Порог RC и толщина покрытия",
                              row=3, col=0, sticky="ew")
        grp_thresh.columnconfigure(1, weight=1)

        # Числовое поле порога RC (дБ), валидация: должно быть < 0
        ttk.Label(grp_thresh, text="Порог RC (дБ):",
                  style="Dark.TLabel").grid(row=0, column=0, sticky="w", padx=(0, 8))
        self._var_rc_threshold = tk.StringVar(value=f"{RC_THRESHOLD:.1f}")
        self._entry_rc_threshold = tk.Entry(
            grp_thresh, textvariable=self._var_rc_threshold, width=8,
            bg=ENTRY_BG, fg=FG, font=FONT, relief="sunken", bd=2)
        self._entry_rc_threshold.grid(row=0, column=1, sticky="w", pady=2)
        # Проверка значения при потере фокуса и при нажатии Enter
        self._entry_rc_threshold.bind("<FocusOut>", self._on_rc_threshold_change)
        self._entry_rc_threshold.bind("<Return>",   self._on_rc_threshold_change)

        self._lbl_rc_hint = ttk.Label(
            grp_thresh, text="(должно быть < 0)",
            style="Dark.TLabel", foreground=FG_DIM)
        self._lbl_rc_hint.grid(row=1, column=0, columnspan=2,
                               sticky="w", pady=(0, 6))

        # Слайдер множителя толщины λ
        self._lbl_lambda_title = ttk.Label(grp_thresh, text="Макс. толщина покрытия:",
                  style="Dark.TLabel")
        self._lbl_lambda_title.grid(row=2, column=0, columnspan=2,
                                    sticky="w", pady=(2, 0))

        self._lambda_values  = LAMBDA_MULTIPLIERS
        default_idx          = self._lambda_values.index(LAMBDA_MULTIPLIER_DEFAULT)
        self._var_lambda_idx = tk.IntVar(value=default_idx)

        self._scale_lambda = tk.Scale(
            grp_thresh, from_=0, to=len(self._lambda_values) - 1,
            orient="horizontal", variable=self._var_lambda_idx,
            showvalue=False, resolution=1,
            bg=BG_DARK, troughcolor="#b0b0a8",
            highlightthickness=0, bd=1, relief="sunken",
            command=self._on_lambda_slider_change, length=260)
        self._scale_lambda.grid(row=3, column=0, columnspan=2,
                                sticky="ew", pady=(2, 0))

        self._var_lambda_label = tk.StringVar(
            value=LAMBDA_MULTIPLIER_LABELS[LAMBDA_MULTIPLIER_DEFAULT])
        self._lbl_lambda = ttk.Label(grp_thresh, textvariable=self._var_lambda_label,
                                     style="Dark.TLabel", anchor="center")
        self._lbl_lambda.grid(row=4, column=0, columnspan=2, sticky="ew", pady=(0, 2))

        # ── Inverse Design: целевая полоса частот вместо полного диапазона ──────
        self._var_inverse_design = tk.BooleanVar(value=False)
        ttk.Checkbutton(grp_thresh, text="Целевая полоса частот (Inverse Design)",
                       variable=self._var_inverse_design,
                       command=self._on_inverse_design_toggle
                       ).grid(row=5, column=0, columnspan=2, sticky="w", pady=(6, 2))

        self._grp_inverse = ttk.Frame(grp_thresh, style="TFrame")
        self._grp_inverse.grid(row=6, column=0, columnspan=2, sticky="ew")
        self._grp_inverse.columnconfigure(1, weight=1)

        ttk.Label(self._grp_inverse, text="Целевая полоса (ГГц):",
                 style="Dark.TLabel").grid(row=0, column=0, columnspan=2,
                                           sticky="w", pady=(2, 0))
        band_row = ttk.Frame(self._grp_inverse, style="TFrame")
        band_row.grid(row=1, column=0, columnspan=2, sticky="w", pady=(0, 4))
        self._var_target_f_lo = tk.StringVar(value="")
        self._var_target_f_hi = tk.StringVar(value="")
        tk.Entry(band_row, textvariable=self._var_target_f_lo, width=7,
                bg=ENTRY_BG, fg=FG, font=FONT, relief="sunken", bd=2
                ).pack(side="left")
        ttk.Label(band_row, text=" — ", style="Dark.TLabel").pack(side="left")
        tk.Entry(band_row, textvariable=self._var_target_f_hi, width=7,
                bg=ENTRY_BG, fg=FG, font=FONT, relief="sunken", bd=2
                ).pack(side="left")

        ttk.Label(self._grp_inverse, text="Макс. толщина покрытия (мм):",
                 style="Dark.TLabel").grid(row=2, column=0, sticky="w", padx=(0, 8))
        self._var_target_max_mm = tk.StringVar(value="20.0")
        tk.Entry(self._grp_inverse, textvariable=self._var_target_max_mm, width=8,
                bg=ENTRY_BG, fg=FG, font=FONT, relief="sunken", bd=2
                ).grid(row=2, column=1, sticky="w", pady=2)

        tk.Label(self._grp_inverse, bg=BG, fg=FG_DIM, font=("Tahoma", 8),
                justify="left", anchor="w", wraplength=280,
                text="Порог RC (дБ) выше применяется как целевой уровень "
                     "внутри указанной полосы. Поиск сфокусирован именно на "
                     "этой полосе; итоговый результат честно проверяется "
                     "TMM по всему измеренному диапазону материалов."
                ).grid(row=3, column=0, columnspan=2, sticky="w", pady=(4, 0))

        self._grp_inverse.grid_remove()  # скрыто по умолчанию — режим выключен

        # ── Метод расчёта: две вкладки ────────────────────────────────────────
        calc_frame = ttk.LabelFrame(left, text="Метод расчёта")
        calc_frame.grid(row=4, column=0, sticky="ew", pady=(0, 6))
        calc_frame.columnconfigure(0, weight=1)

        self._nb_calc = ttk.Notebook(calc_frame)
        self._nb_calc.pack(fill="x", expand=True, padx=4, pady=4)

        tab_tmm = tk.Frame(self._nb_calc, bg=BG, padx=8, pady=8)
        tab_surrogate = tk.Frame(self._nb_calc, bg=BG, padx=8, pady=8)
        self._nb_calc.add(tab_tmm, text="TMM + NSGA-II + DE")
        self._nb_calc.add(tab_surrogate, text="Нейросеть (суррогат)")
        self._nb_calc.bind("<<NotebookTabChanged>>",
                          lambda e: self._update_start_button_label())

        tk.Label(tab_tmm, bg=BG, fg="#444444", justify="left", anchor="w",
                wraplength=280,
                text="Честный расчёт. Материалы, найденные покрытия и полные "
                     "кривые RC(f) сохраняются в БД для последующего "
                     "накопления датасета.").pack(fill="x")

        tk.Label(tab_surrogate, bg=BG, fg=FG, anchor="w",
                text="Папка с обученной моделью:"
                ).grid(row=0, column=0, columnspan=2, sticky="w")
        self._var_surrogate_model_dir = tk.StringVar(value="")
        tab_surrogate.columnconfigure(0, weight=1)
        ttk.Entry(tab_surrogate, textvariable=self._var_surrogate_model_dir
                 ).grid(row=1, column=0, sticky="ew", padx=(0, 4))
        ttk.Button(tab_surrogate, text="Обзор…",
                  command=self._browse_surrogate_model_dir
                  ).grid(row=1, column=1)
        tk.Label(tab_surrogate, bg=BG, fg="#666666", font=("Tahoma", 8),
                justify="left", anchor="w", wraplength=280,
                text="Модель ускоряет ПОИСК внутри NSGA-II/DE. Итоговые "
                     "покрытия перед сохранением в БД и показом в "
                     "предпросмотре всегда честно перепроверяются настоящим "
                     "TMM — суррогат не может исказить сохранённые данные, "
                     "только повлиять на скорость и качество поиска."
                ).grid(row=2, column=0, columnspan=2, sticky="w", pady=(6, 0))

        # ── Сохранение в базу данных ─────────────────────────────────────────
        grp_db = groupbox(left, "База данных", row=5, col=0, sticky="ew")
        self._var_save_to_db       = tk.BooleanVar(value=True)
        self._var_save_curves_db   = tk.BooleanVar(value=True)
        self._var_save_only_valid  = tk.BooleanVar(value=True)
        ttk.Checkbutton(grp_db, text="Сохранять результаты в локальную БД",
                       variable=self._var_save_to_db
                       ).grid(row=0, column=0, sticky="w", pady=1)
        ttk.Checkbutton(grp_db, text="Только валидные (RC ≤ порога)",
                       variable=self._var_save_only_valid
                       ).grid(row=1, column=0, sticky="w", pady=1)
        ttk.Checkbutton(grp_db, text="Хранить полные кривые RC(f) (для ML)",
                       variable=self._var_save_curves_db
                       ).grid(row=2, column=0, sticky="w", pady=1)

        # ── Прогресс ──────────────────────────────────────────────────────────
        grp_prog = groupbox(left, "Прогресс", row=6, col=0, sticky="ew")
        grp_prog.columnconfigure(0, weight=1)
        self._var_progress = tk.DoubleVar(value=0)
        self._var_status   = tk.StringVar(value="Готов к запуску")
        ttk.Progressbar(grp_prog, variable=self._var_progress, maximum=100,
                        style="Horizontal.TProgressbar", length=290
                        ).grid(row=0, column=0, sticky="ew", pady=(2, 4))
        ttk.Label(grp_prog, textvariable=self._var_status,
                  style="Dark.TLabel", foreground=FG_DIM,
                  wraplength=290, anchor="w"
                  ).grid(row=1, column=0, sticky="ew")

        # ── Кнопки ────────────────────────────────────────────────────────────
        btn_frame = ttk.Frame(left, style="TFrame", padding=(0, 6, 0, 0))
        btn_frame.grid(row=7, column=0, sticky="ew")
        for i in range(3):
            btn_frame.columnconfigure(i, weight=1)

        self._btn_start   = ttk.Button(btn_frame, text="Запустить",
                                       command=self._start)
        self._btn_cancel  = ttk.Button(btn_frame, text="Отмена",
                                       command=self._cancel, state="disabled")
        self._btn_preview = ttk.Button(btn_frame, text="Предпросмотр",
                                       command=self._open_preview, state="disabled")

        self._btn_start  .grid(row=0, column=0, sticky="ew", padx=(0, 3), ipady=4)
        self._btn_cancel .grid(row=0, column=1, sticky="ew", padx=(3, 3), ipady=4)
        self._btn_preview.grid(row=0, column=2, sticky="ew", padx=(3, 0), ipady=4)

        self._btn_db = ttk.Button(btn_frame, text="База данных…",
                                  command=self._open_db_viewer)
        self._btn_db.grid(row=1, column=0, columnspan=3, sticky="ew",
                          padx=0, pady=(4, 0), ipady=3)

        # ══════════════════════════════════════════════════════════════════════
        # Правая колонка — журнал
        # ══════════════════════════════════════════════════════════════════════
        right = ttk.Frame(body, style="TFrame")
        right.grid(row=0, column=1, sticky="nsew")
        right.rowconfigure(1, weight=1)
        right.columnconfigure(0, weight=1)

        ttk.Label(right, text="Журнал выполнения",
                  style="Bold.TLabel").grid(row=0, column=0, sticky="w", pady=(0, 2))

        log_frame = tk.Frame(right, relief="sunken", bd=2, bg=LOG_BG)
        log_frame.grid(row=1, column=0, sticky="nsew")
        log_frame.rowconfigure(0, weight=1)
        log_frame.columnconfigure(0, weight=1)

        self._log_text = tk.Text(
            log_frame, state="disabled", wrap="word",
            bg=LOG_BG, fg=LOG_FG, font=(MONO, 9),
            relief="flat", bd=0,
            selectbackground=SEL_BG, selectforeground="#fff",
            width=64
        )
        sb_log = tk.Scrollbar(log_frame, orient="vertical",
                              command=self._log_text.yview,
                              bg=BG_DARK, troughcolor=BG_DARK, relief="flat")
        self._log_text.configure(yscrollcommand=sb_log.set)
        self._log_text.grid(row=0, column=0, sticky="nsew")
        sb_log.grid(row=0, column=1, sticky="ns")

        self._log_text.tag_config("INFO",    foreground="#000080")
        self._log_text.tag_config("WARNING", foreground="#7f5200")
        self._log_text.tag_config("ERROR",   foreground="#cc0000")
        self._log_text.tag_config("plain",   foreground=LOG_FG)

        # ── Статус-бар ────────────────────────────────────────────────────────
        statusbar = tk.Frame(root, bg=BG_DARK, relief="sunken", bd=1)
        statusbar.grid(row=2, column=0, sticky="ew")
        tk.Label(statusbar, textvariable=self._var_status,
                 bg=BG_DARK, fg=FG_DIM, font=FONT_SM,
                 anchor="w", padx=6, pady=2).pack(side="left")
        tk.Label(statusbar, text="RC Absorber Optimiser  v11",
                 bg=BG_DARK, fg=FG_DIM, font=FONT_SM,
                 anchor="e", padx=6, pady=2).pack(side="right")

        # Запустить опрос очереди лога
        self._root.after(100, self._poll_log)
        # Инициализировать подсказку порога RC (на случай нестандартного дефолта)
        self._on_rc_threshold_change()

    # ─────────────────────────────────────────────────────────────────────────
    # Диалоги выбора папок
    # ─────────────────────────────────────────────────────────────────────────

    def _browse_folder(self) -> None:
        d = filedialog.askdirectory(title="Папка с файлами материалов (.xlsx)")
        if d:
            self._var_folder.set(d)

    def _browse_outdir(self) -> None:
        d = filedialog.askdirectory(title="Папка для сохранения результатов")
        if d:
            self._var_outdir.set(d)

    # ─────────────────────────────────────────────────────────────────────────
    # Порог RC и слайдер толщины
    # ─────────────────────────────────────────────────────────────────────────

    def _on_rc_threshold_change(self, event=None) -> None:
        """Валидировать порог RC при потере фокуса / Enter, подсветить ошибку."""
        raw = self._var_rc_threshold.get().strip()
        try:
            value = validate_rc_threshold(raw)
            # Корректное значение — нормальный фон, обновить текст
            self._entry_rc_threshold.config(bg=ENTRY_BG)
            self._lbl_rc_hint.config(text="(должно быть < 0)", foreground=FG_DIM)
            self._var_rc_threshold.set(f"{value:.1f}")
        except ValueError as e:
            # Некорректное значение — подсветить красным, не блокировать ввод
            self._entry_rc_threshold.config(bg="#ffd6d6")
            self._lbl_rc_hint.config(text="Ошибка: значение должно быть < 0",
                                     foreground="#cc0000")

    def _on_lambda_slider_change(self, value) -> None:
        idx = int(round(float(value)))
        idx = max(0, min(idx, len(self._lambda_values) - 1))
        mult = self._lambda_values[idx]
        self._var_lambda_label.set(LAMBDA_MULTIPLIER_LABELS.get(mult, f"×{mult:.3g}"))

    def _on_inverse_design_toggle(self) -> None:
        if self._var_inverse_design.get():
            self._lbl_lambda_title.grid_remove()
            self._scale_lambda.grid_remove()
            self._lbl_lambda.grid_remove()
            self._grp_inverse.grid()
        else:
            self._grp_inverse.grid_remove()
            self._lbl_lambda_title.grid()
            self._scale_lambda.grid()
            self._lbl_lambda.grid()

    def _get_validated_target_band(self) -> tuple:
        """Вернуть (f_lo_ghz, f_hi_ghz, max_thickness_mm) или вызвать ValueError."""
        try:
            f_lo = float(self._var_target_f_lo.get().strip().replace(",", "."))
            f_hi = float(self._var_target_f_hi.get().strip().replace(",", "."))
        except ValueError:
            raise ValueError("Целевая полоса частот должна быть числами (ГГц).")
        if f_lo <= 0 or f_hi <= 0 or f_hi <= f_lo:
            raise ValueError("Целевая полоса некорректна: f_hi должна быть больше "
                             "f_lo, оба значения положительны.")
        try:
            max_mm = float(self._var_target_max_mm.get().strip().replace(",", "."))
        except ValueError:
            raise ValueError("Макс. толщина покрытия должна быть числом (мм).")
        if max_mm <= 0:
            raise ValueError("Макс. толщина покрытия должна быть положительной.")
        return f_lo, f_hi, max_mm

    def _get_validated_rc_threshold(self) -> float:
        """Вернуть валидный порог RC или вызвать ValueError."""
        return validate_rc_threshold(self._var_rc_threshold.get().strip())

    def _get_lambda_multiplier(self) -> float:
        idx = self._var_lambda_idx.get()
        idx = max(0, min(idx, len(self._lambda_values) - 1))
        return self._lambda_values[idx]

    # ─────────────────────────────────────────────────────────────────────────
    # Запуск / отмена оптимизации
    # ─────────────────────────────────────────────────────────────────────────

    def _start(self) -> None:
        folder = self._var_folder.get().strip()
        outdir = self._var_outdir.get().strip()
        if not folder or not os.path.isdir(folder):
            messagebox.showerror("Ошибка",
                                 "Укажите корректную папку с материалами (.xlsx).")
            return
        if not outdir or not os.path.isdir(outdir):
            messagebox.showerror("Ошибка",
                                 "Укажите корректную папку для сохранения.")
            return

        if self._get_calc_method() == "surrogate":
            model_dir = self._var_surrogate_model_dir.get().strip()
            if not model_dir or not os.path.isdir(model_dir):
                messagebox.showerror(
                    "Ошибка",
                    "На вкладке «Нейросеть (суррогат)» не указана папка с "
                    "обученной моделью, или она не существует.\n\n"
                    "Обучить модель можно во вкладке «ML-суррогат» окна "
                    "«База данных…».")
                return

        # Валидация порога RC перед запуском
        try:
            rc_threshold = self._get_validated_rc_threshold()
        except ValueError as e:
            self._on_rc_threshold_change()   # подсветить поле красным
            messagebox.showerror("Ошибка", str(e))
            return

        lambda_multiplier = self._get_lambda_multiplier()

        inverse_design = self._var_inverse_design.get()
        target_f_lo = target_f_hi = target_max_mm = None
        if inverse_design:
            try:
                target_f_lo, target_f_hi, target_max_mm = self._get_validated_target_band()
            except ValueError as e:
                messagebox.showerror("Ошибка", str(e))
                return

        worker.cancel_flag.clear()
        self._last_results = None
        self._btn_start  .config(state="disabled")
        self._btn_cancel .config(state="normal")
        self._btn_preview.config(state="disabled")

        self._log_text.config(state="normal")
        self._log_text.delete("1.0", "end")
        self._log_text.config(state="disabled")
        self._var_progress.set(0)
        self._var_status.set("Инициализация…")

        params = {
            'folder_path':       folder,
            'output_dir':        outdir,
            'mode':              self._var_mode.get(),
            'N_max_layers':      self._var_layers.get(),
            'rc_threshold':      rc_threshold,
            'lambda_multiplier': lambda_multiplier,
            'save_to_db':        self._var_save_to_db.get(),
            'save_curves_to_db': self._var_save_curves_db.get(),
            'save_only_valid_to_db': self._var_save_only_valid.get(),
            'calc_method':        self._get_calc_method(),
            'surrogate_model_dir': self._var_surrogate_model_dir.get().strip(),
            'inverse_design':      inverse_design,
            'target_f_lo_ghz':     target_f_lo,
            'target_f_hi_ghz':     target_f_hi,
            'target_max_thickness_mm': target_max_mm,
        }

        def progress_cb(done, total):
            pct = min(100.0, done / total * 100)
            self._root.after(0, lambda: self._var_progress.set(pct))
            self._root.after(0, lambda: self._var_status.set(
                f"Поколение {done} / {total}  ({pct:.0f}%)"))

        def result_cb(configs, freqs, mat_arrays, names, used_rc_threshold):
            self._last_results = (configs, freqs, mat_arrays, names,
                                  used_rc_threshold)

        def done_cb(success, msg):
            def _finish():
                self._btn_start .config(state="normal")
                self._btn_cancel.config(state="disabled")
                if success:
                    self._var_progress.set(100)
                    self._var_status.set("Завершено успешно")
                    if self._last_results:
                        self._btn_preview.config(state="normal")
                    messagebox.showinfo("Готово", msg)
                else:
                    self._var_status.set("Остановлено")
                    messagebox.showwarning("Отмена / Ошибка", msg)
            self._root.after(0, _finish)

        threading.Thread(
            target=worker.run_optimisation,
            args=(params, progress_cb, done_cb, result_cb),
            daemon=True
        ).start()

    def _cancel(self) -> None:
        worker.cancel_flag.set()
        self._var_status.set("Отмена…")
        self._btn_cancel.config(state="disabled")

    # ─────────────────────────────────────────────────────────────────────────
    # Предпросмотр результатов
    # ─────────────────────────────────────────────────────────────────────────

    def _open_preview(self) -> None:
        if self._last_results is None:
            messagebox.showinfo("Предпросмотр",
                                "Нет готовых результатов. Запустите оптимизацию.")
            return
        # Если окно уже открыто — поднять его на передний план
        if self._preview_win is not None and self._preview_win.winfo_exists():
            self._preview_win.lift()
            self._preview_win.focus_force()
            return
        configs, freqs, mat_arrays, names, used_rc_threshold = self._last_results
        self._preview_win = PreviewWindow(
            self._root, configs, freqs, mat_arrays, names,
            rc_threshold=used_rc_threshold
        )

    # ─────────────────────────────────────────────────────────────────────────
    # Окно базы данных
    # ─────────────────────────────────────────────────────────────────────────

    # ─────────────────────────────────────────────────────────────────────────
    # Метод расчёта (TMM / суррогат)
    # ─────────────────────────────────────────────────────────────────────────

    def _get_calc_method(self) -> str:
        """Метод расчёта определяется тем, какая вкладка сейчас активна:
        «TMM + NSGA-II + DE» (индекс 0) или «Нейросеть (суррогат)» (индекс 1)."""
        return "surrogate" if self._nb_calc.index("current") == 1 else "tmm"

    def _update_start_button_label(self) -> None:
        if self._get_calc_method() == "surrogate":
            self._btn_start.configure(text="Запустить (суррогат)")
        else:
            self._btn_start.configure(text="Запустить (честный TMM)")

    def _browse_surrogate_model_dir(self) -> None:
        d = filedialog.askdirectory(
            initialdir=self._var_surrogate_model_dir.get() or os.getcwd())
        if d:
            self._var_surrogate_model_dir.set(d)

    def _open_db_viewer(self) -> None:
        if self._db_viewer_win is not None and self._db_viewer_win.winfo_exists():
            self._db_viewer_win.lift()
            self._db_viewer_win.focus_force()
            self._db_viewer_win.refresh()
            return
        self._db_viewer_win = DatabaseWindow(self._root)

    # ─────────────────────────────────────────────────────────────────────────
    # Опрос очереди лога
    # ─────────────────────────────────────────────────────────────────────────

    def _poll_log(self) -> None:
        try:
            while True:
                line = log_queue.get_nowait()
                self._log_text.config(state="normal")
                tag = "plain"
                if "  INFO"    in line: tag = "INFO"
                elif "WARNING" in line: tag = "WARNING"
                elif "ERROR"   in line: tag = "ERROR"
                self._log_text.insert("end", line + "\n", tag)
                self._log_text.see("end")
                self._log_text.config(state="disabled")
        except queue.Empty:
            pass
        self._root.after(100, self._poll_log)
