"""
plotting.py — функции построения графиков

Содержит:
  • plot_rc_for_config  — RC(f) + стек слоёв + чувствительность → PNG
  • plot_pareto_front   — фронт Парето RC vs толщина → PNG
  • make_config_figure  — то же, что plot_rc_for_config, но возвращает Figure
                         (для окна предпросмотра без сохранения файла)
"""

import os
import logging

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

from physics import tmm_reflection, compute_bandwidth

log = logging.getLogger("rc_opt")

# Цветовая карта для материалов - исправлено для новых версий Matplotlib
try:
    _CMAP = plt.colormaps['tab10']
except AttributeError:
    _CMAP = plt.cm.get_cmap('tab10', 10)


def _draw_config_figure(cfg: dict,
                        freqs: np.ndarray,
                        mat_arrays: dict,
                        material_names: list,
                        rc_threshold: float = -10.0) -> plt.Figure:
    """
    Внутренняя функция: создать Figure с тремя панелями.

    Левая панель  — RC(f) с полосой
    Средняя панель — стек слоёв (AIR → METAL)
    Правая панель  — гистограмма чувствительности

    rc_threshold : float
        Порог RC [дБ], используемый для отрисовки целевой линии и полосы.

    Возвращает объект Figure (не закрывает его).
    """
    sequence = cfg['sequence']
    hs       = cfg['hs']
    sens     = cfg.get('sensitivity', {})

    fig = plt.figure(figsize=(17, 7))
    gs  = gridspec.GridSpec(1, 3, width_ratios=[3, 1.1, 1.2], wspace=0.40)

    # ── Левая: RC(f) ─────────────────────────────────────────────────────────
    ax_rc = fig.add_subplot(gs[0])
    RC    = tmm_reflection(freqs, sequence, hs, mat_arrays)
    max_rc_actual = float(np.max(RC))
    bw    = compute_bandwidth(RC, freqs, threshold=rc_threshold)
    ax_rc.plot(freqs / 1e9, RC, color='steelblue', lw=2.0, label='RC (дБ)')
    ax_rc.axhline(rc_threshold, color='crimson', ls='--', lw=1.4,
                 label=f'{rc_threshold:.1f} дБ')
    ax_rc.fill_between(freqs / 1e9, RC, rc_threshold,
                       where=(RC <= rc_threshold), alpha=0.15, color='steelblue',
                       label=f'Полоса = {bw:.2f} ГГц')
    ax_rc.set_xlabel('Частота (ГГц)', fontsize=12)
    ax_rc.set_ylabel('RC (дБ)', fontsize=12)
    ax_rc.set_title(
        f"Ранг #{cfg.get('rank', '?')}   "
        f"max RC = {max_rc_actual:+.1f} дБ   "
        f"толщина = {cfg['total_mm']:.1f} мм",
        fontsize=11, fontweight='bold')
    ax_rc.legend(fontsize=10)
    ax_rc.grid(True, alpha=0.35)

    # ── Средняя: стек слоёв ───────────────────────────────────────────────────
    ax_st = fig.add_subplot(gs[1])
    ax_st.set_xlim(0, 1)
    ax_st.set_ylim(0, 1)
    ax_st.axis('off')
    ax_st.set_title('Стек слоёв\n(воздух → металл)', fontsize=10, fontweight='bold')

    total_h = sum(hs)
    bar_w   = 0.42
    x0      = 0.52
    plot_h  = 0.80
    y_cur   = 0.93
    ax_st.text(x0 + bar_w / 2, 0.97, 'ВОЗДУХ', ha='center', va='top',
               fontsize=8, color='gray', fontstyle='italic')

    for l in range(len(sequence)):
        mi    = sequence[l] - 1
        bh    = max(hs[l] / total_h * plot_h, 0.02)
        color = _CMAP(mi % 10)
        ax_st.add_patch(plt.Rectangle((x0, y_cur - bh), bar_w, bh,
                                      color=color, alpha=0.85,
                                      lw=0.8, edgecolor='black'))
        ax_st.text(x0 + bar_w + 0.03, y_cur - bh / 2,
                   f"#{sequence[l]}\n{hs[l]*1000:.1f} мм",
                   va='center', fontsize=8)
        y_cur -= bh

    ax_st.add_patch(plt.Rectangle((x0, y_cur - 0.035), bar_w, 0.035,
                                  color='#555', lw=0.8, edgecolor='black'))
    ax_st.text(x0 + bar_w / 2, y_cur - 0.05, 'МЕТАЛЛ (PEC)', ha='center', va='top',
               fontsize=8, color='#333', fontweight='bold')

    # легенда материалов слева от стека
    unique_mats = sorted(set(sequence))
    swatch_h    = 0.028
    legend_gap  = 0.012
    n_leg       = len(unique_mats)
    leg_block   = n_leg * (swatch_h + legend_gap)
    ly          = 0.50 + leg_block / 2 - swatch_h
    lx_swatch   = 0.01
    lx_text     = lx_swatch + 0.10
    for m in unique_mats:
        ax_st.add_patch(plt.Rectangle((lx_swatch, ly), 0.08, swatch_h,
                                      color=_CMAP((m - 1) % 10), alpha=0.85,
                                      lw=0.5, edgecolor='black'))
        nm = material_names[m - 1] if m - 1 < len(material_names) else f'Мат. {m}'
        ax_st.text(lx_text, ly + swatch_h / 2, f'#{m}: {nm}',
                   va='center', fontsize=7.5)
        ly -= (swatch_h + legend_gap)

    # ── Правая: чувствительность ─────────────────────────────────────────────
    ax_se = fig.add_subplot(gs[2])
    if sens.get('per_layer_dRC'):
        dRC    = sens['per_layer_dRC']
        labels = [f'Слой {i+1}' for i in range(len(dRC))]
        colors = ['#e74c3c' if d >= 2.0 else '#2ecc71' for d in dRC]
        ax_se.barh(labels, dRC, color=colors, edgecolor='black', lw=0.5)
        ax_se.axvline(2.0, color='orange', ls='--', lw=1.2, label='2 дБ')
        robust = sens.get('robust_score', 0)
        ax_se.set_title(
            f'Чувствительность (±5%)\n{robust}/{len(dRC)} робастных',
            fontsize=9, fontweight='bold')
        ax_se.set_xlabel('δRC [дБ]', fontsize=9)
        ax_se.legend(fontsize=8)
        ax_se.grid(True, alpha=0.3, axis='x')
    else:
        ax_se.text(0.5, 0.5, 'Нет данных чувствительности',
                   ha='center', va='center',
                   transform=ax_se.transAxes, fontsize=10, color='gray')
        ax_se.axis('off')

    seq_str = ' → '.join(str(s) for s in sequence)
    fig.text(0.5, 0.005,
             f"Последовательность: {seq_str}  |  Суммарно: {total_h*1000:.1f} мм  |  "
             f"Полоса<{rc_threshold:.0f}дБ: {bw:.2f} ГГц  |  Оценка: {cfg.get('score', 0):.3f}",
             ha='center', fontsize=9, color='#333',
             bbox=dict(boxstyle='round,pad=0.3', fc='#f4f4f4', alpha=0.85))
    plt.tight_layout(rect=[0, 0.06, 1, 1])
    return fig


def plot_rc_for_config(cfg: dict,
                       freqs: np.ndarray,
                       mat_arrays: dict,
                       material_names: list,
                       filename: str = 'config.png',
                       rc_threshold: float = -10.0) -> None:
    """Построить и сохранить детальный график конфигурации в файл PNG."""
    fig = _draw_config_figure(cfg, freqs, mat_arrays, material_names, rc_threshold)
    fig.savefig(filename, dpi=150, bbox_inches='tight')
    plt.close(fig)
    log.info(f"  График сохранён: {filename}")


def make_config_figure(cfg: dict,
                       freqs: np.ndarray,
                       mat_arrays: dict,
                       material_names: list,
                       rc_threshold: float = -10.0) -> plt.Figure:
    """
    Создать Figure для предпросмотра (без сохранения файла).
    Вызывающий код обязан закрыть Figure после использования.
    """
    return _draw_config_figure(cfg, freqs, mat_arrays, material_names, rc_threshold)


def plot_pareto_front(configs: list, filename: str = 'pareto_front.png',
                      rc_threshold: float = -10.0) -> None:
    """Построить и сохранить график Парето-фронта RC vs толщина."""
    if not configs:
        return
    fig, ax = plt.subplots(figsize=(10, 6))
    layers_set = sorted(set(c['n_layers'] for c in configs))
    
    # Исправлено для новых версий Matplotlib
    try:
        cmap = plt.colormaps['tab10']
    except AttributeError:
        cmap = plt.cm.get_cmap('tab10', max(len(layers_set), 2))
    
    for i, nl in enumerate(layers_set):
        xs = [c['total_mm'] for c in configs if c['n_layers'] == nl]
        ys = [c['max_rc']   for c in configs if c['n_layers'] == nl]
        ax.scatter(xs, ys, color=cmap(i), alpha=0.55, s=35, label=f'{nl} сл.')
    pareto = sorted([c for c in configs if c.get('rank') == 1],
                    key=lambda c: c['total_mm'])
    if pareto:
        ax.plot([c['total_mm'] for c in pareto],
                [c['max_rc']   for c in pareto],
                'k--', lw=1.2, alpha=0.6, label='Фронт Парето')
        ax.scatter([c['total_mm'] for c in pareto],
                   [c['max_rc']   for c in pareto],
                   color='black', s=60, zorder=5)
    ax.axhline(rc_threshold, color='crimson', ls=':', lw=1.2,
              label=f'{rc_threshold:.1f} дБ')
    ax.set_xlabel('Суммарная толщина (мм)', fontsize=12)
    ax.set_ylabel('Max RC (дБ)', fontsize=12)
    ax.set_title('Фронт Парето: RC vs толщина', fontsize=13, fontweight='bold')
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.35)
    plt.tight_layout()
    fig.savefig(filename, dpi=150, bbox_inches='tight')
    plt.close(fig)
    log.info(f"  Фронт Парето: {filename}")