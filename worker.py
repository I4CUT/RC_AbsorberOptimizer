"""
worker.py — фоновый поток оптимизации

Содержит функцию run_optimisation(), которая запускается в отдельном потоке,
чтобы не блокировать GUI. Общается с GUI через callback-функции.
"""

import os
import logging
import threading

import numpy as np
from scipy.constants import c

from physics import (
    load_material_data, build_mat_arrays, clear_caches,
    nsga2_optimise, rank_configurations,
    sensitivity_analysis, compute_bandwidth,
    tmm_reflection, write_csv, RC_THRESHOLD,
    H_MIN, LAMBDA_MULTIPLIER_DEFAULT,
    validate_rc_threshold, validate_lambda_multiplier,
)
from plotting import plot_rc_for_config, plot_pareto_front
from database import Database

log = logging.getLogger("rc_opt")

# Глобальный флаг отмены — устанавливается кнопкой «Отмена»
cancel_flag = threading.Event()


def run_optimisation(params: dict, progress_cb, done_cb, result_cb) -> None:
    """
    Запустить оптимизацию в фоновом потоке.

    Параметры
    ---------
    params : dict
        folder_path        — папка с .xlsx файлами материалов
        output_dir         — папка для сохранения результатов
        mode               — '1' | '2' | '3'
        N_max_layers       — максимальное число слоёв
        rc_threshold       — порог RC [дБ], должен быть < 0 (по умолчанию -10.0)
        lambda_multiplier  — множитель λ для ограничения суммарной толщины
                              покрытия; одно из {1/8, 1/4, 1/2, 1, 2} (по умолчанию 1)

    progress_cb(done: int, total: int)
        Вызывается после каждого поколения NSGA-II.

    done_cb(success: bool, message: str)
        Вызывается по окончании (успех или ошибка).

    result_cb(configs, freqs, mat_arrays, material_names, rc_threshold)
        Вызывается с готовыми результатами — для открытия окна предпросмотра.
        rc_threshold — фактический порог, использованный в этом прогоне.
    """
    try:
        folder_path  = params['folder_path']
        output_dir   = params['output_dir']
        mode         = params['mode']
        N_max_layers = params['N_max_layers']

        # ── Порог RC и множитель λ (с валидацией и безопасными дефолтами) ────
        try:
            rc_threshold = validate_rc_threshold(
                params.get('rc_threshold', RC_THRESHOLD))
        except ValueError as e:
            log.warning(f"Некорректный порог RC, использую {RC_THRESHOLD}: {e}")
            rc_threshold = RC_THRESHOLD

        lambda_multiplier = validate_lambda_multiplier(
            params.get('lambda_multiplier', LAMBDA_MULTIPLIER_DEFAULT))

        # ── Переключить лог-файл в папку вывода ──────────────────────────────
        log_path = os.path.join(output_dir, "rc_optimiser.log")
        for h in log.handlers:
            if isinstance(h, logging.FileHandler):
                h.close()
                log.removeHandler(h)
        fh = logging.FileHandler(log_path, mode='w', encoding='utf-8')
        fh.setFormatter(logging.Formatter("%(asctime)s  %(levelname)-8s  %(message)s"))
        log.addHandler(fh)

        log.info("=" * 60)
        log.info("RC Absorber Optimiser  (TMM + NSGA-II + DE)")
        log.info("=" * 60)

        freqs, material_datas, files = load_material_data(folder_path)
        log.info("\nИндексы материалов:")
        for i, fname in enumerate(files, 1):
            log.info(f"  {i:>3}  →  {fname}")

        material_names = [os.path.splitext(f)[0] for f in files]
        mat_arrays_full = build_mat_arrays(material_datas)
        num_materials  = len(material_datas)

        # ── Inverse Design: поиск сфокусирован на целевой полосе частот ──────
        # (пользователь сам выбирает материалы — как обычно — но задаёт полосу
        # интереса и прямую макс. толщину вместо множителя λ по центру всего
        # измеренного диапазона). Итоговая честная проверка ниже всегда
        # использует ПОЛНУЮ сетку материалов — это не меняется.
        inverse_design = bool(params.get('inverse_design', False))
        target_mask = None
        if inverse_design:
            f_lo_hz = float(params['target_f_lo_ghz']) * 1e9
            f_hi_hz = float(params['target_f_hi_ghz']) * 1e9
            target_mask = (freqs >= f_lo_hz) & (freqs <= f_hi_hz)
            if int(np.sum(target_mask)) < 2:
                done_cb(False, f"Целевая полоса {f_lo_hz/1e9:.2f}–{f_hi_hz/1e9:.2f} ГГц "
                        f"не пересекается (или почти не пересекается) с измеренным "
                        f"диапазоном материалов "
                        f"({freqs.min()/1e9:.2f}–{freqs.max()/1e9:.2f} ГГц).")
                return
            freqs_search = freqs[target_mask]
            mat_arrays_search = {k: v[:, target_mask] for k, v in mat_arrays_full.items()}
            h_max_abs = float(params['target_max_thickness_mm']) / 1000.0
            log.info(f"Inverse Design: целевая полоса "
                    f"{f_lo_hz/1e9:.2f}–{f_hi_hz/1e9:.2f} ГГц "
                    f"({int(np.sum(target_mask))} точек из {len(freqs)}), "
                    f"макс. толщина = {h_max_abs*1000:.1f} мм (задана напрямую)")
        else:
            freqs_search = freqs
            mat_arrays_search = mat_arrays_full
            f_center  = (float(np.min(freqs)) + float(np.max(freqs))) / 2.0
            lambda_c  = c / f_center
            h_max_abs = lambda_c * lambda_multiplier
            log.info(f"Центральная частота: {f_center/1e9:.2f} ГГц  |  "
                    f"λ = {lambda_c*1000:.1f} мм")
            log.info(f"Множитель толщины: ×{lambda_multiplier:.3g}  →  "
                    f"макс. суммарная толщина = {h_max_abs*1000:.1f} мм")

        log.info(f"Порог RC: {rc_threshold:+.1f} дБ")

        w_map   = {'1': (1.0, 0.0), '2': (0.1, 0.9), '3': (1.0, 0.3)}
        w_rc, w_th = w_map.get(mode, (1.0, 0.3))
        log.info(f"Режим {mode}:  w_rc={w_rc},  w_th={w_th}")

        # ── Метод расчёта во время ПОИСКА: честный TMM (по умолчанию) или ────
        # экспериментальный суррогат. В ЛЮБОМ случае итоговые конфигурации
        # ниже честно пересчитываются через tmm_reflection перед сохранением —
        # это не меняется независимо от выбора здесь (см. блок all_configs.append).
        rc_evaluator = None  # None → nsga2_optimise использует честный TMM
        calc_method = params.get('calc_method', 'tmm')
        if calc_method == 'surrogate':
            model_dir = params.get('surrogate_model_dir', '')
            try:
                from ml_surrogate import SurrogateModel, build_rc_evaluator
                surrogate_model = SurrogateModel.load(model_dir)
                rc_evaluator = build_rc_evaluator(surrogate_model, material_datas)
                log.info(f"Метод расчёта при поиске: СУРРОГАТ-МОДЕЛЬ "
                        f"({model_dir}). Итоговые результаты будут перепроверены "
                        f"честным TMM перед сохранением.")
            except Exception as exc:
                log.warning(f"Не удалось загрузить суррогат-модель ({exc}) — "
                           f"использую честный TMM для поиска.")
                rc_evaluator = None
        if rc_evaluator is None:
            log.info("Метод расчёта при поиске: честный TMM.")

        # Суммарное число поколений с адаптивной формулой
        total_gens = sum(max(30, 80 - n * 5) for n in range(1, N_max_layers + 1))
        gen_done   = [0]

        all_configs: list = []

        for n_layers in range(1, N_max_layers + 1):
            if cancel_flag.is_set():
                done_cb(False, "Отменено пользователем.")
                return

            log.info(f"\n{'─'*55}")
            log.info(f"NSGA-II  —  {n_layers} слой(-ёв/-ев)")
            clear_caches()

            def _progress():
                gen_done[0] += 1
                progress_cb(gen_done[0], total_gens)

            t_layer = __import__('time').time()
            try:
                candidates = nsga2_optimise(
                    n_layers, num_materials, freqs_search, mat_arrays_search,
                    h_max_total=h_max_abs, w_rc=w_rc, w_th=w_th,
                    rc_threshold=rc_threshold,
                    rc_evaluator=rc_evaluator,
                    _cancel_flag=cancel_flag,
                    _progress_cb=_progress,
                )
            except Exception as exc:
                log.error(f"  NSGA-II n={n_layers} завершился с ошибкой: {exc}")
                continue
            log.info(f"  Время для n={n_layers}: {__import__('time').time()-t_layer:.1f}с")

            if cancel_flag.is_set():
                done_cb(False, "Отменено пользователем.")
                return

            if not candidates:
                log.warning(f"  n={n_layers}: кандидатов не найдено, пропускаем.")
                continue

            for seq, hs, rc, th in candidates:
                # Честный пересчёт ВСЕГДА на полной измеренной сетке материалов —
                # это то, что сохраняется в БД и показывается в графиках,
                # независимо от того, был ли поиск сфокусирован на подполосе.
                RC_arr_full = tmm_reflection(freqs, seq, hs, mat_arrays_full)
                if inverse_design:
                    eval_curve, eval_freqs = RC_arr_full[target_mask], freqs[target_mask]
                else:
                    eval_curve, eval_freqs = RC_arr_full, freqs
                max_rc = float(np.max(eval_curve))      # источник истины — честный пересчёт
                bw     = compute_bandwidth(eval_curve, eval_freqs, threshold=rc_threshold)
                if abs(rc - max_rc) > 0.05:
                    log.warning(
                        f"  Расхождение RC оптимизатора ({rc:+.2f} дБ) и честного "
                        f"пересчёта ({max_rc:+.2f} дБ) для seq={seq}, "
                        f"hs={tuple(round(h*1000,3) for h in hs)} мм — "
                        f"использую пересчёт.")
                try:
                    sens = sensitivity_analysis(seq, hs, freqs_search, mat_arrays_search)
                except Exception:
                    sens = {}
                all_configs.append({
                    'n_layers': n_layers, 'sequence': seq, 'hs': hs,
                    'max_rc': max_rc, 'total_mm': th,
                    'bandwidth_ghz': bw, 'sensitivity': sens,
                })
            log.info(f"  {len(candidates)} Парето-кандидатов для {n_layers} слоя(-ёв/-ев).")

        log.info(f"\nВсего кандидатов: {len(all_configs)}")
        if not all_configs:
            done_cb(False, "Не найдено ни одной конфигурации. "
                    "Проверьте файлы материалов и параметры.")
            return
        all_configs   = rank_configurations(all_configs, rc_threshold=rc_threshold)
        valid_configs = [c for c in all_configs if c.get('rank') is not None]

        if not valid_configs:
            log.warning(f"Нет конфигураций с RC <= {rc_threshold} дБ.")

        # ── Сохранить в локальную БД (материалы + прогон + покрытия) ──────────
        # По умолчанию сохраняются только валидные конфигурации (RC <= порога) —
        # это то же множество, что видно в окне предпросмотра. Невалидные тоже
        # можно копить (чекбокс в GUI), но по умолчанию выключено, чтобы БД не
        # засорялась «мусорными» решениями, которые всё равно никто не использует.
        if params.get('save_to_db', True):
            try:
                db = Database()
                material_ids = db.register_materials(material_datas, files)
                run_id = db.create_run(params, material_ids, w_rc=w_rc, w_th=w_th)
                saved_ids = db.add_coatings(
                    run_id, all_configs, material_ids,
                    rc_threshold=rc_threshold,
                    freqs=freqs, mat_arrays=mat_arrays_full,
                    only_valid=params.get('save_only_valid_to_db', True),
                    save_curves=params.get('save_curves_to_db', True),
                )
                log.info(f"БД: run_id={run_id}, сохранено {len(saved_ids)} "
                         f"покрытий ({db.db_path})")

                if inverse_design:
                    best_coating_id = None
                    rank1 = [c for c in db.list_coatings(run_id=run_id)
                            if c.get('rank') == 1]
                    if rank1:
                        best_coating_id = rank1[0]['coating_id']
                    target_id = db.create_design_target(
                        f_lo_hz=float(params['target_f_lo_ghz']) * 1e9,
                        f_hi_hz=float(params['target_f_hi_ghz']) * 1e9,
                        rc_target_db=rc_threshold,
                        max_thickness_mm=float(params['target_max_thickness_mm']),
                        allowed_material_ids=material_ids,
                        comment=f"run_id={run_id}",
                    )
                    db.set_design_target_result(target_id, best_coating_id)
                    if best_coating_id:
                        log.info(f"Inverse Design: цель target_id={target_id} → "
                                f"лучшее покрытие coating_id={best_coating_id}")
                    else:
                        log.warning(f"Inverse Design: цель target_id={target_id} — "
                                   f"подходящего покрытия не найдено.")

                db.close()
            except Exception as exc:
                log.warning(f"Не удалось сохранить результаты в БД: {exc}")

        # ── Сохранить CSV и Парето-фронт ─────────────────────────────────────
        csv_path    = os.path.join(output_dir, 'configurations.csv')
        pareto_path = os.path.join(output_dir, 'pareto_front.png')
        write_csv(csv_path, all_configs, N_max_layers)
        plot_pareto_front(valid_configs or all_configs, pareto_path,
                          rc_threshold=rc_threshold)

        # ── Сохранить PNG для топ-конфигураций ───────────────────────────────
        pool    = valid_configs if valid_configs else all_configs
        to_plot = list(pool[:3])
        for special in [min(pool, key=lambda c: c['max_rc'],   default=None),
                        min(pool, key=lambda c: c['total_mm'], default=None)]:
            if special and special not in to_plot:
                to_plot.append(special)

        for cfg in to_plot:
            tag = (f"rank{cfg.get('rank', '?')}_{cfg['n_layers']}L_"
                   f"rc{cfg['max_rc']:+.1f}dB_{cfg['total_mm']:.1f}mm"
                   ).replace('+', 'p').replace('-', 'm')
            plot_rc_for_config(cfg, freqs, mat_arrays_full, material_names,
                               filename=os.path.join(output_dir, f"config_{tag}.png"),
                               rc_threshold=rc_threshold)

        # ── Статистика ────────────────────────────────────────────────────────
        log.info("\n" + "=" * 55)
        if valid_configs:
            best_rc = min(c['max_rc']        for c in valid_configs)
            thin    = min(c['total_mm']      for c in valid_configs)
            best_bw = max(c['bandwidth_ghz'] for c in valid_configs)
            log.info(f"Лучший RC:     {best_rc:+.2f} дБ")
            log.info(f"Тонкая:        {thin:.1f} мм")
            log.info(f"Макс. полоса:  {best_bw:.2f} ГГц")
            summary = (f"Готово!\n\nЛучший RC: {best_rc:+.2f} дБ\n"
                       f"Мин. толщина: {thin:.1f} мм\n"
                       f"Макс. полоса: {best_bw:.2f} ГГц\n\n"
                       f"Результаты в:\n{output_dir}")
        else:
            best = min(c['max_rc'] for c in all_configs)
            log.info(f"Нет валидных конфигураций. Лучший: {best:+.2f} дБ")
            summary = (f"Готово (нет конфигураций ≤ {rc_threshold} дБ).\n"
                       f"Лучший RC: {best:+.2f} дБ")

        log.info("Готово.")

        # ── Передать результаты для окна предпросмотра ────────────────────────
        result_cb(pool, freqs, mat_arrays_full, material_names, rc_threshold)
        done_cb(True, summary)

    except Exception as exc:
        log.exception("Оптимизация завершилась с ошибкой")
        done_cb(False, f"Ошибка:\n{exc}")
