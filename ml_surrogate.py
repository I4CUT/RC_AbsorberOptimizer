"""
ml_surrogate.py — forward-суррогат: предсказание RC(f) по составу покрытия
без честного TMM-расчёта

Идея (этап 2 дорожной карты): обучить модель, которая по последовательности
материалов + толщинам слоёв предсказывает всю кривую RC(f), минуя
tmm_reflection(). Дальше это можно использовать как быстрый предфильтр
внутри NSGA-II — считать честный TMM только для кандидатов, прошедших
предсказание суррогата (см. roadmap, этап 3 — seed-proposer для inverse design,
который будет опираться именно на этот суррогат).

Модель: sklearn RandomForestRegressor с MultiOutputRegressor (предсказывает
сразу все точки канонической сетки частот как отдельные выходы). Это НЕ
глубокая сеть — сознательный выбор для baseline: работает на CPU без
дополнительных тяжёлых зависимостей (PyTorch и т.п.), не требует много данных
для первых экспериментов, даёт понятные метрики важности признаков.
При достаточном росте датасета это место естественно заменяется на
sequence-encoder (см. docstring в конце файла).

Входные признаки на один слой: [er_mean, er_std, eim_mean, eim_std, mr_mean,
mr_std, mim_mean, mim_std, loss_tangent_mean, er_slope, thickness_mm] —
конкатенируются по всем слоям с паддингом нулями до n_layers_cap (то же
паддинг, что и в dataset_export.py). Порядок слоёв (воздух→металл)
сохраняется как есть — позиция слоя в этом фиксированном порядке уже несёт
физический смысл (близость к металлу/воздуху), поэтому даже плоский
табличный вход способен уловить часть этой структуры.

Использование как скрипт:
    python ml_surrogate.py train --dataset ./ml_dataset --output ./surrogate_model
    python ml_surrogate.py eval  --dataset ./ml_dataset --model ./surrogate_model/model.joblib

Использование как модуль (для будущей интеграции в worker.py/NSGA-II):
    from ml_surrogate import SurrogateModel
    model = SurrogateModel.load("./surrogate_model/model.joblib")
    rc_curve = model.predict_curve(material_scalar_layers, thickness_mm_layers)
"""

from __future__ import annotations

import os
import json
import argparse
import logging

import numpy as np

log = logging.getLogger("rc_opt")

MODEL_FILENAME = "model.joblib"
META_FILENAME = "model_meta.json"


# ─────────────────────────────────────────────────────────────────────────────
# Подготовка признаков из тензоров dataset_export.py
# ─────────────────────────────────────────────────────────────────────────────

def _flatten_features(material_scalar: np.ndarray, thickness_mm: np.ndarray,
                      layer_mask: np.ndarray) -> np.ndarray:
    """
    (N, n_layers_cap, n_scalar) + (N, n_layers_cap) + (N, n_layers_cap)
    → (N, n_layers_cap * (n_scalar + 2))  — признаки на слой + толщина + маска.

    Маска включена как признак явно (а не только как способ игнорировать
    пустые слои), чтобы модель могла обучиться отличать "слоя нет" от
    "слой с нулевыми параметрами".
    """
    n, n_layers_cap, n_scalar = material_scalar.shape
    parts = [
        material_scalar.reshape(n, n_layers_cap * n_scalar),
        thickness_mm.reshape(n, n_layers_cap),
        layer_mask.reshape(n, n_layers_cap),
    ]
    return np.concatenate(parts, axis=1)


def load_dataset(dataset_dir: str) -> dict:
    """Загрузить dataset_tensors.npz + dataset_meta.json из папки экспорта."""
    npz_path = os.path.join(dataset_dir, "dataset_tensors.npz")
    meta_path = os.path.join(dataset_dir, "dataset_meta.json")
    if not os.path.exists(npz_path):
        raise FileNotFoundError(
            f"Не найден {npz_path}. Сначала выполните экспорт "
            f"(dataset_export.py или вкладка «Экспорт датасета»), "
            f"обязательно с включёнными кривыми (--no-curves НЕ указан).")

    data = np.load(npz_path, allow_pickle=True)
    if "material_curve" not in data or "rc_curve_db" not in data:
        raise ValueError(
            "В датасете нет полных кривых (material_curve/rc_curve_db). "
            "Суррогат по кривым требует экспорта с include_curves=True "
            "(это значение по умолчанию и в CLI, и в окне БД).")

    meta = {}
    if os.path.exists(meta_path):
        with open(meta_path, encoding="utf-8") as f:
            meta = json.load(f)

    return {
        "material_scalar": data["material_scalar"],
        "thickness_mm": data["thickness_mm"],
        "layer_mask": data["layer_mask"],
        "rc_curve_db": data["rc_curve_db"],
        "canon_freq_hz": data["canon_freq_hz"],
        "scalar_feature_names": list(data["scalar_feature_names"]),
        "coating_id": data["coating_id"],
        "meta": meta,
    }


# ─────────────────────────────────────────────────────────────────────────────
# Модель
# ─────────────────────────────────────────────────────────────────────────────

class SurrogateModel:
    """
    Обёртка вокруг sklearn-регрессора, предсказывающего кривую RC(f).

    Хранит вместе с моделью всё, что нужно для инференса: каноническую
    сетку частот, на которой обучалась модель (предсказания вне её не имеют
    смысла), число слоёв паддинга и имена скалярных признаков материала —
    чтобы при использовании модели нельзя было случайно перепутать порядок
    входных признаков.
    """

    def __init__(self, regressor, canon_freq_hz: np.ndarray,
                n_layers_cap: int, scalar_feature_names: list):
        self.regressor = regressor
        self.canon_freq_hz = np.asarray(canon_freq_hz, dtype=float)
        self.n_layers_cap = n_layers_cap
        self.scalar_feature_names = list(scalar_feature_names)

    # ── Обучение ──────────────────────────────────────────────────────────

    @classmethod
    def train(cls, dataset: dict, n_estimators: int = 200,
              max_depth: int | None = None, test_size: float = 0.2,
              random_state: int = 0) -> tuple["SurrogateModel", dict]:
        from sklearn.ensemble import RandomForestRegressor
        from sklearn.model_selection import train_test_split
        from sklearn.metrics import mean_absolute_error, r2_score

        X = _flatten_features(dataset["material_scalar"], dataset["thickness_mm"],
                              dataset["layer_mask"])
        y = dataset["rc_curve_db"]

        n = X.shape[0]
        if n < 10:
            log.warning(
                f"В датасете всего {n} сэмплов — этого достаточно только для "
                f"проверки работоспособности пайплайна (smoke-test), но "
                f"НЕ для содержательного обучения. Для реального суррогата "
                f"нужны как минимум сотни-тысячи разнообразных покрытий.")

        # При очень малом числе сэмплов train_test_split может не оставить
        # вообще ничего в тесте — тогда просто оцениваем на обучающей выборке
        # с явным предупреждением (только для smoke-теста, не для отчёта).
        if n * test_size < 2:
            X_train, y_train = X, y
            X_test, y_test = X, y
            log.warning("Слишком мало данных для отдельной тестовой выборки — "
                       "метрики ниже посчитаны на обучающих данных (не показательны).")
        else:
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=test_size, random_state=random_state)

        base = RandomForestRegressor(
            n_estimators=n_estimators, max_depth=max_depth,
            random_state=random_state, n_jobs=-1)
        base.fit(X_train, y_train)

        y_pred = base.predict(X_test)
        mae_per_freq = mean_absolute_error(y_test, y_pred, multioutput="raw_values")
        mae_curve_mean = float(np.mean(mae_per_freq))
        r2 = float(r2_score(y_test, y_pred, multioutput="variance_weighted"))

        # Метрика, физически более понятная, чем усреднённый MAE по частотам:
        # ошибка предсказания max(RC) — именно её оптимизирует NSGA-II.
        max_rc_true = np.max(y_test, axis=1)
        max_rc_pred = np.max(y_pred, axis=1)
        mae_max_rc = float(mean_absolute_error(max_rc_true, max_rc_pred))

        metrics = {
            "n_samples": n, "n_train": int(len(X_train)), "n_test": int(len(X_test)),
            "mae_curve_db_mean": round(mae_curve_mean, 3),
            "r2_curve": round(r2, 3),
            "mae_max_rc_db": round(mae_max_rc, 3),
        }
        log.info(f"Обучение завершено: {metrics}")

        model = cls(base, dataset["canon_freq_hz"],
                   dataset["material_scalar"].shape[1],
                   dataset["scalar_feature_names"])
        return model, metrics

    # ── Инференс ──────────────────────────────────────────────────────────

    def predict_curve(self, material_scalar_layers: np.ndarray,
                      thickness_mm_layers: np.ndarray) -> np.ndarray:
        """
        Предсказать RC(f) на канонической сетке self.canon_freq_hz для ОДНОГО
        покрытия.

        material_scalar_layers — (n_layers, n_scalar) — скалярные признаки
                                 материала для каждого слоя, в порядке
                                 воздух → металл. n_layers может быть меньше
                                 self.n_layers_cap — недостающее паддится нулями.
        thickness_mm_layers    — (n_layers,) — толщины слоёв, мм, тот же порядок.

        Возвращает (grid_points,) — предсказанный RC(f), дБ.
        """
        n_layers = material_scalar_layers.shape[0]
        if n_layers > self.n_layers_cap:
            raise ValueError(
                f"Модель обучена на максимум {self.n_layers_cap} слоёв, "
                f"получено {n_layers}. Переобучите модель с бОльшим паддингом "
                f"(max_layers в dataset_export.py), либо ограничьте запрос.")

        n_scalar = len(self.scalar_feature_names)
        mat_padded = np.zeros((self.n_layers_cap, n_scalar))
        th_padded = np.zeros(self.n_layers_cap)
        mask_padded = np.zeros(self.n_layers_cap)
        mat_padded[:n_layers] = material_scalar_layers
        th_padded[:n_layers] = thickness_mm_layers
        mask_padded[:n_layers] = 1.0

        X = np.concatenate([mat_padded.flatten(), th_padded, mask_padded])[None, :]
        return self.regressor.predict(X)[0]

    def predict_max_rc(self, material_scalar_layers: np.ndarray,
                       thickness_mm_layers: np.ndarray) -> float:
        """Удобный шорткат: сразу max(RC(f)) вместо всей кривой."""
        return float(np.max(self.predict_curve(material_scalar_layers,
                                                thickness_mm_layers)))

    # ── Сохранение / загрузка ────────────────────────────────────────────

    def save(self, output_dir: str) -> None:
        import joblib
        os.makedirs(output_dir, exist_ok=True)
        joblib.dump(self.regressor, os.path.join(output_dir, MODEL_FILENAME))
        meta = {
            "canon_freq_hz": self.canon_freq_hz.tolist(),
            "n_layers_cap": self.n_layers_cap,
            "scalar_feature_names": self.scalar_feature_names,
        }
        with open(os.path.join(output_dir, META_FILENAME), "w", encoding="utf-8") as f:
            json.dump(meta, f, ensure_ascii=False, indent=2)
        log.info(f"Модель сохранена: {output_dir}")

    @classmethod
    def load(cls, model_dir: str) -> "SurrogateModel":
        import joblib
        regressor = joblib.load(os.path.join(model_dir, MODEL_FILENAME))
        with open(os.path.join(model_dir, META_FILENAME), encoding="utf-8") as f:
            meta = json.load(f)
        return cls(regressor, np.array(meta["canon_freq_hz"]),
                   meta["n_layers_cap"], meta["scalar_feature_names"])


# ─────────────────────────────────────────────────────────────────────────────
# Интеграция с physics.py — обёртка суррогата под интерфейс tmm_reflection
# ─────────────────────────────────────────────────────────────────────────────

def material_scalar_features(er, eim, mr, mim) -> np.ndarray:
    """
    Скалярные признаки материала — то же самое, что dataset_export.py
    считает при экспорте (сознательно продублировано здесь, а не
    импортировано, чтобы ml_surrogate.py оставался самодостаточным модулем
    инференса, не зависящим от database.py/dataset_export.py).

    Порядок признаков ДОЛЖЕН совпадать с dataset_export._SCALAR_FEATURE_NAMES:
    er_mean, er_std, eim_mean, eim_std, mr_mean, mr_std, mim_mean, mim_std,
    loss_tangent_mean, er_slope.
    """
    er, eim, mr, mim = (np.asarray(x, dtype=float) for x in (er, eim, mr, mim))
    with np.errstate(divide="ignore", invalid="ignore"):
        loss_tangent = np.where(er != 0, eim / er, 0.0)
    return np.array([
        np.mean(er), np.std(er), np.mean(eim), np.std(eim),
        np.mean(mr), np.std(mr), np.mean(mim), np.std(mim),
        np.mean(loss_tangent),
        (er[-1] - er[0]) / max(len(er) - 1, 1),
    ])


def build_rc_evaluator(model: "SurrogateModel", material_datas: list):
    """
    Обернуть обученную SurrogateModel в функцию с той же сигнатурой, что и
    physics.tmm_reflection: evaluator(freqs, sequence, hs, mat_arrays) -> RC.

    Это позволяет подставить суррогат вместо честного TMM ВНУТРИ поиска
    NSGA-II/DE (см. physics.nsga2_optimise(..., rc_evaluator=...)) без
    изменения остального кода оптимизатора — интерфейс идентичен.

    ВАЖНО: это ускоряет только сам ПОИСК. worker.py всегда пересчитывает
    max_rc честным TMM для итоговых конфигураций перед сохранением в БД —
    ошибки суррогата не могут испортить сохранённые данные, только качество
    и скорость поиска.

    material_datas — список {'freq','er','eim','mr','mim'} материалов
                     ТЕКУЩЕГО прогона, в том же порядке, что использует
                     sequence (1-based индексы, как всюду в physics.py).
                     Скалярные признаки считаются один раз при создании
                     evaluator'а и кешируются — иначе смысл ускорения теряется.

    Предсказанная кривая интерполируется с канонической сетки модели
    (model.canon_freq_hz) на сетку freqs, переданную вызывающим кодом — так
    evaluator можно использовать, даже если материалы текущего прогона
    измерены на сетке, отличной от той, на которой обучалась модель
    (в пределах пересечения диапазонов; вне его np.interp экстраполирует
    константой, результат на краях может быть недостоверным).
    """
    feature_cache: dict[int, np.ndarray] = {}

    def _features_for(material_index_0based: int) -> np.ndarray:
        if material_index_0based not in feature_cache:
            m = material_datas[material_index_0based]
            feature_cache[material_index_0based] = material_scalar_features(
                m["er"], m["eim"], m["mr"], m["mim"])
        return feature_cache[material_index_0based]

    def evaluator(freqs: np.ndarray, sequence: tuple, hs, mat_arrays) -> np.ndarray:
        mat_scalar_layers = np.stack([_features_for(mi - 1) for mi in sequence])
        thickness_mm_layers = np.array([h * 1000.0 for h in hs])
        rc_on_canon = model.predict_curve(mat_scalar_layers, thickness_mm_layers)
        return np.interp(freqs, model.canon_freq_hz, rc_on_canon)

    return evaluator


# ─────────────────────────────────────────────────────────────────────────────
# Оценка существующей модели на (возможно новом) датасете
# ─────────────────────────────────────────────────────────────────────────────

def evaluate_model(model: SurrogateModel, dataset: dict) -> dict:
    from sklearn.metrics import mean_absolute_error, r2_score

    model_grid = model.canon_freq_hz
    data_grid = dataset["canon_freq_hz"]
    if model_grid.shape != data_grid.shape or not np.allclose(model_grid, data_grid):
        raise ValueError(
            f"Каноническая сетка частот модели ({len(model_grid)} точек, "
            f"{model_grid[0]/1e9:.2f}-{model_grid[-1]/1e9:.2f} ГГц) не совпадает "
            f"с сеткой датасета ({len(data_grid)} точек, "
            f"{data_grid[0]/1e9:.2f}-{data_grid[-1]/1e9:.2f} ГГц) — предсказания "
            f"были бы бессмысленны. Экспортируйте датасет с теми же "
            f"--grid-points/--f-min/--f-max, что использовались при обучении.")

    X = _flatten_features(dataset["material_scalar"], dataset["thickness_mm"],
                          dataset["layer_mask"])
    y_true = dataset["rc_curve_db"]
    y_pred = model.regressor.predict(X)

    mae_curve = float(np.mean(mean_absolute_error(y_true, y_pred, multioutput="raw_values")))
    r2 = float(r2_score(y_true, y_pred, multioutput="variance_weighted"))
    mae_max_rc = float(mean_absolute_error(np.max(y_true, axis=1), np.max(y_pred, axis=1)))
    return {"n_samples": int(len(X)), "mae_curve_db_mean": round(mae_curve, 3),
           "r2_curve": round(r2, 3), "mae_max_rc_db": round(mae_max_rc, 3)}


# ─────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────

def _main():
    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s  %(levelname)-8s  %(message)s")
    p = argparse.ArgumentParser(description="Forward-суррогат RC(f)")
    sub = p.add_subparsers(dest="command", required=True)

    p_train = sub.add_parser("train", help="Обучить суррогат на экспортированном датасете")
    p_train.add_argument("--dataset", required=True, help="Папка с dataset_tensors.npz")
    p_train.add_argument("--output", required=True, help="Папка для сохранения модели")
    p_train.add_argument("--n-estimators", type=int, default=200)
    p_train.add_argument("--max-depth", type=int, default=None)
    p_train.add_argument("--test-size", type=float, default=0.2)

    p_eval = sub.add_parser("eval", help="Оценить сохранённую модель на датасете")
    p_eval.add_argument("--dataset", required=True)
    p_eval.add_argument("--model", required=True, help="Папка с сохранённой моделью")

    args = p.parse_args()

    if args.command == "train":
        dataset = load_dataset(args.dataset)
        model, metrics = SurrogateModel.train(
            dataset, n_estimators=args.n_estimators,
            max_depth=args.max_depth, test_size=args.test_size)
        model.save(args.output)
        metrics["model_dir"] = args.output
        print(json.dumps(metrics, ensure_ascii=False, indent=2))

    elif args.command == "eval":
        dataset = load_dataset(args.dataset)
        model = SurrogateModel.load(args.model)
        metrics = evaluate_model(model, dataset)
        print(json.dumps(metrics, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    _main()


# ─────────────────────────────────────────────────────────────────────────────
# Примечание на будущее (когда датасет вырастет на порядки):
#
# Текущий RandomForest с плоским паддингом слоёв — намеренно простой baseline.
# Следующий шаг посложнее — sequence-encoder (небольшой 1D-CNN или transformer
# с self-attention по слоям), который:
#   • не зависит от максимального числа слоёв (n_layers_cap) — работает с
#     переменной длиной последовательности напрямую;
#   • явно моделирует физическую структуру (слой N ближе к металлу сильнее
#     влияет на многократные переотражения) через позиционное кодирование;
#   • сможет учиться на признаках сырого спектра материала (er(f)/eim(f)/...),
#     а не только на его скалярных сводках, что уже прямо доступно в
#     dataset_tensors.npz как material_curve.
# Менять стоит именно тогда, когда RandomForest перестанет расти в качестве
# с ростом датасета (плато R²) — это будет понятным сигналом, что не хватает
# емкости модели, а не данных.
# ─────────────────────────────────────────────────────────────────────────────
