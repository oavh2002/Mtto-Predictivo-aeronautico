"""
Pipeline reproducible para NASA C-MAPSS FD001.
Ejecutar desde la raíz: python src/pipeline.py
"""
from pathlib import Path
import json
import numpy as np
import pandas as pd
from scipy.stats import spearmanr, mannwhitneyu
from sklearn.model_selection import GroupKFold, GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.ensemble import RandomForestRegressor, HistGradientBoostingRegressor
from sklearn.linear_model import LinearRegression
import joblib

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data" / "raw"
PROC = ROOT / "data" / "processed"
MODELS = ROOT / "models"
REPORTS = ROOT / "reports"
COLS = ["unit_number", "time_in_cycles"] + [f"op_setting_{i}" for i in range(1,4)] + [f"sensor_{i}" for i in range(1,22)]
CONSTANT_SENSORS_FD001 = ["sensor_1", "sensor_5", "sensor_10", "sensor_16", "sensor_18", "sensor_19"]

def load_data():
    """Carga los tres archivos FD001 con separador de espacios compatible con pandas moderno."""
    paths = {n: RAW / f"{n}_FD001.txt" for n in ["train", "test", "RUL"]}
    missing = [str(p) for p in paths.values() if not p.exists()]
    if missing:
        raise FileNotFoundError("Faltan archivos NASA en data/raw/: " + ", ".join(missing))
    train = pd.read_csv(paths["train"], sep=r"\s+", header=None)
    test = pd.read_csv(paths["test"], sep=r"\s+", header=None)
    rul = pd.read_csv(paths["RUL"], sep=r"\s+", header=None)
    # Los archivos suelen incluir dos columnas finales vacías por espacios; se descartan.
    train = train.iloc[:, :26].copy()
    test = test.iloc[:, :26].copy()
    train.columns = COLS
    test.columns = COLS
    rul = rul.iloc[:, 0].rename("RUL").to_frame()
    if len(test.groupby("unit_number")) != len(rul):
        raise ValueError("La cantidad de motores en test no coincide con las filas de RUL_FD001.")
    return train, test, rul

def add_targets(train, cap=125):
    """Calcula RUL por motor y agrega RUL limitada a 125 ciclos."""
    out = train.copy()
    last = out.groupby("unit_number")["time_in_cycles"].max()
    out["RUL_unbounded"] = out["unit_number"].map(last) - out["time_in_cycles"]
    out["RUL"] = out["RUL_unbounded"].clip(upper=cap)
    return out

def etl(train, test):
    """Audita calidad; elimina duplicados exactos y sensores constantes en train."""
    audit = {}
    for name, df in [("train", train), ("test", test)]:
        audit[name] = {
            "shape": list(df.shape),
            "missing_by_column": df.isna().sum().to_dict(),
            "duplicate_rows": int(df.duplicated().sum())
        }
    train = train.drop_duplicates().copy()
    test = test.drop_duplicates().copy()
    sensor_cols = [c for c in train.columns if c.startswith("sensor_")]
    constant = [c for c in sensor_cols if train[c].nunique(dropna=True) <= 1]
    # Se retiran de ambos conjuntos para mantener idénticas las variables predictoras.
    retained = [c for c in sensor_cols if c not in constant]
    audit["constant_sensors_removed"] = constant
    # Los outliers se detectan y documentan, pero no se eliminan: pueden ser degradación real.
    outlier_counts = {}
    for c in retained:
        q1, q3 = train[c].quantile([.25, .75])
        iqr = q3 - q1
        lo, hi = q1 - 1.5 * iqr, q3 + 1.5 * iqr
        outlier_counts[c] = int(((train[c] < lo) | (train[c] > hi)).sum())
    audit["iqr_outlier_counts_train"] = outlier_counts
    audit["unit_conversion"] = (
        "No se convierten unidades: C-MAPSS entrega settings y sensores en escalas "
        "propias del simulador; no se proporcionan aquí unidades SI verificadas. "
        "No se inventan conversiones. La estandarización se aplica dentro de pipelines "
        "solo cuando el modelo lo requiere."
    )
    return train, test, retained, audit

def metrics(y_true, y_pred):
    return {
        "MAE": float(mean_absolute_error(y_true, y_pred)),
        "RMSE": float(mean_squared_error(y_true, y_pred) ** .5),
        "R2": float(r2_score(y_true, y_pred)),
        "NASA_score": nasa_score(y_true, y_pred)
    }

def nasa_score(y_true, y_pred):
    """Score asimétrico NASA: penaliza más la sobreestimación de RUL."""
    d = np.asarray(y_pred) - np.asarray(y_true)
    terms = np.where(d < 0, np.exp(-d / 13.0) - 1.0, np.exp(d / 10.0) - 1.0)
    return float(np.sum(terms))

def run():
    PROC.mkdir(parents=True, exist_ok=True); MODELS.mkdir(parents=True, exist_ok=True); REPORTS.mkdir(parents=True, exist_ok=True)
    train0, test0, rul = load_data()
    train0, test0, features, audit = etl(train0, test0)
    train = add_targets(train0)
    test = test0.copy()
    X, y, groups = train[features], train["RUL"], train["unit_number"]
    # Grupo externo reservado para validación: motores completos, sin compartir unidades.
    from sklearn.model_selection import GroupShuffleSplit
    splitter = GroupShuffleSplit(n_splits=1, test_size=.20, random_state=42)
    tr_idx, va_idx = next(splitter.split(X, y, groups))
    Xtr, Xva, ytr, yva = X.iloc[tr_idx], X.iloc[va_idx], y.iloc[tr_idx], y.iloc[va_idx]
    gtr = groups.iloc[tr_idx]
    models = {
        "LinearRegression": Pipeline([("imputer", SimpleImputer(strategy="median")), ("scale", StandardScaler()), ("model", LinearRegression())]),
        "RandomForest": Pipeline([("imputer", SimpleImputer(strategy="median")), ("model", RandomForestRegressor(random_state=42, n_jobs=-1))]),
        "HistGradientBoosting": Pipeline([("imputer", SimpleImputer(strategy="median")), ("model", HistGradientBoostingRegressor(random_state=42))])
    }
    # Ajuste moderado de RF con CV agrupada únicamente dentro del conjunto de entrenamiento.
    param_grid = {"model__n_estimators": [150, 300], "model__max_depth": [None, 12], "model__min_samples_leaf": [1, 3]}
    cv = GroupKFold(n_splits=5)
    search = GridSearchCV(models["RandomForest"], param_grid, scoring="neg_root_mean_squared_error",
                          cv=cv, n_jobs=-1, refit=True)
    search.fit(Xtr, ytr, groups=gtr)
    fitted = {"RandomForest_tuned": search.best_estimator_}
    for name, model in models.items():
        if name == "RandomForest":
            continue
        model.fit(Xtr, ytr)
        fitted[name] = model
    val_results = {}
    for name, model in fitted.items():
        pred = np.clip(model.predict(Xva), 0, 125)
        val_results[name] = metrics(yva, pred)
    best_name = min(val_results, key=lambda k: val_results[k]["RMSE"])
    best = fitted[best_name]
    joblib.dump({"model": best, "features": features, "name": best_name}, MODELS / "fd001_rul_model.joblib")

    # Evaluación oficial: predicción de la última fila de cada motor de test.
    last_test = test.sort_values(["unit_number", "time_in_cycles"]).groupby("unit_number").tail(1).sort_values("unit_number")
    Xtest = last_test[features]
    ytest = rul["RUL"].to_numpy()
    pred_test = np.clip(best.predict(Xtest), 0, 125)
    test_metrics = metrics(ytest, pred_test)
    pred_frame = pd.DataFrame({"unit_number": last_test["unit_number"].to_numpy(),
                               "RUL_real": ytest, "RUL_predicha": pred_test,
                               "error": pred_test-ytest})
    pred_frame.to_csv(PROC / "predicciones_test_fd001.csv", index=False)
    train.to_csv(PROC / "train_fd001_processed.csv", index=False)
    test.to_csv(PROC / "test_fd001_processed.csv", index=False)
    report = {"audit": audit, "features": features, "best_model": best_name,
              "best_params": search.best_params_, "validation": val_results,
              "official_test": test_metrics, "n_train_engines": int(train.unit_number.nunique()),
              "n_validation_engines": int(groups.iloc[va_idx].nunique())}
    (REPORTS / "metrics.json").write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(report, indent=2, ensure_ascii=False))
    return report

if __name__ == "__main__":
    run()
