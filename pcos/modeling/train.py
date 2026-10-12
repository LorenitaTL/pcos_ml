"""Entrena y evalúa una regresión logística para PCOS y registra la corrida en MLflow.

Uso:
    python -m pcos.modeling.train                      # valores por defecto
    python -m pcos.modeling.train --C 0.1 --penalty l1 --class-weight balanced --seed 7
"""

import argparse
import json
from pathlib import Path
import random
import tempfile

from loguru import logger
from matplotlib.figure import Figure
import mlflow
from mlflow.models import infer_signature
import mlflow.sklearn
import numpy as np
import pandas as pd
from sklearn.dummy import DummyClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.pipeline import Pipeline

from pcos.config import MLFLOW_ARTIFACTS_URI, MLFLOW_TRACKING_URI
from pcos.dataset import clean_data, load_data
from pcos.features import build_preprocessor, split_data, split_features_target

CLASS_NAMES = ["No PCOS", "PCOS"]
CV_FOLDS = 5
# scikit-learn >= 1.8 depreca `penalty`; el tipo de regularización se expresa con l1_ratio
L1_RATIO = {"l1": 1.0, "l2": 0.0}


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Hiperparámetros y semilla desde la línea de comandos, todos con valor por defecto."""
    parser = argparse.ArgumentParser(
        description="Entrena una regresión logística para PCOS."
    )
    parser.add_argument(
        "--C",
        type=float,
        default=1.0,
        help="Inverso de la fuerza de regularización (menor = más regularización).",
    )
    parser.add_argument(
        "--penalty", choices=["l1", "l2"], default="l2", help="Tipo de regularización."
    )
    parser.add_argument(
        "--class-weight",
        choices=["none", "balanced"],
        default="none",
        help="'balanced' pondera las clases de forma inversa a su frecuencia.",
    )
    parser.add_argument(
        "--max-iter", type=int, default=1000, help="Iteraciones máximas."
    )
    parser.add_argument(
        "--test-size",
        type=float,
        default=0.2,
        help="Proporción del conjunto de prueba.",
    )
    parser.add_argument("--seed", type=int, default=42, help="Semilla aleatoria.")
    parser.add_argument(
        "--experiment-name",
        default="pcos_logreg_hyperparams",
        help="Experimento de MLflow donde se agrupan las corridas.",
    )
    parser.add_argument(
        "--run-name",
        default=None,
        help="Nombre de la corrida (por defecto se arma con los hiperparámetros).",
    )
    return parser.parse_args(argv)


def set_seed(seed: int) -> None:
    """Fija las semillas globales. scikit-learn recibe la semilla vía `random_state`."""
    random.seed(seed)
    np.random.seed(seed)


def build_model(
    columns: list[str],
    C: float,
    penalty: str,
    class_weight: str,
    max_iter: int,
    seed: int,
) -> Pipeline:
    """Pipeline de preprocesamiento + regresión logística."""
    classifier = LogisticRegression(
        C=C,
        l1_ratio=L1_RATIO[penalty],
        class_weight=None if class_weight == "none" else class_weight,
        solver="liblinear",  # admite L1 y L2
        max_iter=max_iter,
        random_state=seed,
    )
    return Pipeline([("prep", build_preprocessor(columns)), ("logreg", classifier)])


def evaluate(
    model: Pipeline, X_test: pd.DataFrame, y_test: pd.Series
) -> tuple[dict, np.ndarray]:
    """Métricas en el conjunto de prueba. Devuelve también las predicciones."""
    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1]
    metrics = {
        "accuracy": accuracy_score(y_test, y_pred),
        "precision": precision_score(y_test, y_pred, zero_division=0),
        "recall": recall_score(y_test, y_pred, zero_division=0),
        "f1": f1_score(y_test, y_pred, zero_division=0),
        "roc_auc": roc_auc_score(y_test, y_proba),
    }
    return metrics, y_pred


def cross_validate(model: Pipeline, X: pd.DataFrame, y: pd.Series, seed: int) -> dict:
    """Validación cruzada estratificada: estima qué tan estable es el desempeño."""
    cv = StratifiedKFold(n_splits=CV_FOLDS, shuffle=True, random_state=seed)
    results = {}
    for scoring in ["accuracy", "recall", "f1"]:
        scores = cross_val_score(model, X, y, cv=cv, scoring=scoring)
        results[f"cv_{scoring}_mean"] = scores.mean()
        results[f"cv_{scoring}_std"] = scores.std()
    return results


def baseline_accuracy(X_train, y_train, X_test, y_test) -> float:
    """Accuracy de un clasificador que siempre predice la clase mayoritaria."""
    dummy = DummyClassifier(strategy="most_frequent").fit(X_train, y_train)
    return accuracy_score(y_test, dummy.predict(X_test))


def save_confusion_matrix(y_test: pd.Series, y_pred: np.ndarray, path: Path) -> None:
    """Guarda la matriz de confusión como imagen."""
    fig = Figure(figsize=(4.5, 4))  # sin pyplot: no abre ventanas al correr en terminal
    ax = fig.subplots()
    ConfusionMatrixDisplay(
        confusion_matrix(y_test, y_pred), display_labels=CLASS_NAMES
    ).plot(ax=ax, colorbar=False, cmap="Blues")
    ax.set_title("Matriz de confusión (prueba)")
    fig.tight_layout()
    fig.savefig(path, dpi=120)


def save_metrics(metrics: dict, path: Path) -> None:
    """Guarda las métricas en JSON."""
    path.write_text(
        json.dumps({k: round(float(v), 6) for k, v in metrics.items()}, indent=2)
    )


def setup_mlflow(experiment_name: str) -> None:
    """Apunta MLflow al historial del proyecto y crea el experimento si no existe."""
    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    if mlflow.get_experiment_by_name(experiment_name) is None:
        mlflow.create_experiment(
            experiment_name, artifact_location=MLFLOW_ARTIFACTS_URI
        )
    mlflow.set_experiment(experiment_name)


def main(argv: list[str] | None = None) -> dict:
    args = parse_args(argv)
    set_seed(args.seed)

    # 1. Datos
    df, cleaning_report = clean_data(load_data())
    X, y = split_features_target(df)
    X_train, X_test, y_train, y_test = split_data(X, y, args.test_size, args.seed)

    # 2. Entrenamiento
    model = build_model(
        list(X.columns),
        args.C,
        args.penalty,
        args.class_weight,
        args.max_iter,
        args.seed,
    )
    model.fit(X_train, y_train)

    # 3. Evaluación
    metrics, y_pred = evaluate(model, X_test, y_test)
    metrics.update(cross_validate(model, X, y, args.seed))
    metrics["dummy_accuracy"] = baseline_accuracy(X_train, y_train, X_test, y_test)

    # 4. Registro en MLflow
    setup_mlflow(args.experiment_name)
    run_name = (
        args.run_name
        or f"C={args.C}_{args.penalty}_cw={args.class_weight}_seed={args.seed}"
    )
    with mlflow.start_run(run_name=run_name) as run:
        # Parámetros: la "receta" de la corrida
        params = {
            "C": args.C,
            "penalty": args.penalty,
            "class_weight": args.class_weight,
            "max_iter": args.max_iter,
            "test_size": args.test_size,
            "seed": args.seed,
            "solver": "liblinear",
            "n_train": len(X_train),
            "n_test": len(X_test),
            "duplicates_removed": cleaning_report["duplicates_removed"],
        }
        for key, value in params.items():
            mlflow.log_param(key, value)

        # Métricas: el resultado de la corrida
        for key, value in metrics.items():
            mlflow.log_metric(key, value)

        # Artefactos: archivos que acompañan a la corrida
        with tempfile.TemporaryDirectory() as tmp:
            cm_path = Path(tmp) / "confusion_matrix.png"
            metrics_path = Path(tmp) / "metrics.json"
            save_confusion_matrix(y_test, y_pred, cm_path)
            save_metrics(metrics, metrics_path)
            mlflow.log_artifact(str(cm_path))
            mlflow.log_artifact(str(metrics_path))

        # Modelo: el pipeline entrenado, listo para cargarse y predecir
        X_signature = X_train.astype(
            "float64"
        )  # float: el esquema tolera valores nulos
        mlflow.sklearn.log_model(
            model,
            name="model",
            serialization_format="cloudpickle",
            signature=infer_signature(X_signature, model.predict(X_train)),
            input_example=X_signature.head(3),
        )

    summary = {
        k: round(float(metrics[k]), 3) for k in ["accuracy", "recall", "f1", "roc_auc"]
    }
    logger.success(f"Corrida '{run_name}' registrada ({run.info.run_id}): {summary}")
    return metrics


if __name__ == "__main__":
    main()
