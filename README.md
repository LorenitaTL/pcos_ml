# Clasificación de PCOS con MLflow Tracking

Clasificación binaria para predecir si una paciente tiene síndrome de ovario poliquístico (PCOS)
a partir de datos clínicos, hormonales y de ultrasonido, con una regresión logística. El flujo del
notebook original se migró a un script parametrizable desde la línea de comandos que registra cada
corrida en MLflow para comparar configuraciones de hiperparámetros.

**Curso:** TC5061 · **Alumna:** Lorena Trujillo Landeros (A01797858)

## Requisitos

- Python 3.12 (probado con 3.12.8)
- Dataset: `data/raw/PCOS_data_without_infertility.xlsx` (incluido; fuente: [Kaggle](https://www.kaggle.com/datasets/prasoonkottarathil/polycystic-ovary-syndrome-pcos))

## Instalación

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
```

## Ejecución

Con valores por defecto:

```bash
python -m pcos.modeling.train
```

Con otros hiperparámetros:

```bash
python -m pcos.modeling.train --C 0.1 --penalty l1 --class-weight balanced --seed 42
```

| Argumento | Descripción | Default |
|---|---|---|
| `--C` | Inverso de la fuerza de regularización (menor = más regularización) | `1.0` |
| `--penalty` | Tipo de regularización: `l1` o `l2` | `l2` |
| `--class-weight` | `balanced` pondera las clases de forma inversa a su frecuencia; `none` no las pondera | `none` |
| `--max-iter` | Iteraciones máximas del optimizador | `1000` |
| `--test-size` | Proporción de datos reservada para prueba | `0.2` |
| `--seed` | Semilla aleatoria | `42` |
| `--experiment-name` | Experimento de MLflow donde se agrupa la corrida | `pcos_logreg_hyperparams` |
| `--run-name` | Nombre de la corrida en MLflow | generado con los hiperparámetros |

Ver todas las opciones: `python -m pcos.modeling.train --help`

## Experimentos y MLflow

```bash
make experiments    # corre las 7 configuraciones comparadas
make mlflow-ui      # abre la interfaz en http://127.0.0.1:5000
```

Cada corrida registra sus hiperparámetros (`log_param`), las métricas en prueba y en validación
cruzada de 5 pliegues (`log_metric`), la matriz de confusión y un `metrics.json` (`log_artifact`), y
el pipeline entrenado (`mlflow.sklearn.log_model`). El historial se guarda en `mlflow.db` y `mlruns/`.

## Reproducibilidad

La semilla (`--seed`) se pasa como `random_state` a la división entrenamiento/prueba, a la
validación cruzada y a la regresión logística, y fija además las semillas globales de `random` y
`numpy`. Dos ejecuciones con los mismos argumentos producen métricas idénticas; cambiar solo la
semilla las modifica. Las versiones exactas de las dependencias están en `requirements.txt`.

## Estructura

| Archivo | Contenido |
|---|---|
| `pcos/dataset.py` | Carga del Excel y limpieza de datos |
| `pcos/features.py` | Separación entrenamiento/prueba y preprocesamiento (imputación, escalado, one-hot) |
| `pcos/modeling/train.py` | Entrenamiento, evaluación y registro en MLflow |
| `notebooks/` | Exploración de datos y justificación de la limpieza |
| `reports/figures/` | Capturas de MLflow (comparación de corridas, artefactos, reproducibilidad) |
