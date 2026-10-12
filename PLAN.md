# Plan: migración del notebook a script con MLflow Tracking

Punto de partida: `notebooks/TC5061_Notebook_Baseline_TrujilloLorena.ipynb`, que ya contiene la limpieza, un pipeline `ColumnTransformer` + `LogisticRegression`, métricas, matriz de confusión y `log_model`. El trabajo consiste en reorganizar ese código, parametrizarlo y documentarlo.

## Entregables (checklist)

- [x] Script ejecutable con carga, preprocesamiento, entrenamiento y evaluación organizados en funciones
- [x] Notebook conservado solo para exploración de datos
- [x] Al menos 3 hiperparámetros y la semilla expuestos por línea de comandos, con valores por defecto
- [x] MLflow: `log_param`, al menos 2 `log_metric`, al menos 1 `log_artifact` y `mlflow.sklearn.log_model`
- [x] Al menos 5 corridas agrupadas en un experimento con nombre descriptivo
- [x] Comparación de corridas en la interfaz de MLflow
- [ ] Reproducibilidad documentada (mismos argumentos → mismo resultado)
- [x] `requirements.txt` con versiones fijadas
- [ ] Análisis breve (máximo 2 cuartillas) con capturas de la tabla comparativa y del detalle de una corrida con artefactos
- [ ] Repositorio individual de GitHub accesible para el profesor (script, `requirements.txt`, README breve)
- [ ] PDF `TC5061_Experimentacion_MLFlow_TrujilloLorena.pdf` (Arial/Times 11, interlineado 1.5, con enlace al repositorio y comando exacto)

---

## 1. Preparar el entorno ✅

- Entorno venv en `.venv/` con Python 3.12.8 (intérprete de Anaconda base). Se activa con `source .venv/bin/activate`.
- Se descartó Python 3.10: scipy 1.15.3, la última versión disponible para 3.10, no carga en macOS 27 (error de `dlopen` en `_spropack`), y scikit-learn depende de scipy. Por eso `requires-python` en `pyproject.toml` pasó de `~=3.10.0` a `>=3.10`.
- Dependencias instaladas y fijadas con versiones exactas en `requirements.txt`.
- Pandas quedó en la versión 3.0; al migrar el código hay que revisar cambios de comportamiento respecto al notebook (por ejemplo, el tipo `string` por defecto).

## 2. Pasar el código a módulos con funciones de una sola responsabilidad ✅

Se aprovecha la estructura de Cookiecutter que ya existe en `pcos/`. Los archivos actuales son solo la plantilla y se reemplazan.

| Archivo | Funciones | Viene de la celda |
|---|---|---|
| `pcos/dataset.py` | `load_data(path)`, `clean_data(df)` (descartar IDs, convertir tipos, quitar duplicados, presiones imposibles, outliers) | 3 y 16 |
| `pcos/features.py` | `split_data(df, test_size, seed)`, `build_preprocessor(cols_num, cols_cat)` | 19 |
| `pcos/modeling/train.py` | `parse_args()`, `build_model(...)`, `evaluate(model, X_test, y_test)`, `plot_confusion_matrix(...)`, `main()` con MLflow | 19, 21, 22 y 25 |

Ejecución:

```bash
python -m pcos.modeling.train
```

## 3. Parametrizar con argparse ✅

Todos los argumentos tienen valor por defecto, así que el script corre sin pasar nada.

| Argumento | Descripción | Valor por defecto |
|---|---|---|
| `--C` | Inverso de la fuerza de regularización | `1.0` |
| `--penalty` | Tipo de regularización (`l1` / `l2`; `liblinear` acepta ambas) | `l2` |
| `--class-weight` | `none` / `balanced` (clases desbalanceadas, 67/33) | `none` |
| `--max-iter` | Iteraciones máximas del solver | `1000` |
| `--test-size` | Proporción del conjunto de prueba | `0.2` |
| `--seed` | Semilla aleatoria | `42` |
| `--experiment-name` | Nombre del experimento en MLflow | `pcos_logreg_hyperparams` |

## 4. Instrumentar MLflow dentro de `mlflow.start_run()` ✅

- **`log_param`:** cada hiperparámetro, la semilla, `n_train`, `n_test` y los duplicados eliminados.
- **`log_metric`:** `accuracy`, `precision`, `recall`, `f1`, `roc_auc`, `cv_accuracy_mean`, `cv_recall_mean` y `dummy_accuracy`.
- **`log_artifact`:** `confusion_matrix.png` y `metrics.json`, guardados en un directorio temporal y registrados con `log_artifact`. El notebook usa `log_figure`, que no es lo que pide la consigna.
- **Modelo:** `mlflow.sklearn.log_model` con `signature` e `input_example`, como en el notebook.

## 5. Reproducibilidad (verificada; falta documentarla en el README)

- La misma `seed` se pasa a `train_test_split`, `StratifiedKFold` y `LogisticRegression`, y también se llama `np.random.seed(seed)`.
- Verificación: correr dos veces con los mismos argumentos y confirmar que las métricas son idénticas. El resultado se documenta en el README.

## 6. Correr al menos 5 configuraciones en un solo experimento ✅

Se agrega un `make experiments` (o `run_experiments.sh`) con las siguientes corridas:

| Corrida | C | penalty | class_weight | Propósito |
|---|---|---|---|---|
| 1 | 1.0 | l2 | none | Baseline |
| 2 | 0.01 | l2 | none | Regularización fuerte |
| 3 | 100 | l2 | none | Regularización débil |
| 4 | 0.1 | l1 | none | Efecto de L1 (selección de variables) |
| 5 | 1.0 | l2 | balanced | Efecto del balanceo de clases |
| 6 | 0.1 | l1 | balanced | Combinación L1 + balanceo |
| 7 | 1.0 | l2 | none | Repetición de la corrida 1 (prueba de reproducibilidad) |

Con este diseño se puede aislar el efecto de cada hiperparámetro.

## 7. Dejar el notebook solo para exploración ✅

- Eliminar las celdas 18 a 28 (entrenamiento, evaluación y registro en MLflow).
- Conservar carga, diccionario de datos, EDA y limpieza.
- Corregir la ruta de lectura a `../data/raw/PCOS_data_without_infertility.xlsx`.
- Agregar una nota que apunte a `pcos/modeling/train.py`.

## 8. Capturas de MLflow ✅

- Capturas guardadas en `reports/figures/`: tabla de corridas, vista Compare (2 partes), coordenadas paralelas, detalle de una corrida con sus artefactos y tabla de reproducibilidad.

## 9. README y repositorio de GitHub

- Reescribir el README (breve): instalación, comando exacto de ejecución, `make experiments`, `mlflow ui`, y cómo se fija la semilla.
- Decidir si el dataset (`data/raw/*.xlsx`, 125 KB, público en Kaggle) se incluye en el repositorio: hoy `.gitignore` excluye toda la carpeta `data/`, y sin el archivo el profesor no puede ejecutar el script.
- Inicializar git, primer commit y publicar el repositorio individual en GitHub (cuenta personal `LorenitaTL`).
- Verificar que el profesor tenga acceso (repositorio público, o privado con el profesor como colaborador).
- Debe contener al menos: el script de entrenamiento, `requirements.txt` y el README.

## 10. Documento PDF

- Nombre: `TC5061_Experimentacion_MLFlow_TrujilloLorena.pdf`.
- Formato: Arial o Times New Roman 11, interlineado 1.5, máximo 2 cuartillas de texto (sin contar capturas ni anexos).
- Contenido obligatorio:
  - Enlace al repositorio de GitHub.
  - Comando exacto de ejecución del script.
  - Análisis: mejor configuración, hiperparámetro más influyente y limitaciones.
  - Captura de la tabla comparativa de corridas en MLflow.
  - Captura del detalle de una corrida con sus artefactos.
- Contenido recomendado: reproducibilidad (semilla y captura de repro_A / repro_B / repro_C) y dependencias.
- Conclusiones del experimento para el análisis:
  - **Hiperparámetro más influyente:** `C` para accuracy/F1 (C=0.1 mejora al baseline con L1 o L2); `class_weight` para recall (balanced sube cv_recall de 0.814 a 0.876 con L2 y de 0.859 a 0.904 con L1). L1 vs L2 a C=0.1 empatan en validación cruzada.
  - **Mejor configuración para tamizaje:** C=0.1, L1, balanced (recall 0.917 en prueba, 0.904 en CV; cv_f1 0.856).
  - **Limitaciones:** dataset pequeño (541 registros, 109 en prueba); la semilla del split cambia F1 en ~5 puntos (0.831 vs 0.784), del mismo orden que las diferencias entre configuraciones; un solo tipo de modelo; búsqueda manual de hiperparámetros; eje C lineal en el gráfico de MLflow.

---

## Decisiones pendientes

1. ~~**Versión de Python**~~: resuelto, Python 3.12 (ver paso 1).
2. ~~**Ubicación del script**~~: resuelto, `pcos/modeling/train.py`.
