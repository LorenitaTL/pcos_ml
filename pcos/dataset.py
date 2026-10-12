from pathlib import Path

from loguru import logger
import numpy as np
import pandas as pd

from pcos.config import RAW_DATA_DIR

RAW_DATA_FILE = RAW_DATA_DIR / "PCOS_data_without_infertility.xlsx"

ID_COLUMNS = ["Sl. No", "Patient File No."]
# Mínimos fisiológicos: por debajo se consideran errores de captura
MIN_BLOOD_PRESSURE = {"BP _Systolic (mmHg)": 60, "BP _Diastolic (mmHg)": 40}


def load_data(path: Path = RAW_DATA_FILE, sheet_name: str = "Full_new") -> pd.DataFrame:
    """Lee el Excel original y quita espacios sobrantes en los nombres de columna."""
    df = pd.read_excel(path, sheet_name=sheet_name)
    df.columns = df.columns.str.strip()
    logger.info(f"Datos cargados: {df.shape[0]} registros, {df.shape[1]} columnas")
    return df


def drop_non_predictive_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Descarta identificadores y columnas residuales del Excel (`Unnamed: *`)."""
    to_drop = [c for c in ID_COLUMNS if c in df.columns]
    to_drop += [c for c in df.columns if c.startswith("Unnamed")]
    logger.info(f"Columnas descartadas: {to_drop}")
    return df.drop(columns=to_drop)


def fix_numeric_types(df: pd.DataFrame) -> pd.DataFrame:
    """Convierte a número las columnas leídas como texto; lo no numérico queda como NaN."""
    df = df.copy()
    for col in df.select_dtypes(include=["object", "string"]).columns:
        original = df[col]
        df[col] = pd.to_numeric(original, errors="coerce")
        invalid = original[df[col].isna() & original.notna()]
        logger.info(
            f"{col}: {len(invalid)} valor(es) no numérico(s) -> NaN {invalid.unique().tolist()}"
        )
    return df


def drop_duplicates(df: pd.DataFrame) -> tuple[pd.DataFrame, int]:
    """Elimina registros duplicados y devuelve cuántos se quitaron."""
    n_duplicates = int(df.duplicated().sum())
    logger.info(f"Registros duplicados eliminados: {n_duplicates}")
    return df.drop_duplicates(), n_duplicates


def mask_impossible_blood_pressure(df: pd.DataFrame) -> pd.DataFrame:
    """Marca como NaN las presiones arteriales fisiológicamente imposibles."""
    df = df.copy()
    for col, minimum in MIN_BLOOD_PRESSURE.items():
        mask = df[col] < minimum
        logger.info(f"{col} < {minimum}: {int(mask.sum())} registro(s) -> NaN")
        df.loc[mask, col] = np.nan
    return df


def clean_data(df: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    """Aplica la limpieza completa, en el mismo orden que el notebook.

    Los atípicos no se eliminan y los faltantes se imputan después, dentro del
    pipeline del modelo, para evitar fuga de información.
    """
    n_rows = len(df)
    df = drop_non_predictive_columns(df)
    df = fix_numeric_types(df)
    df, n_duplicates = drop_duplicates(df)
    df = mask_impossible_blood_pressure(df)
    df = df.reset_index(drop=True)

    report = {
        "rows_raw": n_rows,
        "rows_clean": len(df),
        "duplicates_removed": n_duplicates,
    }
    logger.info(f"Limpieza terminada: {report}")
    return df, report
