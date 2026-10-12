"""Pruebas básicas de la limpieza de datos y de los argumentos del script.

Usan un DataFrame pequeño construido a mano, así no dependen del Excel original.
Ejecutar con: make test
"""

import numpy as np
import pandas as pd
import pytest

from pcos.dataset import clean_data
from pcos.features import TARGET
from pcos.modeling.train import parse_args


@pytest.fixture
def raw_df():
    """Mini dataset con los problemas que la limpieza debe corregir."""
    return pd.DataFrame(
        {
            "Sl. No": [1, 2, 3, 4],
            "Patient File No.": [10, 20, 30, 40],
            "Unnamed: 44": [np.nan] * 4,
            "AMH(ng/mL)": ["2.5", "a", "3.1", "4.0"],  # 'a' no es numérico
            "BP _Systolic (mmHg)": [110, 12, 120, 115],  # 12 es imposible
            "BP _Diastolic (mmHg)": [70, 80, 8, 75],  # 8 es imposible
            TARGET: [0, 1, 0, 1],
        }
    )


def test_clean_data_drops_identifier_columns(raw_df):
    df, _ = clean_data(raw_df)
    assert not {"Sl. No", "Patient File No.", "Unnamed: 44"} & set(df.columns)


def test_clean_data_converts_text_to_numeric(raw_df):
    df, _ = clean_data(raw_df)
    assert pd.api.types.is_numeric_dtype(df["AMH(ng/mL)"])
    assert df["AMH(ng/mL)"].isna().sum() == 1


def test_clean_data_masks_impossible_blood_pressure(raw_df):
    df, _ = clean_data(raw_df)
    assert df["BP _Systolic (mmHg)"].isna().sum() == 1
    assert df["BP _Diastolic (mmHg)"].isna().sum() == 1


def test_parse_args_defaults():
    args = parse_args([])
    assert (args.C, args.penalty, args.class_weight, args.seed) == (1.0, "l2", "none", 42)
