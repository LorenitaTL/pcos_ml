import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

TARGET = "PCOS (Y/N)"
CATEGORICAL_COLUMNS = ["Blood Group", "Cycle(R/I)"]


def split_features_target(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
    """Separa las variables predictoras (X) de la variable objetivo (y)."""
    return df.drop(columns=[TARGET]), df[TARGET]


def split_data(X: pd.DataFrame, y: pd.Series, test_size: float, seed: int):
    """Divide en entrenamiento y prueba conservando la proporción de clases."""
    return train_test_split(X, y, test_size=test_size, random_state=seed, stratify=y)


def build_preprocessor(columns: list[str]) -> ColumnTransformer:
    """Numéricas: imputa con la mediana y escala. Categóricas: imputa con la moda y one-hot."""
    categorical = [c for c in columns if c in CATEGORICAL_COLUMNS]
    numeric = [c for c in columns if c not in CATEGORICAL_COLUMNS]

    numeric_pipeline = Pipeline(
        [("imputer", SimpleImputer(strategy="median")), ("scaler", StandardScaler())]
    )
    categorical_pipeline = Pipeline(
        [
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("onehot", OneHotEncoder(handle_unknown="ignore")),
        ]
    )
    return ColumnTransformer(
        [("num", numeric_pipeline, numeric), ("cat", categorical_pipeline, categorical)]
    )
