"""
Download and persist the UCI AI4I 2020 Predictive Maintenance Dataset (ID: 601).
Strictly validates features and stores local CSV in data/ai4i2020.csv.
"""

import os
import io
import zipfile
import requests
import pandas as pd

DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data")
CSV_PATH = os.path.join(DATA_DIR, "ai4i2020.csv")

UCI_DATASET_ID = 601
FALLBACK_ZIP_URL = "https://archive.ics.uci.edu/static/public/601/ai4i+2020+predictive+maintenance+dataset.zip"

EXPECTED_PREDICTORS = [
    "Type",
    "Air temperature [K]",
    "Process temperature [K]",
    "Rotational speed [rpm]",
    "Torque [Nm]",
    "Tool wear [min]"
]
TARGET_COL = "Machine failure"

# Mapping from alternative or unitless column names to standard required names
COLUMN_MAPPING = {
    "Air temperature": "Air temperature [K]",
    "Process temperature": "Process temperature [K]",
    "Rotational speed": "Rotational speed [rpm]",
    "Torque": "Torque [Nm]",
    "Tool wear": "Tool wear [min]",
    "Air_temperature": "Air temperature [K]",
    "Process_temperature": "Process temperature [K]",
    "Rotational_speed": "Rotational speed [rpm]",
    "Tool_wear": "Tool wear [min]"
}


def download_via_ucimlrepo() -> pd.DataFrame:
    """Fetch dataset using official ucimlrepo package."""
    from ucimlrepo import fetch_ucirepo
    print(f"Fetching dataset id={UCI_DATASET_ID} from ucimlrepo...")
    dataset = fetch_ucirepo(id=UCI_DATASET_ID)
    
    # Extract features and targets
    X = dataset.data.features.copy()
    y = dataset.data.targets.copy()
    
    # Merge into a single dataframe for local persistence
    df = pd.concat([X, y], axis=1)
    return df


def download_via_fallback_url() -> pd.DataFrame:
    """Fallback download directly from UCI archive zip."""
    print(f"Fallback: Downloading from official UCI static zip: {FALLBACK_ZIP_URL}...")
    headers = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64)"}
    resp = requests.get(FALLBACK_ZIP_URL, headers=headers, timeout=30)
    resp.raise_for_status()
    
    with zipfile.ZipFile(io.BytesIO(resp.content)) as z:
        csv_filename = [f for f in z.namelist() if f.endswith(".csv")][0]
        with z.open(csv_filename) as f:
            df = pd.read_csv(f)
    return df


def standardize_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Standardize column names to include units matching the official spec."""
    rename_dict = {}
    for col in df.columns:
        if col in COLUMN_MAPPING:
            rename_dict[col] = COLUMN_MAPPING[col]
        elif col.lower() == "machine failure":
            rename_dict[col] = TARGET_COL
    if rename_dict:
        df = df.rename(columns=rename_dict)
    return df


def download_and_persist_dataset(force: bool = False) -> str:
    """
    Downloads and verifies AI4I 2020 dataset, saving to data/ai4i2020.csv.
    """
    os.makedirs(DATA_DIR, exist_ok=True)

    if os.path.exists(CSV_PATH) and not force:
        print(f"Found existing dataset at: {CSV_PATH}")
        df = pd.read_csv(CSV_PATH)
        df = standardize_columns(df)
        df.to_csv(CSV_PATH, index=False)
    else:
        try:
            df = download_via_ucimlrepo()
            print(f"Successfully downloaded via ucimlrepo ({len(df)} rows, {df.shape[1]} cols).")
        except Exception as e:
            print(f"ucimlrepo download failed ({e}). Trying fallback URL...")
            try:
                df = download_via_fallback_url()
                print(f"Successfully downloaded via fallback ({len(df)} rows, {df.shape[1]} cols).")
            except Exception as e2:
                raise RuntimeError(
                    f"Failed to download AI4I 2020 dataset via both ucimlrepo and fallback URL: {e2}"
                )

        df = standardize_columns(df)
        df.to_csv(CSV_PATH, index=False)
        print(f"Dataset persisted locally to {CSV_PATH}")

    # Validate schema
    missing_predictors = [col for col in EXPECTED_PREDICTORS if col not in df.columns]
    if missing_predictors:
        raise ValueError(f"Dataset is missing expected predictors: {missing_predictors}")
    
    if TARGET_COL not in df.columns:
        raise ValueError(f"Dataset is missing target column: {TARGET_COL}")

    print(f"Validation Passed: Found all {len(EXPECTED_PREDICTORS)} predictors and target '{TARGET_COL}'.")
    print(f"Dataset Shape: {df.shape}")
    dist = df[TARGET_COL].value_counts(normalize=True).round(4) * 100
    print(f"Target Distribution:\n{dist.to_dict()}%")
    return CSV_PATH


if __name__ == "__main__":
    download_and_persist_dataset(force=False)
