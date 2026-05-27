from __future__ import annotations

import logging
import time
import zipfile
from pathlib import Path
from typing import Any
from urllib.parse import parse_qs, unquote, urlparse

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import requests
from sklearn.ensemble import RandomForestClassifier
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.calibration import CalibratedClassifierCV
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import OrdinalEncoder
from sklearn.metrics import accuracy_score, f1_score, mean_absolute_error
from sklearn.model_selection import train_test_split
from sklearn.utils.class_weight import compute_sample_weight
from tqdm import tqdm

logger = logging.getLogger(__name__)
PROJECT_ROOT = Path(__file__).resolve().parents[1]

RANDOM_STATE = 42
URLS: dict[str, str] = {
    "investor_2021": (
        "https://finrafoundation.org/modal/node/1686"
        "?file=%2Fsites%2Ffinrafoundation%2Ffiles%2F2021-Inv-Data-and-Data-Info.zip"
    ),
    "state_2021": (
        "https://finrafoundation.org/modal/node/1686"
        "?file=%2Fsites%2Ffinrafoundation%2Ffiles%2F2021-SxS-Data-and-Data-Info.zip"
    ),
}


def _resolve_path(path_like: str | Path, default_relative: str) -> Path:
    """Resolve relative paths against project root for stable notebook execution."""
    path = Path(path_like) if path_like else PROJECT_ROOT / default_relative
    if not path.is_absolute():
        path = PROJECT_ROOT / path
    return path


def find_csv(folder: Path) -> Path:
    """Find the most likely main CSV in a folder tree.

    If multiple CSV files exist, return the largest one.
    """
    csvs = list(folder.rglob("*.csv"))
    if not csvs:
        raise FileNotFoundError(f"No se encontro ningun CSV en {folder}")
    return max(csvs, key=lambda f: f.stat().st_size)


def _download_with_retries(url: str, dest_zip: Path, max_retries: int = 3) -> None:
    """Download a file with retries and progress bar."""
    headers = {"User-Agent": "Mozilla/5.0"}
    last_error: Exception | None = None

    for attempt in range(1, max_retries + 1):
        try:
            with requests.get(url, stream=True, timeout=60, headers=headers) as response:
                response.raise_for_status()
                total = int(response.headers.get("content-length", 0))
                with tqdm(
                    total=total,
                    unit="B",
                    unit_scale=True,
                    desc=f"Downloading {dest_zip.name}",
                ) as bar:
                    with dest_zip.open("wb") as out:
                        for chunk in response.iter_content(chunk_size=1024 * 64):
                            if chunk:
                                out.write(chunk)
                                bar.update(len(chunk))
            return
        except Exception as exc:  # noqa: BLE001
            last_error = exc
            logger.warning(
                "Download attempt %s/%s failed for %s: %s",
                attempt,
                max_retries,
                url,
                exc,
            )
            if attempt < max_retries:
                time.sleep(5)

    raise RuntimeError(f"Failed downloading {url} after {max_retries} retries") from last_error


def _derive_direct_file_url(url: str) -> str | None:
    """Derive direct FINRA file URL from modal URL query parameter."""
    parsed = urlparse(url)
    query = parse_qs(parsed.query)
    file_values = query.get("file", [])
    if not file_values:
        return None
    file_path = unquote(file_values[0])
    if not file_path.startswith("/"):
        file_path = f"/{file_path}"
    return f"https://finrafoundation.org{file_path}"


def _ensure_valid_zip_download(url: str, dest_zip: Path) -> None:
    """Download ZIP with fallback to direct file URL when modal endpoint returns HTML."""
    _download_with_retries(url, dest_zip)
    if zipfile.is_zipfile(dest_zip):
        return

    logger.warning("Downloaded file from modal URL is not a valid ZIP: %s", dest_zip)
    direct_url = _derive_direct_file_url(url)
    if direct_url is None:
        raise zipfile.BadZipFile(f"File is not a zip file and no direct URL available: {dest_zip}")

    logger.info("Retrying with direct file URL: %s", direct_url)
    _download_with_retries(direct_url, dest_zip)
    if not zipfile.is_zipfile(dest_zip):
        raise zipfile.BadZipFile(f"File is not a zip file after direct download attempt: {dest_zip}")


def download_nfcs_data(raw_dir: str | Path = "data/raw") -> dict[str, Path]:
    """Download and extract NFCS datasets, then auto-detect CSV paths.

    Returns:
        Dict with resolved CSV paths for investor and state datasets.
    """
    raw_path = _resolve_path(raw_dir, "data/raw")
    raw_path.mkdir(parents=True, exist_ok=True)

    zip_targets = {
        "investor_2021": raw_path / "investor_2021.zip",
        "state_2021": raw_path / "state_2021.zip",
    }
    extract_targets = {
        "investor_2021": raw_path / "investor_2021",
        "state_2021": raw_path / "state_2021",
    }

    for key, url in URLS.items():
        zip_path = zip_targets[key]
        extract_dir = extract_targets[key]

        if not zip_path.exists():
            logger.info("Downloading %s to %s", key, zip_path)
            _ensure_valid_zip_download(url, zip_path)
        else:
            logger.info("Using cached ZIP: %s", zip_path)

        if not zipfile.is_zipfile(zip_path):
            logger.warning("Cached ZIP is invalid. Re-downloading: %s", zip_path)
            zip_path.unlink(missing_ok=True)
            _ensure_valid_zip_download(url, zip_path)

        if not extract_dir.exists() or not any(extract_dir.iterdir()):
            extract_dir.mkdir(parents=True, exist_ok=True)
            logger.info("Extracting %s to %s", zip_path.name, extract_dir)
            with zipfile.ZipFile(zip_path, "r") as zf:
                zf.extractall(extract_dir)
        else:
            logger.info("Using cached extraction: %s", extract_dir)

    investor_csv = find_csv(extract_targets["investor_2021"])
    state_csv = find_csv(extract_targets["state_2021"])

    return {
        "investor_csv": investor_csv,
        "state_csv": state_csv,
    }


def explore_nfcs_columns(investor_csv: str | Path, state_csv: str | Path) -> dict[str, Any]:
    """Explore real columns in both NFCS CSV files.

    This function reads only a small sample to infer schema and target candidates.
    """
    investor_csv = Path(investor_csv)
    state_csv = Path(state_csv)

    df_inv = pd.read_csv(investor_csv, nrows=5, low_memory=False)
    df_sxs = pd.read_csv(state_csv, nrows=5, low_memory=False)

    inv_cols = list(df_inv.columns)
    sxs_cols = list(df_sxs.columns)
    shared = sorted(set(inv_cols) & set(sxs_cols))
    only_inv = sorted(set(inv_cols) - set(sxs_cols))

    risk_candidates = [
        col
        for col in inv_cols
        if any(kw in col.upper() for kw in ["RISK", "J2", "J1", "TOLER"])
    ]

    if not risk_candidates:
        fallback_priority = ["J2", "J1", "B31", "G31", "B32", "G5", "G21"]
        risk_candidates = [col for col in fallback_priority if col in inv_cols]
        if risk_candidates:
            logger.warning(
                "No explicit risk keyword column found. Falling back to likely proxy column(s): %s",
                risk_candidates,
            )

    risk_candidate_values: dict[str, dict[Any, int]] = {}
    for col in risk_candidates:
        risk_candidate_values[col] = df_inv[col].value_counts(dropna=False).head(10).to_dict()

    logger.info("Investor columns: %s", len(inv_cols))
    logger.info("State columns: %s", len(sxs_cols))
    logger.info("Shared columns: %s", len(shared))
    logger.info("Only investor columns: %s", len(only_inv))
    logger.info("Risk candidates: %s", risk_candidates)

    return {
        "investor_columns": inv_cols,
        "state_columns": sxs_cols,
        "shared_columns": shared,
        "only_investor": only_inv,
        "risk_candidates": risk_candidates,
        "risk_candidate_values": risk_candidate_values,
    }


def _encode_features(
    df: pd.DataFrame,
    feature_cols: list[str],
    *,
    fit: bool,
    numeric_medians: pd.Series | None = None,
) -> tuple[pd.DataFrame, pd.Series]:
    """Prepare feature matrix with adaptive numeric/categorical handling."""
    X = df.reindex(columns=feature_cols).copy()

    # DESIGN: Coerce non-numeric survey codings when possible, but keep true categoricals.
    for col in X.columns:
        if X[col].dtype == object:
            coerced = pd.to_numeric(X[col], errors="coerce")
            if coerced.notna().mean() > 0.9:
                X[col] = coerced

    numeric_cols = X.select_dtypes(include=[np.number, "bool"]).columns.tolist()
    categorical_cols = [c for c in X.columns if c not in numeric_cols]

    if fit:
        medians = X[numeric_cols].median(numeric_only=True)
    else:
        if numeric_medians is None:
            raise ValueError("numeric_medians must be provided when fit=False")
        medians = numeric_medians

    if numeric_cols:
        X[numeric_cols] = X[numeric_cols].fillna(medians)
    if categorical_cols:
        X[categorical_cols] = X[categorical_cols].fillna("MISSING").astype(str)

    X_enc = pd.get_dummies(X, drop_first=True)
    return X_enc, medians


def _score_risk_candidate(df_inv: pd.DataFrame, candidate: str) -> dict[str, float | str]:
    """Score potential target columns and prefer informative, non-degenerate proxies."""
    values = pd.to_numeric(df_inv[candidate], errors="coerce")
    valid = values[(values.notna()) & (values < 97)]

    if len(valid) == 0:
        return {
            "candidate": candidate,
            "coverage": 0.0,
            "n_unique_valid": 0.0,
            "entropy_norm": 0.0,
            "dominant_share": 1.0,
            "score": -1.0,
        }

    probs = valid.value_counts(normalize=True)
    n_unique = int(probs.shape[0])
    dominant_share = float(probs.max())

    if n_unique > 1:
        entropy = float(-(probs * np.log(probs + 1e-12)).sum())
        entropy_norm = float(entropy / np.log(n_unique))
    else:
        entropy_norm = 0.0

    coverage = float(len(valid) / len(df_inv))

    # DESIGN: Prefer proxies with broad support, multiple categories and lower dominance.
    score = coverage * entropy_norm * max(0.0, 1.0 - dominant_share)

    return {
        "candidate": candidate,
        "coverage": coverage,
        "n_unique_valid": float(n_unique),
        "entropy_norm": entropy_norm,
        "dominant_share": dominant_share,
        "score": score,
    }


def detect_target_column(
    df_investor: pd.DataFrame,
    risk_candidates: list[str] | None = None,
    target_col_override: str | None = None,
) -> tuple[str, pd.DataFrame]:
    """Detect risk target proxy with adaptive scoring and optional manual override."""
    if risk_candidates is None or not risk_candidates:
        inferred = [
            col
            for col in df_investor.columns
            if any(kw in col.upper() for kw in ["RISK", "J2", "J1", "TOLER"])
        ]
        if not inferred:
            fallback_priority = ["J2", "J1", "B31", "G31", "B32", "G5", "G21"]
            inferred = [col for col in fallback_priority if col in df_investor.columns]
        risk_candidates = inferred

    if not risk_candidates:
        raise ValueError("No risk candidate column found in investor dataset")

    candidate_scores = pd.DataFrame(
        [_score_risk_candidate(df_investor, c) for c in risk_candidates]
    ).sort_values("score", ascending=False)

    target_col = str(candidate_scores.iloc[0]["candidate"])

    if target_col_override is not None:
        if target_col_override not in risk_candidates:
            raise ValueError(
                f"target_col_override={target_col_override} is not in detected risk candidates: {risk_candidates}"
            )
        logger.warning("Using manual target override: %s", target_col_override)
        target_col = target_col_override

    return target_col, candidate_scores


def get_shared_features(
    df_investor: pd.DataFrame,
    df_state: pd.DataFrame,
    target_col: str,
) -> list[str]:
    """Return robust shared feature set excluding IDs/weights/target fields.

    DESIGN: Restrict to columns present in both datasets to avoid zero-filled transfer matrices.
    """
    exclude_patterns = [
        "ID",
        "NFCS",
        "WEIGHT",
        "WT",
        "STATEQ",
        "FINALWT",
        target_col.upper(),
    ]

    shared = set(df_investor.columns) & set(df_state.columns)
    features = [
        col
        for col in shared
        if not any(pat in col.upper() for pat in exclude_patterns)
    ]

    features = sorted(features)
    logger.info("Features compartidas disponibles: %s", len(features))
    logger.info("Features seleccionadas: %s", features)
    return features


def _infer_feature_types(
    df_reference: pd.DataFrame,
    feature_cols: list[str],
) -> tuple[list[str], list[str]]:
    """Infer numeric vs categorical columns robustly from reference dataset."""
    numeric_cols: list[str] = []
    categorical_cols: list[str] = []

    for col in feature_cols:
        series = df_reference[col]
        if pd.api.types.is_numeric_dtype(series):
            numeric_cols.append(col)
            continue

        coerced = pd.to_numeric(series, errors="coerce")
        parse_ratio = float(coerced.notna().mean())
        if parse_ratio >= 0.85:
            numeric_cols.append(col)
        else:
            categorical_cols.append(col)

    return numeric_cols, categorical_cols


def _normalize_missing_tokens(df: pd.DataFrame) -> pd.DataFrame:
    """Normalize common survey missing markers to NaN before preprocessing."""
    return df.replace({"": np.nan, " ": np.nan, "  ": np.nan, "NA": np.nan, "N/A": np.nan})


def build_preprocessor(
    feature_cols: list[str],
    df_reference: pd.DataFrame,
) -> tuple[ColumnTransformer, list[str]]:
    """Build one consistent preprocessor to fit on investor and transform state.

    DESIGN: Use a single fitted transformer to prevent schema drift during transfer.
    """
    numeric_cols, cat_cols = _infer_feature_types(df_reference, feature_cols)

    numeric_transformer = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
    ])

    categorical_transformer = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        (
            "encoder",
            OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1),
        ),
    ])

    preprocessor = ColumnTransformer(
        [
            ("num", numeric_transformer, numeric_cols),
            ("cat", categorical_transformer, cat_cols),
        ],
        remainder="drop",
    )

    ordered_cols = numeric_cols + cat_cols
    return preprocessor, ordered_cols


def clean_nfcs(
    investor_csv: str | Path,
    state_csv: str | Path,
    col_info: dict[str, Any],
    processed_dir: str | Path = "data/processed",
    target_col_override: str | None = None,
) -> dict[str, Any]:
    """Clean NFCS datasets in an adaptive way and save parquet outputs."""
    processed_path = _resolve_path(processed_dir, "data/processed")
    processed_path.mkdir(parents=True, exist_ok=True)

    df_inv = pd.read_csv(investor_csv, low_memory=False)
    df_sxs = pd.read_csv(state_csv, low_memory=False)

    risk_candidates = col_info.get("risk_candidates", [])
    if not risk_candidates:
        raise ValueError(
            "No risk candidate column found. Run explore_nfcs_columns and inspect candidates."
        )
    shared_columns = set(col_info.get("shared_columns", []))
    candidate_scores = []
    for c in risk_candidates:
        row = _score_risk_candidate(df_inv, c)
        row["is_shared_with_state"] = 1.0 if c in shared_columns else 0.0
        # DESIGN: Prefer proxies available in both datasets to improve transfer stability.
        row["transfer_score"] = float(row["score"]) + 0.35 * float(row["is_shared_with_state"])
        candidate_scores.append(row)

    candidate_scores_df = pd.DataFrame(candidate_scores).sort_values("transfer_score", ascending=False)

    # DESIGN: Choose candidate balancing information quality and transferability.
    target_col = str(candidate_scores_df.iloc[0]["candidate"])

    if target_col_override is not None:
        if target_col_override not in risk_candidates:
            raise ValueError(
                f"target_col_override={target_col_override} is not in detected risk candidates: {risk_candidates}"
            )
        logger.warning("Using manual target override: %s", target_col_override)
        target_col = target_col_override

    # DESIGN: NFCS uses high codes (97/98/99) for non-response. Keep only valid values.
    target_numeric = pd.to_numeric(df_inv[target_col], errors="coerce")
    valid_mask = target_numeric.notna() & (target_numeric < 97)
    df_inv = df_inv.loc[valid_mask].copy()
    df_inv[target_col] = target_numeric.loc[valid_mask]

    # DESIGN: Use ordinal range-based bins instead of quantile bins.
    # This preserves the real skewness of risk preference in the survey.
    risk_level = pd.cut(
        df_inv[target_col],
        bins=5,
        labels=[1, 2, 3, 4, 5],
        include_lowest=True,
    )

    # Fallback when values are degenerate and cut produces NaNs.
    if risk_level.isna().any():
        risk_level = pd.cut(
            df_inv[target_col].rank(method="first"),
            bins=5,
            labels=[1, 2, 3, 4, 5],
            include_lowest=True,
        )
    df_inv["risk_level"] = risk_level.astype(int)

    feature_candidates = get_shared_features(df_inv, df_sxs, target_col)
    if len(feature_candidates) < 3:
        raise ValueError(
            f"Solo {len(feature_candidates)} features compartidas entre datasets. "
            "Revisar nombres de columnas; posible problema de schema."
        )

    investor_clean = df_inv[["risk_level", *feature_candidates]].reset_index(drop=True)
    state_clean = df_sxs[feature_candidates].reset_index(drop=True)

    investor_clean_path = processed_path / "nfcs_investor_clean.parquet"
    state_clean_path = processed_path / "nfcs_state_clean.parquet"

    investor_clean.to_parquet(investor_clean_path, index=False)
    state_clean.to_parquet(state_clean_path, index=False)

    raw_target_distribution = (
        df_inv[target_col].value_counts(normalize=True).sort_index().rename("raw_target_pct")
    )
    risk_level_distribution = (
        df_inv["risk_level"].value_counts(normalize=True).sort_index().reindex([1, 2, 3, 4, 5], fill_value=0.0)
    )

    return {
        "target_col": target_col,
        "candidate_scores": candidate_scores_df,
        "feature_columns_raw": feature_candidates,
        "feature_columns_encoded": feature_candidates,
        "investor_shape": investor_clean.shape,
        "state_shape": state_clean.shape,
        "investor_clean_path": investor_clean_path,
        "state_clean_path": state_clean_path,
        "raw_target_distribution": raw_target_distribution,
        "risk_level_distribution": risk_level_distribution,
    }


def train_risk_model(
    investor_clean_path: str | Path,
    state_clean_path: str | Path,
    output_dir: str | Path = "data/outputs",
    processed_dir: str | Path = "data/processed",
    apply_prior_shift_correction: bool = True,
    compute_shap: bool = True,
    persist_artifacts: bool = True,
) -> dict[str, Any]:
    """Train XGBoost risk model, compare with RF, and transfer to state dataset."""
    output_path = _resolve_path(output_dir, "data/outputs")
    processed_path = _resolve_path(processed_dir, "data/processed")
    output_path.mkdir(parents=True, exist_ok=True)
    processed_path.mkdir(parents=True, exist_ok=True)

    investor_clean = pd.read_parquet(investor_clean_path)
    if "risk_level" not in investor_clean.columns:
        raise ValueError("Expected 'risk_level' in investor_clean dataset")

    state_clean = pd.read_parquet(state_clean_path)

    feature_cols = [c for c in investor_clean.columns if c != "risk_level" and c in state_clean.columns]
    if len(feature_cols) < 3:
        raise ValueError(
            f"Only {len(feature_cols)} shared features between investor/state clean datasets"
        )

    X_inv_raw = _normalize_missing_tokens(investor_clean[feature_cols].copy())
    X_state_raw_df = _normalize_missing_tokens(state_clean[feature_cols].copy())

    numeric_cols, _ = _infer_feature_types(X_inv_raw, feature_cols)
    for col in numeric_cols:
        X_inv_raw[col] = pd.to_numeric(X_inv_raw[col], errors="coerce")
        X_state_raw_df[col] = pd.to_numeric(X_state_raw_df[col], errors="coerce")

    preprocessor, ordered_cols = build_preprocessor(feature_cols, X_inv_raw)

    X_inv_proc = preprocessor.fit_transform(X_inv_raw)
    X_state_proc = preprocessor.transform(X_state_raw_df)

    X = pd.DataFrame(X_inv_proc, columns=ordered_cols, index=investor_clean.index)
    X_state_raw = pd.DataFrame(X_state_proc, columns=ordered_cols, index=state_clean.index)

    y = investor_clean["risk_level"].astype(int)

    # DESIGN: After adaptive binning, some classes (1..5) can be absent.
    # Encode only observed classes for XGBoost contiguous target requirement.
    label_encoder = LabelEncoder()
    y_encoded = label_encoder.fit_transform(y)

    X_train, X_temp, y_train, y_temp = train_test_split(
        X,
        y_encoded,
        test_size=0.30,
        stratify=y_encoded,
        random_state=RANDOM_STATE,
    )
    X_val, X_test, y_val, y_test = train_test_split(
        X_temp,
        y_temp,
        test_size=0.50,
        stratify=y_temp,
        random_state=RANDOM_STATE,
    )

    from xgboost import XGBClassifier

    xgb = XGBClassifier(
        objective="multi:softprob",
        num_class=len(label_encoder.classes_),
        n_estimators=500,
        max_depth=5,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=RANDOM_STATE,
        eval_metric="mlogloss",
        early_stopping_rounds=50,
    )
    sample_weight_train = compute_sample_weight(class_weight="balanced", y=y_train)
    xgb.fit(
        X_train,
        y_train,
        sample_weight=sample_weight_train,
        eval_set=[(X_val, y_val)],
        verbose=False,
    )

    rf = RandomForestClassifier(
        n_estimators=200,
        random_state=RANDOM_STATE,
        class_weight="balanced_subsample",
    )
    rf.fit(X_train, y_train)

    y_pred_xgb_enc = np.asarray(xgb.predict(X_test))
    if y_pred_xgb_enc.ndim > 1:
        y_pred_xgb_enc = y_pred_xgb_enc.argmax(axis=1)
    y_pred_xgb = label_encoder.inverse_transform(y_pred_xgb_enc)

    y_pred_rf_enc = rf.predict(X_test)
    y_pred_rf = label_encoder.inverse_transform(y_pred_rf_enc)

    y_test_labels = label_encoder.inverse_transform(y_test)

    metrics = {
        "Model": ["XGBoost", "RandomForest"],
        "Accuracy": [
            accuracy_score(y_test_labels, y_pred_xgb),
            accuracy_score(y_test_labels, y_pred_rf),
        ],
        "F1_macro": [
            f1_score(y_test_labels, y_pred_xgb, average="macro"),
            f1_score(y_test_labels, y_pred_rf, average="macro"),
        ],
        "MAE_ordinal": [
            mean_absolute_error(y_test_labels, y_pred_xgb),
            mean_absolute_error(y_test_labels, y_pred_rf),
        ],
    }
    comparison_df = pd.DataFrame(metrics)

    shap_path: Path | None = None
    if compute_shap:
        import shap

        explainer = shap.TreeExplainer(xgb)
        shap_values = explainer.shap_values(X_test)
        shap.summary_plot(shap_values, X_test, show=False)
        plt.title("Importancia de variables - nivel de riesgo")
        plt.tight_layout()
        shap_path = output_path / "shap_risk_model.png"
        plt.savefig(shap_path, dpi=150)
        plt.close()

    model_path: Path | None = None
    if persist_artifacts:
        model_path = output_path / "xgb_risk_model.pkl"
        joblib.dump(xgb, model_path)

    X_state = X_state_raw.reindex(columns=X_train.columns, fill_value=0)

    state_with_risk = X_state.copy()
    state_proba_raw = xgb.predict_proba(X_state)
    state_pred_raw_enc = state_proba_raw.argmax(axis=1)

    # DESIGN: Correct for label shift by aligning predicted class prior to investor prior.
    if apply_prior_shift_correction:
        investor_prior_enc = np.bincount(y_encoded, minlength=len(label_encoder.classes_)).astype(float)
        investor_prior_enc = investor_prior_enc / investor_prior_enc.sum()

        state_prior_raw = state_proba_raw.mean(axis=0)
        correction = investor_prior_enc / np.clip(state_prior_raw, 1e-6, None)

        state_proba_adj = state_proba_raw * correction
        state_proba_adj = state_proba_adj / state_proba_adj.sum(axis=1, keepdims=True)
    else:
        state_proba_adj = state_proba_raw

    state_pred_enc = state_proba_adj.argmax(axis=1)

    state_with_risk["predicted_risk_level_raw"] = label_encoder.inverse_transform(state_pred_raw_enc)
    state_with_risk["predicted_risk_level"] = label_encoder.inverse_transform(state_pred_enc)
    state_with_risk["risk_proba_raw"] = state_proba_raw.max(axis=1)
    state_with_risk["risk_proba"] = state_proba_adj.max(axis=1)

    state_with_risk_path: Path | None = None
    if persist_artifacts:
        state_with_risk_path = processed_path / "nfcs_state_with_risk.parquet"
        state_with_risk.to_parquet(state_with_risk_path, index=False)

    investor_distribution = y.value_counts(normalize=True).sort_index().reindex([1, 2, 3, 4, 5], fill_value=0.0)
    state_distribution_raw = (
        state_with_risk["predicted_risk_level_raw"]
        .value_counts(normalize=True)
        .sort_index()
        .reindex([1, 2, 3, 4, 5], fill_value=0.0)
    )
    state_distribution = (
        state_with_risk["predicted_risk_level"]
        .value_counts(normalize=True)
        .sort_index()
        .reindex([1, 2, 3, 4, 5], fill_value=0.0)
    )

    return {
        "comparison_df": comparison_df,
        "model_path": model_path,
        "shap_path": shap_path,
        "state_with_risk_path": state_with_risk_path,
        "investor_distribution": investor_distribution,
        "state_distribution_raw": state_distribution_raw,
        "state_distribution": state_distribution,
        "apply_prior_shift_correction": apply_prior_shift_correction,
        "train_shape": X_train.shape,
        "val_shape": X_val.shape,
        "test_shape": X_test.shape,
        "debug": {
            "X_train": X_train.copy(),
            "X_val": X_val.copy(),
            "X_test": X_test.copy(),
            "X_state_raw": X_state_raw.copy(),
            "X_state": X_state.copy(),
            "y_train": label_encoder.inverse_transform(y_train),
            "xgb": xgb,
            "state_df": state_with_risk.copy(),
        },
    }


def plot_transfer_validation(
    y_investor: pd.Series | np.ndarray,
    y_state_pred: pd.Series | np.ndarray,
    save_path: str | Path = "data/outputs/transfer_validation.png",
) -> Path:
    """Plot global transfer validation (distribution + absolute gap by risk level)."""
    output_img = _resolve_path(save_path, "data/outputs/transfer_validation.png")
    output_img.parent.mkdir(parents=True, exist_ok=True)

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    plt.style.use("seaborn-v0_8-whitegrid")

    levels = [1, 2, 3, 4, 5]
    y_inv = np.asarray(y_investor)
    y_sxs = np.asarray(y_state_pred)
    inv_dist = [float(np.mean(y_inv == l)) for l in levels]
    sxs_dist = [float(np.mean(y_sxs == l)) for l in levels]

    x = np.arange(5)
    w = 0.35
    axes[0].bar(x - w / 2, inv_dist, w, label="Investor Survey (real)", color="#4C72B0")
    axes[0].bar(x + w / 2, sxs_dist, w, label="State Survey (predicha)", color="#DD8452")
    axes[0].set_xticks(x)
    axes[0].set_xticklabels([
        "Nivel 1\n(defensivo)",
        "Nivel 2",
        "Nivel 3",
        "Nivel 4",
        "Nivel 5\n(agresivo)",
    ])
    axes[0].set_ylabel("Proporcion")
    axes[0].set_title("Distribucion de niveles de riesgo - transfer learning")
    axes[0].legend()

    gaps = [abs(i - s) for i, s in zip(inv_dist, sxs_dist)]
    colors = [
        "#2ca02c" if g < 0.10 else "#ff7f0e" if g < 0.20 else "#d62728"
        for g in gaps
    ]
    axes[1].bar(x, gaps, color=colors)
    axes[1].axhline(0.10, color="green", linestyle="--", alpha=0.7, label="Gap aceptable (0.10)")
    axes[1].axhline(0.20, color="orange", linestyle="--", alpha=0.7, label="Gap preocupante (0.20)")
    axes[1].set_xticks(x)
    axes[1].set_xticklabels(["N1", "N2", "N3", "N4", "N5"])
    axes[1].set_ylabel("Gap absoluto")
    axes[1].set_title("Domain shift por nivel de riesgo")
    axes[1].legend()

    plt.tight_layout()
    plt.savefig(output_img, dpi=150, bbox_inches="tight")
    plt.show()
    return output_img


def recode_risk_target(series: pd.Series, n_levels: int = 5) -> pd.Series:
    """Recode risk tolerance to ordinal levels using quantiles for balanced classes.

    DESIGN: Use qcut over cut for skewed NFCS risk variables. This reduces class-prior
    collapse where one level dominates and transfer degenerates.
    """
    try:
        recoded = pd.qcut(
            series,
            q=n_levels,
            labels=range(1, n_levels + 1),
            duplicates="drop",
        )
    except ValueError:
        recoded = pd.qcut(
            series.rank(method="first"),
            q=n_levels,
            labels=range(1, n_levels + 1),
        )

    recoded = recoded.astype(int)
    dist = recoded.value_counts(normalize=True).sort_index()
    print("Distribucion target recodificado (objetivo ~20% por nivel):")
    for level, pct in dist.items():
        bar = "#" * int(pct * 40)
        print(f"  Nivel {level}: {pct:.1%}  {bar}")

    max_imbalance = float(dist.max() - dist.min())
    if max_imbalance > 0.15:
        print(
            f"  AVISO: desbalance de {max_imbalance:.1%} entre niveles. "
            "Se aplicara class_weight/sample_weight."
        )

    return recoded


def validate_model_before_transfer(
    model: Any,
    X_test: np.ndarray,
    y_test: np.ndarray,
    label_encoder: LabelEncoder | None = None,
) -> bool:
    """Check that model predictions are not collapsed before transfer."""
    y_pred_raw = np.asarray(model.predict(X_test))
    if y_pred_raw.ndim > 1:
        y_pred_raw = y_pred_raw.argmax(axis=1)

    if label_encoder is not None:
        y_pred = label_encoder.inverse_transform(y_pred_raw.astype(int))
        y_true = label_encoder.inverse_transform(np.asarray(y_test).astype(int))
    else:
        y_pred = y_pred_raw + 1
        y_true = np.asarray(y_test)

    unique_preds = np.unique(y_pred)
    print(f"Clases predichas en test set: {unique_preds}")

    if len(unique_preds) == 1:
        raise ValueError(
            f"El modelo predice solo la clase {unique_preds[0]} en test. "
            "Transfer no interpretable; revisar recodificacion y balance."
        )

    if len(unique_preds) < 3:
        print(
            f"AVISO: el modelo usa solo {len(unique_preds)} clases en test. "
            "El transfer puede ser poco informativo."
        )

    pred_dist = pd.Series(y_pred).value_counts(normalize=True).sort_index()
    print("Distribucion predicciones en test set:")
    print(pred_dist.round(3).to_string())
    _ = y_true  # keeps API parity for future test diagnostics

    return len(unique_preds) >= 3


def plot_transfer_validation_v2(
    y_inv_train: pd.Series | np.ndarray,
    y_inv_test: pd.Series | np.ndarray,
    y_inv_test_pred: pd.Series | np.ndarray,
    y_state_pred: pd.Series | np.ndarray,
    save_path: str | Path = "data/outputs/transfer_validation_v2.png",
) -> Path:
    """Three-panel transfer validation: train balance, test quality, and transfer gap."""
    output_img = _resolve_path(save_path, "data/outputs/transfer_validation_v2.png")
    output_img.parent.mkdir(parents=True, exist_ok=True)

    fig, axes = plt.subplots(1, 3, figsize=(16, 5))
    plt.style.use("seaborn-v0_8-whitegrid")
    levels = [1, 2, 3, 4, 5]

    def get_dist(y: pd.Series | np.ndarray) -> list[float]:
        arr = np.asarray(y)
        return [float(np.mean(arr == l)) for l in levels]

    blue = "#4C72B0"
    orange = "#DD8452"
    green = "#55A868"
    x = np.arange(5)
    w = 0.35

    axes[0].bar(x, get_dist(y_inv_train), color=blue)
    axes[0].axhline(0.20, color="red", linestyle="--", alpha=0.7, label="20% ideal")
    axes[0].set_title("Target en train\n(debe ser ~20% por nivel)")
    axes[0].set_xticks(x)
    axes[0].set_xticklabels([f"N{l}" for l in levels])
    axes[0].set_ylabel("Proporcion")
    axes[0].legend()

    axes[1].bar(x - w / 2, get_dist(y_inv_test), w, label="Real", color=blue)
    axes[1].bar(x + w / 2, get_dist(y_inv_test_pred), w, label="Predicho", color=orange)
    axes[1].set_title("Test Investor Survey\n(calidad del modelo XGBoost)")
    axes[1].set_xticks(x)
    axes[1].set_xticklabels([f"N{l}" for l in levels])
    axes[1].legend()

    axes[2].bar(x - w / 2, get_dist(y_inv_train), w, label="Investor (real)", color=blue)
    axes[2].bar(x + w / 2, get_dist(y_state_pred), w, label="State (predicha)", color=green)
    axes[2].set_title("Transfer learning\n(Investor real vs State predicho)")
    axes[2].set_xticks(x)
    axes[2].set_xticklabels([f"N{l}" for l in levels])
    axes[2].legend()

    for ax in axes:
        ax.set_ylim(0, 1)

    plt.tight_layout()
    plt.savefig(output_img, dpi=150, bbox_inches="tight")
    plt.show()
    return output_img


def train_risk_model_v2(
    investor_csv: str | Path,
    state_csv: str | Path,
    output_dir: str | Path = "data/outputs",
    processed_dir: str | Path = "data/processed",
    target_col_override: str | None = None,
    compute_shap: bool = True,
    apply_prior_shift_correction: bool = True,
    calibration_method: str = "sigmoid",
    smoothing_alpha: float | None = None,
    feature_set_mode: str = "full",
    smd_threshold: float = 0.5,
) -> dict[str, Any]:
    """Corrected end-to-end train/transfer pipeline with robust shared-feature preprocessing."""
    output_path = _resolve_path(output_dir, "data/outputs")
    processed_path = _resolve_path(processed_dir, "data/processed")
    output_path.mkdir(parents=True, exist_ok=True)
    processed_path.mkdir(parents=True, exist_ok=True)

    df_inv = pd.read_csv(investor_csv, low_memory=False)
    df_sxs = pd.read_csv(state_csv, low_memory=False)

    target_col, candidate_scores = detect_target_column(
        df_inv,
        target_col_override=target_col_override,
    )
    logger.info("Target detectado: %s", target_col)

    target_numeric = pd.to_numeric(df_inv[target_col], errors="coerce")
    valid_mask = target_numeric.notna() & (target_numeric < 97)
    df_inv = df_inv.loc[valid_mask].copy()
    df_inv[target_col] = target_numeric.loc[valid_mask]

    df_inv["risk_level"] = recode_risk_target(df_inv[target_col], n_levels=5)

    feature_cols = get_shared_features(df_inv, df_sxs, target_col)
    if len(feature_cols) < 3:
        raise ValueError(
            f"Solo {len(feature_cols)} features compartidas entre datasets. "
            "Revisar nombres de columnas: posible problema de schema."
        )

    X_inv = _normalize_missing_tokens(df_inv[feature_cols].copy())
    y_inv = df_inv["risk_level"].astype(int).values
    label_encoder = LabelEncoder()
    y_inv_enc = label_encoder.fit_transform(y_inv)
    X_sxs = _normalize_missing_tokens(df_sxs[feature_cols].copy())

    numeric_cols, _ = _infer_feature_types(X_inv, feature_cols)
    for col in numeric_cols:
        X_inv[col] = pd.to_numeric(X_inv[col], errors="coerce")
        X_sxs[col] = pd.to_numeric(X_sxs[col], errors="coerce")

    preprocessor, ordered_cols = build_preprocessor(feature_cols, X_inv)
    X_inv_proc = preprocessor.fit_transform(X_inv)
    X_sxs_proc = preprocessor.transform(X_sxs)

    X_inv_proc = np.asarray(X_inv_proc)
    X_sxs_proc = np.asarray(X_sxs_proc)

    from xgboost import XGBClassifier

    train_mean = X_inv_proc.mean(axis=0)
    train_std = X_inv_proc.std(axis=0)
    state_mean = X_sxs_proc.mean(axis=0)
    state_std = X_sxs_proc.std(axis=0)
    smd = np.abs(train_mean - state_mean) / (train_std + 1e-6)

    drift_df = pd.DataFrame(
        {
            "feature": ordered_cols,
            "train_mean": train_mean,
            "state_mean": state_mean,
            "train_std": train_std,
            "state_std": state_std,
            "smd": smd,
        }
    ).sort_values("smd", ascending=False)

    drift_score = float(np.median(smd))
    high_drift_ratio = float((smd > smd_threshold).mean())
    robust_mode = drift_score > 0.35 or high_drift_ratio > 0.30

    removed_features: list[str] = []
    used_feature_cols = list(ordered_cols)
    if feature_set_mode not in {"full", "reduced"}:
        raise ValueError("feature_set_mode must be 'full' or 'reduced'")

    if feature_set_mode == "reduced":
        removed_features = drift_df.loc[drift_df["smd"] > smd_threshold, "feature"].tolist()
        kept = [f for f in ordered_cols if f not in removed_features]
        if len(kept) >= 3:
            idx = [ordered_cols.index(f) for f in kept]
            X_inv_proc = X_inv_proc[:, idx]
            X_sxs_proc = X_sxs_proc[:, idx]
            used_feature_cols = kept
        else:
            removed_features = []
            feature_set_mode = "full"

    X_tr, X_tmp, y_tr, y_tmp = train_test_split(
        X_inv_proc,
        y_inv_enc,
        test_size=0.30,
        stratify=y_inv_enc,
        random_state=RANDOM_STATE,
    )
    X_val, X_te, y_val, y_te = train_test_split(
        X_tmp,
        y_tmp,
        test_size=0.50,
        stratify=y_tmp,
        random_state=RANDOM_STATE,
    )

    print(f"Drift median SMD: {drift_score:.3f}")
    print(f"High-drift feature ratio (SMD>{smd_threshold}): {high_drift_ratio:.1%}")
    if robust_mode:
        print("AVISO: mismatch fuerte entre P(X|Y) source/target. Activando hiperparametros robustos.")
    print(f"Feature set mode: {feature_set_mode} | usadas: {len(used_feature_cols)}")
    if removed_features:
        print(f"Features removidas por drift (>{smd_threshold}): {removed_features}")

    xgb = XGBClassifier(
        objective="multi:softprob",
        num_class=len(label_encoder.classes_),
        n_estimators=500,
        max_depth=3 if robust_mode else 5,
        learning_rate=0.05,
        subsample=0.7 if robust_mode else 0.8,
        colsample_bytree=0.7 if robust_mode else 0.8,
        min_child_weight=5 if robust_mode else 1,
        reg_lambda=5.0 if robust_mode else 1.0,
        gamma=0.2 if robust_mode else 0.0,
        random_state=RANDOM_STATE,
        eval_metric="mlogloss",
        early_stopping_rounds=50,
    )
    xgb.fit(
        X_tr,
        y_tr,
        sample_weight=compute_sample_weight(class_weight="balanced", y=y_tr),
        eval_set=[(X_val, y_val)],
        verbose=False,
    )

    if calibration_method not in {"sigmoid", "isotonic"}:
        raise ValueError("calibration_method must be 'sigmoid' or 'isotonic'")

    xgb_for_calibration = XGBClassifier(
        objective="multi:softprob",
        num_class=len(label_encoder.classes_),
        n_estimators=350,
        max_depth=3 if robust_mode else 5,
        learning_rate=0.05,
        subsample=0.7 if robust_mode else 0.8,
        colsample_bytree=0.7 if robust_mode else 0.8,
        min_child_weight=5 if robust_mode else 1,
        reg_lambda=5.0 if robust_mode else 1.0,
        gamma=0.2 if robust_mode else 0.0,
        random_state=RANDOM_STATE,
        eval_metric="mlogloss",
    )
    calibrator = CalibratedClassifierCV(
        estimator=xgb_for_calibration,
        method=calibration_method,
        cv=3,
    )
    calibrator.fit(X_tr, y_tr)

    y_pred_proba_test = np.asarray(calibrator.predict_proba(X_te))
    y_pred_enc = y_pred_proba_test.argmax(axis=1)
    y_pred = label_encoder.inverse_transform(y_pred_enc.astype(int))
    y_te_labels = label_encoder.inverse_transform(y_te.astype(int))

    usable_model = validate_model_before_transfer(
        calibrator,
        X_te,
        y_te,
        label_encoder=label_encoder,
    )

    comparison_df = pd.DataFrame(
        {
            "Model": ["XGBoost"],
            "Accuracy": [accuracy_score(y_te_labels, y_pred)],
            "F1_macro": [f1_score(y_te_labels, y_pred, average="macro")],
            "MAE_ordinal": [mean_absolute_error(y_te_labels, y_pred)],
        }
    )

    shap_path: Path | None = None
    if compute_shap:
        import shap

        explainer = shap.TreeExplainer(xgb)
        shap_values = explainer.shap_values(X_te)
        shap.summary_plot(shap_values, X_te, feature_names=ordered_cols, show=False)
        plt.title("Importancia de variables - nivel de riesgo (v2)")
        plt.tight_layout()
        shap_path = output_path / "shap_risk_model_v2.png"
        plt.savefig(shap_path, dpi=150)
        plt.close()

    state_proba_raw = np.asarray(calibrator.predict_proba(X_sxs_proc))
    state_pred_raw_enc = state_proba_raw.argmax(axis=1)

    investor_prior_enc = np.bincount(y_inv_enc, minlength=len(label_encoder.classes_)).astype(float)
    investor_prior_enc = investor_prior_enc / investor_prior_enc.sum()
    state_prior_raw = state_proba_raw.mean(axis=0)

    print("\nDistribucion train (source P(Y)):")
    print(pd.Series(investor_prior_enc, index=label_encoder.classes_).round(4).to_string())
    print("\nDistribucion State predicha ANTES de correccion:")
    print(pd.Series(state_prior_raw, index=label_encoder.classes_).round(4).to_string())

    if apply_prior_shift_correction:
        correction = investor_prior_enc / np.clip(state_prior_raw, 1e-6, None)
        state_proba_adj = state_proba_raw * correction
        state_proba_adj = state_proba_adj / state_proba_adj.sum(axis=1, keepdims=True)
    else:
        state_proba_adj = state_proba_raw

    # Smooth corrected probabilities towards source prior to reduce residual collapse.
    raw_dominance = float(state_prior_raw.max())
    if smoothing_alpha is None:
        smoothing_alpha = 0.12 if raw_dominance > 0.60 else 0.05
    smoothing_alpha = float(np.clip(smoothing_alpha, 0.0, 0.5))
    state_proba_final = (1.0 - smoothing_alpha) * state_proba_adj + smoothing_alpha * investor_prior_enc
    state_proba_final = state_proba_final / state_proba_final.sum(axis=1, keepdims=True)

    y_sxs_pred_enc = state_proba_final.argmax(axis=1)
    y_sxs_pred = label_encoder.inverse_transform(y_sxs_pred_enc.astype(int))

    y_sxs_proba_raw = state_proba_raw.max(axis=1)
    y_sxs_proba = state_proba_final.max(axis=1)

    state_with_risk = df_sxs.copy()
    state_with_risk["predicted_risk_level_raw"] = label_encoder.inverse_transform(state_pred_raw_enc.astype(int))
    state_with_risk["predicted_risk_level"] = y_sxs_pred
    state_with_risk["risk_proba_raw"] = y_sxs_proba_raw
    state_with_risk["risk_proba"] = y_sxs_proba

    state_with_risk_path = processed_path / "nfcs_state_with_risk_v2.parquet"
    state_with_risk.to_parquet(state_with_risk_path, index=False)

    inv_dist = (
        pd.Series(y_inv)
        .value_counts(normalize=True)
        .sort_index()
        .reindex([1, 2, 3, 4, 5], fill_value=0.0)
    )
    sxs_dist_raw = (
        pd.Series(label_encoder.inverse_transform(state_pred_raw_enc.astype(int)))
        .value_counts(normalize=True)
        .sort_index()
        .reindex([1, 2, 3, 4, 5], fill_value=0.0)
    )
    sxs_dist = (
        pd.Series(y_sxs_pred)
        .value_counts(normalize=True)
        .sort_index()
        .reindex([1, 2, 3, 4, 5], fill_value=0.0)
    )

    print("\nDistribucion State predicha DESPUES de correccion/smoothing:")
    print(sxs_dist.round(4).to_string())

    state_proxy_crosstab = pd.DataFrame()
    if target_col in df_sxs.columns:
        state_proxy_num = pd.to_numeric(df_sxs[target_col], errors="coerce")
        proxy_mask = state_proxy_num.notna() & (state_proxy_num < 97)
        if int(proxy_mask.sum()) > 50:
            proxy_recoded = recode_risk_target(state_proxy_num.loc[proxy_mask], n_levels=5)
            pred_proxy_aligned = pd.Series(y_sxs_pred, index=df_sxs.index).loc[proxy_mask]
            state_proxy_crosstab = pd.crosstab(
                pred_proxy_aligned,
                proxy_recoded,
                normalize="index",
            )

    feature_importance_df = pd.DataFrame(
        {
            "feature": used_feature_cols,
            "importance": xgb.feature_importances_,
        }
    ).sort_values("importance", ascending=False)

    distribution_comparison = pd.DataFrame(
        {
            "Investor_real": inv_dist,
            "State_predicha_raw": sxs_dist_raw,
            "State_predicha": sxs_dist,
        }
    )
    max_gap = float((distribution_comparison["Investor_real"] - distribution_comparison["State_predicha"]).abs().max())
    max_gap_raw = float((distribution_comparison["Investor_real"] - distribution_comparison["State_predicha_raw"]).abs().max())

    model_bundle_path = output_path / "xgb_risk_model_v2.pkl"
    joblib.dump(
        {
            "model": xgb,
            "calibrated_model": calibrator,
            "preprocessor": preprocessor,
            "feature_cols": feature_cols,
            "ordered_cols": ordered_cols,
            "used_feature_cols": used_feature_cols,
            "target_col": target_col,
        },
        model_bundle_path,
    )

    return {
        "target_col": target_col,
        "candidate_scores": candidate_scores,
        "feature_cols": feature_cols,
        "ordered_cols": ordered_cols,
        "comparison_df": comparison_df,
        "distribution_comparison": distribution_comparison,
        "max_gap": max_gap,
        "max_gap_raw": max_gap_raw,
        "apply_prior_shift_correction": apply_prior_shift_correction,
        "calibration_method": calibration_method,
        "smoothing_alpha": smoothing_alpha,
        "feature_set_mode": feature_set_mode,
        "removed_features": removed_features,
        "smd_threshold": smd_threshold,
        "drift_score_median_smd": drift_score,
        "high_drift_ratio": high_drift_ratio,
        "robust_mode": robust_mode,
        "drift_table": drift_df,
        "feature_importance": feature_importance_df,
        "state_proxy_crosstab": state_proxy_crosstab,
        "model_path": model_bundle_path,
        "shap_path": shap_path,
        "state_with_risk_path": state_with_risk_path,
        "y_investor": y_inv,
        "y_investor_train": label_encoder.inverse_transform(y_tr.astype(int)),
        "y_investor_test": y_te_labels,
        "y_investor_test_pred": y_pred,
        "y_state_pred": y_sxs_pred,
        "usable_model": usable_model,
        "X_inv_shape": X_inv_proc.shape,
        "X_sxs_shape": X_sxs_proc.shape,
    }


def _compute_smd_table(
    X_source: np.ndarray,
    X_target: np.ndarray,
    feature_names: list[str],
) -> pd.DataFrame:
    """Compute standardized mean differences for source vs target features."""
    src_mean = X_source.mean(axis=0)
    src_std = X_source.std(axis=0)
    tgt_mean = X_target.mean(axis=0)
    tgt_std = X_target.std(axis=0)
    smd = np.abs(src_mean - tgt_mean) / (src_std + 1e-6)
    return pd.DataFrame(
        {
            "feature": feature_names,
            "source_mean": src_mean,
            "target_mean": tgt_mean,
            "source_std": src_std,
            "target_std": tgt_std,
            "smd": smd,
        }
    ).sort_values("smd", ascending=False)


def _compute_domain_importance_weights(
    X_source: np.ndarray,
    X_target: np.ndarray,
    clip_min: float = 0.2,
    clip_max: float = 5.0,
) -> tuple[np.ndarray, dict[str, float]]:
    """Estimate covariate-shift importance weights via domain classifier."""
    X_domain = np.vstack([X_source, X_target])
    y_domain = np.concatenate([
        np.zeros(X_source.shape[0], dtype=int),
        np.ones(X_target.shape[0], dtype=int),
    ])

    domain_clf = RandomForestClassifier(
        n_estimators=300,
        max_depth=6,
        random_state=RANDOM_STATE,
        class_weight="balanced_subsample",
    )
    domain_clf.fit(X_domain, y_domain)

    p_target = domain_clf.predict_proba(X_source)[:, 1]
    p_source = np.clip(1.0 - p_target, 1e-4, None)
    weights = p_target / p_source
    weights = np.clip(weights, clip_min, clip_max)

    stats = {
        "weight_mean": float(np.mean(weights)),
        "weight_std": float(np.std(weights)),
        "weight_min": float(np.min(weights)),
        "weight_max": float(np.max(weights)),
    }
    return weights, stats


def _prior_shift_correct(
    probs: np.ndarray,
    source_prior: np.ndarray,
    enabled: bool,
) -> np.ndarray:
    """Apply label prior correction from source prior to target predicted prior."""
    if not enabled:
        return probs
    target_prior = probs.mean(axis=0)
    correction = source_prior / np.clip(target_prior, 1e-6, None)
    adj = probs * correction
    adj = adj / np.clip(adj.sum(axis=1, keepdims=True), 1e-9, None)
    return adj


def _fit_calibrated_multiclass(
    X_train: np.ndarray,
    y_train: np.ndarray,
    calibration_method: str,
    sample_weight: np.ndarray | None = None,
) -> CalibratedClassifierCV:
    """Train calibrated multiclass XGBoost model with conservative hyperparameters."""
    from xgboost import XGBClassifier

    base = XGBClassifier(
        objective="multi:softprob",
        num_class=len(np.unique(y_train)),
        n_estimators=350,
        max_depth=3,
        min_child_weight=5,
        learning_rate=0.05,
        subsample=0.7,
        colsample_bytree=0.7,
        reg_lambda=5.0,
        gamma=0.2,
        random_state=RANDOM_STATE,
        eval_metric="mlogloss",
    )
    calibrated = CalibratedClassifierCV(estimator=base, method=calibration_method, cv=3)
    if sample_weight is None:
        calibrated.fit(X_train, y_train)
    else:
        calibrated.fit(X_train, y_train, sample_weight=sample_weight)
    return calibrated


def _ordinal_probs_from_binary(
    models: list[CalibratedClassifierCV],
    X: np.ndarray,
) -> np.ndarray:
    """Reconstruct class probabilities from calibrated cumulative binary models."""
    q2 = models[0].predict_proba(X)[:, 1]
    q3 = models[1].predict_proba(X)[:, 1]
    q4 = models[2].predict_proba(X)[:, 1]
    q5 = models[3].predict_proba(X)[:, 1]

    q = np.column_stack([q2, q3, q4, q5])
    for i in range(1, q.shape[1]):
        q[:, i] = np.minimum(q[:, i], q[:, i - 1])

    p1 = 1.0 - q[:, 0]
    p2 = q[:, 0] - q[:, 1]
    p3 = q[:, 1] - q[:, 2]
    p4 = q[:, 2] - q[:, 3]
    p5 = q[:, 3]
    probs = np.column_stack([p1, p2, p3, p4, p5])
    probs = np.clip(probs, 1e-8, None)
    probs = probs / probs.sum(axis=1, keepdims=True)
    return probs


def _fit_ordinal_binary_models(
    X_train: np.ndarray,
    y_train_raw: np.ndarray,
    calibration_method: str,
    sample_weight: np.ndarray | None = None,
) -> list[CalibratedClassifierCV]:
    """Fit calibrated binary models for P(Y>=k), k in {2,3,4,5}."""
    from xgboost import XGBClassifier

    models: list[CalibratedClassifierCV] = []
    for threshold in [2, 3, 4, 5]:
        y_bin = (y_train_raw >= threshold).astype(int)
        base = XGBClassifier(
            objective="binary:logistic",
            n_estimators=300,
            max_depth=3,
            min_child_weight=5,
            learning_rate=0.05,
            subsample=0.7,
            colsample_bytree=0.7,
            reg_lambda=5.0,
            gamma=0.2,
            random_state=RANDOM_STATE,
            eval_metric="logloss",
        )
        calib = CalibratedClassifierCV(estimator=base, method=calibration_method, cv=3)

        if sample_weight is None:
            calib.fit(X_train, y_bin)
        else:
            w_bin = np.asarray(sample_weight, dtype=float).copy()
            class_bal = compute_sample_weight(class_weight="balanced", y=y_bin)
            w_bin = w_bin * class_bal
            calib.fit(X_train, y_bin, sample_weight=w_bin)

        models.append(calib)
    return models


def compare_transfer_approaches(
    investor_csv: str | Path,
    state_csv: str | Path,
    target_col_override: str | None = None,
    calibration_method: str = "isotonic",
    apply_prior_shift_correction: bool = True,
    output_dir: str | Path = "data/outputs",
) -> dict[str, Any]:
    """Compare baseline/weighted multiclass and ordinal alternatives under domain shift."""
    output_path = _resolve_path(output_dir, "data/outputs")
    output_path.mkdir(parents=True, exist_ok=True)

    df_inv = pd.read_csv(investor_csv, low_memory=False)
    df_sxs = pd.read_csv(state_csv, low_memory=False)

    target_col, candidate_scores = detect_target_column(
        df_inv,
        target_col_override=target_col_override,
    )

    target_numeric = pd.to_numeric(df_inv[target_col], errors="coerce")
    valid_mask = target_numeric.notna() & (target_numeric < 97)
    df_inv = df_inv.loc[valid_mask].copy()
    df_inv[target_col] = target_numeric.loc[valid_mask]
    df_inv["risk_level"] = recode_risk_target(df_inv[target_col], n_levels=5)

    feature_cols = get_shared_features(df_inv, df_sxs, target_col)
    X_inv = _normalize_missing_tokens(df_inv[feature_cols].copy())
    X_sxs = _normalize_missing_tokens(df_sxs[feature_cols].copy())
    numeric_cols, _ = _infer_feature_types(X_inv, feature_cols)
    for col in numeric_cols:
        X_inv[col] = pd.to_numeric(X_inv[col], errors="coerce")
        X_sxs[col] = pd.to_numeric(X_sxs[col], errors="coerce")

    preprocessor, ordered_cols = build_preprocessor(feature_cols, X_inv)
    X_source = np.asarray(preprocessor.fit_transform(X_inv))
    X_target = np.asarray(preprocessor.transform(X_sxs))

    y_raw = df_inv["risk_level"].astype(int).to_numpy()
    label_encoder = LabelEncoder()
    y_enc = label_encoder.fit_transform(y_raw)

    X_train, X_test, y_train_enc, y_test_enc, y_train_raw, y_test_raw = train_test_split(
        X_source,
        y_enc,
        y_raw,
        test_size=0.25,
        stratify=y_enc,
        random_state=RANDOM_STATE,
    )

    source_prior = np.bincount(y_enc, minlength=len(label_encoder.classes_)).astype(float)
    source_prior = source_prior / source_prior.sum()

    drift_table = _compute_smd_table(X_source, X_target, ordered_cols)
    high_drift_features = drift_table.loc[drift_table["smd"] > 0.5, "feature"].tolist()
    domain_weights_all, domain_weight_stats = _compute_domain_importance_weights(X_source, X_target)
    domain_weights_train, _ = train_test_split(
        domain_weights_all,
        test_size=0.25,
        stratify=y_enc,
        random_state=RANDOM_STATE,
    )

    class_bal_train = compute_sample_weight(class_weight="balanced", y=y_train_enc)
    weighted_train = class_bal_train * domain_weights_train

    results: dict[str, dict[str, Any]] = {}

    def evaluate_approach(name: str, test_probs: np.ndarray, state_probs_raw: np.ndarray) -> None:
        test_pred_enc = test_probs.argmax(axis=1)
        test_pred = label_encoder.inverse_transform(test_pred_enc.astype(int))
        state_raw_enc = state_probs_raw.argmax(axis=1)

        state_probs_corr = _prior_shift_correct(state_probs_raw, source_prior, apply_prior_shift_correction)
        state_corr_enc = state_probs_corr.argmax(axis=1)

        train_dist = (
            pd.Series(y_raw).value_counts(normalize=True).sort_index().reindex([1, 2, 3, 4, 5], fill_value=0.0)
        )
        state_raw_dist = (
            pd.Series(label_encoder.inverse_transform(state_raw_enc.astype(int)))
            .value_counts(normalize=True)
            .sort_index()
            .reindex([1, 2, 3, 4, 5], fill_value=0.0)
        )
        state_corr_dist = (
            pd.Series(label_encoder.inverse_transform(state_corr_enc.astype(int)))
            .value_counts(normalize=True)
            .sort_index()
            .reindex([1, 2, 3, 4, 5], fill_value=0.0)
        )

        max_gap_raw = float((train_dist - state_raw_dist).abs().max())
        max_gap_corr = float((train_dist - state_corr_dist).abs().max())
        classes_present = int((state_corr_dist > 0).sum())
        collapse = classes_present < 5

        results[name] = {
            "approach": name,
            "train_distribution": train_dist,
            "state_distribution_raw": state_raw_dist,
            "state_distribution_corrected": state_corr_dist,
            "max_gap_raw": max_gap_raw,
            "max_gap": max_gap_corr,
            "classes_present": classes_present,
            "class_collapse": collapse,
            "dominance": float(state_corr_dist.max()),
            "test_accuracy": float(accuracy_score(y_test_raw, test_pred)),
            "test_f1_macro": float(f1_score(y_test_raw, test_pred, average="macro")),
        }

    # 1) Baseline multiclass
    base_multi = _fit_calibrated_multiclass(
        X_train,
        y_train_enc,
        calibration_method=calibration_method,
        sample_weight=class_bal_train,
    )
    evaluate_approach(
        "baseline_multiclass",
        test_probs=np.asarray(base_multi.predict_proba(X_test)),
        state_probs_raw=np.asarray(base_multi.predict_proba(X_target)),
    )

    # 2) Weighted multiclass (covariate shift)
    weighted_multi = _fit_calibrated_multiclass(
        X_train,
        y_train_enc,
        calibration_method=calibration_method,
        sample_weight=weighted_train,
    )
    evaluate_approach(
        "weighted_multiclass",
        test_probs=np.asarray(weighted_multi.predict_proba(X_test)),
        state_probs_raw=np.asarray(weighted_multi.predict_proba(X_target)),
    )

    # 3) Ordinal model
    ordinal_models = _fit_ordinal_binary_models(
        X_train,
        y_train_raw,
        calibration_method=calibration_method,
        sample_weight=class_bal_train,
    )
    evaluate_approach(
        "ordinal",
        test_probs=_ordinal_probs_from_binary(ordinal_models, X_test),
        state_probs_raw=_ordinal_probs_from_binary(ordinal_models, X_target),
    )

    # 4) Ordinal + weighted
    ordinal_weighted_models = _fit_ordinal_binary_models(
        X_train,
        y_train_raw,
        calibration_method=calibration_method,
        sample_weight=weighted_train,
    )
    evaluate_approach(
        "ordinal_weighted",
        test_probs=_ordinal_probs_from_binary(ordinal_weighted_models, X_test),
        state_probs_raw=_ordinal_probs_from_binary(ordinal_weighted_models, X_target),
    )

    summary_df = pd.DataFrame(results.values()).sort_values(
        ["max_gap", "class_collapse", "dominance"],
        ascending=[True, True, True],
    )

    best_row = summary_df.iloc[0]
    best_name = str(best_row["approach"])
    best_result = results[best_name]

    print("\n=== DOMAIN ADAPTATION COMPARISON SUMMARY ===")
    print(
        summary_df[[
            "approach",
            "max_gap_raw",
            "max_gap",
            "classes_present",
            "class_collapse",
            "dominance",
            "test_accuracy",
            "test_f1_macro",
        ]].to_string(index=False)
    )

    print("\nBEST APPROACH:")
    print(f"- {best_name}")
    print(f"- max_gap: {best_result['max_gap']:.3f}")
    print("- final class distribution:")
    print(best_result["state_distribution_corrected"].round(4).to_string())

    return {
        "target_col": target_col,
        "candidate_scores": candidate_scores,
        "feature_cols": feature_cols,
        "drift_table": drift_table,
        "high_drift_features": high_drift_features,
        "domain_weight_stats": domain_weight_stats,
        "summary_df": summary_df,
        "results": results,
        "best_approach": best_name,
        "best_result": best_result,
    }
