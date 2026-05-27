from __future__ import annotations

import logging
import os
import zipfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import requests
from dotenv import load_dotenv
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder, StandardScaler
from xgboost import XGBClassifier

logger = logging.getLogger(__name__)
PROJECT_ROOT = Path(__file__).resolve().parents[1]
RANDOM_STATE = 42


@dataclass(slots=True)
class TransferDataPaths:
    scf_data_file: Path
    hfcs_microdata_file: Path | None
    hfcs_access_form: Path
    hfcs_stat_tables_zip: Path
    raw_dir: Path


DEFAULTS = {
    "SCF_MICRODATA_URL": "https://www.federalreserve.gov/econres/files/scfp2022excel.zip",
    "SCF_MICRODATA_FALLBACK_URL": "https://www.federalreserve.gov/econres/files/scfp2022.zip",
    "HFCS_STAT_TABLES_URL": "https://www.ecb.europa.eu/home/pdf/research/hfcn/HFCS_Statistical_Tables_Wave_2021_July2023.zip?399f19fc6c4ccc75c8150537516350b9",
    "HFCS_ACCESS_FORM_URL": "https://www.ecb.europa.eu/home/pdf/research/hfcn/access_form_leadresearchersurname_researchersurname.pdf",
}


FEATURE_ALIASES: dict[str, list[str]] = {
    "age": ["age", "x14", "respondent_age", "ha_age", "ra0010"],
    "income": ["income", "wageinc", "netinc", "y1", "dn3001", "gross_income"],
    "net_wealth": ["networth", "net_wealth", "wealth", "w3", "dn3002", "total_net_wealth"],
    "education_level": ["education", "educ", "edcl", "x5902", "isced", "pe0400"],
    "stock_ownership": ["stocks", "stock_ownership", "eq_own", "x3915", "da2100", "risky_asset_owner"],
}


def _resolve_path(path_like: str | Path, default_relative: str) -> Path:
    path = Path(path_like) if path_like else PROJECT_ROOT / default_relative
    if not path.is_absolute():
        path = PROJECT_ROOT / path
    return path


def _download_file(url: str, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    with requests.get(url, stream=True, timeout=120) as response:
        response.raise_for_status()
        with destination.open("wb") as handle:
            for chunk in response.iter_content(chunk_size=1024 * 64):
                if chunk:
                    handle.write(chunk)


def _extract_zip(zip_path: Path, extract_dir: Path) -> None:
    extract_dir.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(zip_path, "r") as zf:
        zf.extractall(extract_dir)


def _clear_directory(path: Path) -> None:
    if not path.exists():
        return
    for child in path.rglob("*"):
        if child.is_file():
            child.unlink(missing_ok=True)


def _find_first_file(root: Path, suffixes: tuple[str, ...]) -> Path:
    candidates: list[Path] = []
    for suffix in suffixes:
        candidates.extend(root.rglob(f"*{suffix}"))
    if not candidates:
        raise FileNotFoundError(f"No se encontro archivo con extensiones {suffixes} en {root}")
    return sorted(candidates, key=lambda p: p.stat().st_size, reverse=True)[0]


def download_scf_hfcs_data(
    raw_dir: str | Path = "data/raw/scf_hfcs",
    force: bool = False,
) -> TransferDataPaths:
    """Download SCF files and HFCS public access artifacts.

    HFCS microdata are access-restricted by ECB. This function downloads the
    request form and public statistical tables, and optionally picks up local
    HFCS microdata if HFCS_MICRODATA_PATH is defined in .env.
    """
    load_dotenv(PROJECT_ROOT / ".env")
    raw_path = _resolve_path(raw_dir, "data/raw/scf_hfcs")
    raw_path.mkdir(parents=True, exist_ok=True)

    scf_url = os.getenv("SCF_MICRODATA_URL", DEFAULTS["SCF_MICRODATA_URL"])
    scf_fallback_url = os.getenv("SCF_MICRODATA_FALLBACK_URL", DEFAULTS["SCF_MICRODATA_FALLBACK_URL"])
    hfcs_stats_url = os.getenv("HFCS_STAT_TABLES_URL", DEFAULTS["HFCS_STAT_TABLES_URL"])
    hfcs_access_url = os.getenv("HFCS_ACCESS_FORM_URL", DEFAULTS["HFCS_ACCESS_FORM_URL"])

    scf_zip = raw_path / "scf_2022_microdata.zip"
    scf_extract = raw_path / "scf_2022"

    hfcs_stats_zip = raw_path / "hfcs_2021_stat_tables.zip"
    hfcs_access_pdf = raw_path / "hfcs_access_form.pdf"

    if force or not scf_zip.exists():
        logger.info("Downloading SCF microdata ZIP from official source...")
        _download_file(scf_url, scf_zip)
    if force or not scf_extract.exists() or not any(scf_extract.iterdir()):
        logger.info("Extracting SCF ZIP...")
        _extract_zip(scf_zip, scf_extract)

    if force or not hfcs_stats_zip.exists():
        logger.info("Downloading HFCS public statistical tables ZIP...")
        _download_file(hfcs_stats_url, hfcs_stats_zip)

    if force or not hfcs_access_pdf.exists():
        logger.info("Downloading HFCS access request form...")
        _download_file(hfcs_access_url, hfcs_access_pdf)

    try:
        scf_data_file = _find_first_file(scf_extract, (".csv", ".dta", ".sas7bdat"))
    except FileNotFoundError:
        logger.warning("Primary SCF file format not found in cache. Re-downloading primary SCF URL...")
        _download_file(scf_url, scf_zip)
        _clear_directory(scf_extract)
        _extract_zip(scf_zip, scf_extract)
        try:
            scf_data_file = _find_first_file(scf_extract, (".csv", ".dta", ".sas7bdat"))
        except FileNotFoundError:
            logger.warning("Primary SCF URL still missing tabular file. Retrying with fallback SCF URL...")
            _download_file(scf_fallback_url, scf_zip)
            _clear_directory(scf_extract)
            _extract_zip(scf_zip, scf_extract)
            scf_data_file = _find_first_file(scf_extract, (".csv", ".dta", ".sas7bdat"))

    hfcs_micro_path_env = None
    try:
        hfcs_micro_path_env = Path(str(os.getenv("HFCS_MICRODATA_PATH", "")).strip())
    except Exception:  # noqa: BLE001
        hfcs_micro_path_env = None

    hfcs_microdata_file: Path | None = None
    if hfcs_micro_path_env and str(hfcs_micro_path_env) not in {"", "."}:
        resolved = _resolve_path(hfcs_micro_path_env, "")
        if resolved.exists():
            hfcs_microdata_file = resolved
        else:
            logger.warning("HFCS_MICRODATA_PATH is set but file does not exist: %s", resolved)

    return TransferDataPaths(
        scf_data_file=scf_data_file,
        hfcs_microdata_file=hfcs_microdata_file,
        hfcs_access_form=hfcs_access_pdf,
        hfcs_stat_tables_zip=hfcs_stats_zip,
        raw_dir=raw_path,
    )


def load_dataset(path: str | Path, nrows: int | None = None) -> pd.DataFrame:
    source = Path(path)
    suffix = source.suffix.lower()

    if suffix == "":
        # Try common tabular readers for extensionless files.
        try:
            return pd.read_stata(source, convert_categoricals=False)
        except Exception:  # noqa: BLE001
            try:
                return pd.read_csv(source, nrows=nrows, low_memory=False)
            except Exception as exc:  # noqa: BLE001
                raise ValueError(f"No se pudo inferir formato del archivo sin extension: {source}") from exc

    if suffix == ".csv":
        return pd.read_csv(source, nrows=nrows, low_memory=False)
    if suffix == ".dta":
        return pd.read_stata(source, convert_categoricals=False)
    if suffix == ".parquet":
        return pd.read_parquet(source)
    if suffix == ".sas7bdat":
        return pd.read_sas(source)

    raise ValueError(f"Formato no soportado: {source}")


def inspect_dataset(df: pd.DataFrame, name: str) -> dict[str, Any]:
    missing = df.isna().mean().sort_values(ascending=False)
    out = {
        "name": name,
        "shape": df.shape,
        "n_columns": int(df.shape[1]),
        "sample_columns": list(df.columns[:40]),
        "dtype_counts": df.dtypes.astype(str).value_counts().to_dict(),
        "top_missing": missing.head(15).rename("missing_ratio").to_frame(),
    }
    return out


def _build_casefold_map(columns: list[str]) -> dict[str, str]:
    return {c.lower(): c for c in columns}


def _pick_alias(df: pd.DataFrame, aliases: list[str]) -> str | None:
    lower_map = _build_casefold_map(df.columns.tolist())
    for alias in aliases:
        if alias.lower() in lower_map:
            return lower_map[alias.lower()]
    return None


def infer_feature_mapping(
    scf_df: pd.DataFrame,
    hfcs_df: pd.DataFrame,
    manual_scf_mapping: dict[str, str] | None = None,
    manual_hfcs_mapping: dict[str, str] | None = None,
) -> dict[str, dict[str, str | None]]:
    manual_scf_mapping = manual_scf_mapping or {}
    manual_hfcs_mapping = manual_hfcs_mapping or {}

    mapping: dict[str, dict[str, str | None]] = {}
    for feature, aliases in FEATURE_ALIASES.items():
        scf_col = manual_scf_mapping.get(feature) or _pick_alias(scf_df, aliases)
        hfcs_col = manual_hfcs_mapping.get(feature) or _pick_alias(hfcs_df, aliases)
        mapping[feature] = {"scf": scf_col, "hfcs": hfcs_col}
    return mapping


def _to_numeric(series: pd.Series) -> pd.Series:
    if pd.api.types.is_numeric_dtype(series):
        return pd.to_numeric(series, errors="coerce")

    mapped = (
        series.astype(str)
        .str.strip()
        .replace(
            {
                "": np.nan,
                "nan": np.nan,
                "none": np.nan,
                "yes": "1",
                "no": "0",
                "true": "1",
                "false": "0",
            }
        )
    )
    return pd.to_numeric(mapped, errors="coerce")


def _normalize_education(series: pd.Series) -> pd.Series:
    # If already numeric, keep as ordinal numeric scale.
    numeric = pd.to_numeric(series, errors="coerce")
    if numeric.notna().mean() > 0.7:
        return numeric

    s = series.astype(str).str.lower().str.strip()
    mapping = {
        "primary": 1,
        "basic": 1,
        "lower": 1,
        "secondary": 2,
        "high school": 2,
        "upper": 2,
        "tertiary": 3,
        "university": 3,
        "college": 3,
        "postgraduate": 4,
        "master": 4,
        "phd": 4,
    }
    out = pd.Series(np.nan, index=series.index, dtype=float)
    for key, val in mapping.items():
        out[s.str.contains(key, na=False)] = float(val)
    return out


def _normalize_stock_ownership(series: pd.Series) -> pd.Series:
    numeric = pd.to_numeric(series, errors="coerce")
    if numeric.notna().mean() > 0.6:
        return (numeric > 0).astype(float)

    s = series.astype(str).str.lower().str.strip()
    yes_tokens = {"yes", "y", "1", "true", "owner", "owns"}
    out = s.isin(yes_tokens).astype(float)
    out[s.isin({"", "nan", "none"})] = np.nan
    return out


def build_aligned_feature_frames(
    scf_df: pd.DataFrame,
    hfcs_df: pd.DataFrame,
    mapping: dict[str, dict[str, str | None]],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    missing = [
        f
        for f, cols in mapping.items()
        if cols.get("scf") is None or cols.get("hfcs") is None
    ]
    if missing:
        raise ValueError(
            "No se pudieron alinear todas las features obligatorias. "
            f"Faltantes: {missing}. Pasa mapping manual para esas columnas."
        )

    scf_out = pd.DataFrame(index=scf_df.index)
    hfcs_out = pd.DataFrame(index=hfcs_df.index)

    for feat, cols in mapping.items():
        scf_col = str(cols["scf"])
        hfcs_col = str(cols["hfcs"])

        if feat == "education_level":
            scf_out[feat] = _normalize_education(scf_df[scf_col])
            hfcs_out[feat] = _normalize_education(hfcs_df[hfcs_col])
        elif feat == "stock_ownership":
            scf_out[feat] = _normalize_stock_ownership(scf_df[scf_col])
            hfcs_out[feat] = _normalize_stock_ownership(hfcs_df[hfcs_col])
        else:
            scf_out[feat] = _to_numeric(scf_df[scf_col])
            hfcs_out[feat] = _to_numeric(hfcs_df[hfcs_col])

    return scf_out, hfcs_out


def build_source_risk_labels(source_df: pd.DataFrame) -> pd.Series:
    income = source_df["income"].abs().fillna(0.0)
    wealth = source_df["net_wealth"].clip(lower=0).fillna(0.0)
    stock = source_df["stock_ownership"].fillna(0.0)

    # Proxy risk exposure: risky ownership weighted by wealth participation.
    risky_exposure = stock * (wealth / (wealth + income + 1.0))
    risky_exposure = risky_exposure.fillna(0.0)

    if risky_exposure.nunique(dropna=True) < 5:
        rank_pct = risky_exposure.rank(pct=True, method="average")
        bins = np.clip(np.ceil(rank_pct * 5), 1, 5).astype(int)
        return pd.Series(bins, index=source_df.index, name="risk_level")

    labels = pd.qcut(risky_exposure, q=5, labels=[1, 2, 3, 4, 5], duplicates="drop")
    if labels.isna().any() or len(pd.unique(labels.dropna())) < 5:
        rank_pct = risky_exposure.rank(pct=True, method="average")
        labels = np.clip(np.ceil(rank_pct * 5), 1, 5)
    return pd.Series(labels.astype(int), index=source_df.index, name="risk_level")


def _fit_transform_source_target(
    X_source: pd.DataFrame,
    X_target: pd.DataFrame,
) -> tuple[np.ndarray, np.ndarray, Pipeline]:
    prep = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )
    Xs = prep.fit_transform(X_source)
    Xt = prep.transform(X_target)
    return Xs, Xt, prep


def _prior_shift_correct(
    probs: np.ndarray,
    source_prior: np.ndarray,
    target_prior_est: np.ndarray | None = None,
) -> np.ndarray:
    p = np.clip(probs, 1e-9, 1.0)
    src = np.clip(source_prior, 1e-9, 1.0)
    if target_prior_est is None:
        target_prior_est = p.mean(axis=0)
    tgt = np.clip(target_prior_est, 1e-9, 1.0)
    adjusted = p * (src / tgt)
    adjusted = adjusted / adjusted.sum(axis=1, keepdims=True)
    return adjusted


def _compute_domain_weights(Xs: np.ndarray, Xt: np.ndarray) -> np.ndarray:
    X_dom = np.vstack([Xs, Xt])
    y_dom = np.r_[np.zeros(len(Xs), dtype=int), np.ones(len(Xt), dtype=int)]

    clf = LogisticRegression(max_iter=2000, class_weight="balanced", random_state=RANDOM_STATE)
    clf.fit(X_dom, y_dom)
    p_target = clf.predict_proba(Xs)[:, 1]
    p_source = 1.0 - p_target
    w = p_target / np.clip(p_source, 1e-6, None)
    return np.clip(w, 0.1, 10.0)


def _fit_multiclass_xgb(
    X_train: np.ndarray,
    y_train: np.ndarray,
    sample_weight: np.ndarray | None = None,
) -> XGBClassifier:
    model = XGBClassifier(
        objective="multi:softprob",
        num_class=5,
        n_estimators=350,
        max_depth=4,
        learning_rate=0.05,
        subsample=0.9,
        colsample_bytree=0.8,
        reg_lambda=1.0,
        random_state=RANDOM_STATE,
        n_jobs=4,
        eval_metric="mlogloss",
    )
    model.fit(X_train, y_train, sample_weight=sample_weight)
    return model


def _fit_ordinal_models(
    X_train: np.ndarray,
    y_train_class: np.ndarray,
    sample_weight: np.ndarray | None = None,
) -> list[XGBClassifier]:
    models: list[XGBClassifier] = []
    for threshold in [2, 3, 4, 5]:
        y_bin = (y_train_class >= threshold).astype(int)
        model = XGBClassifier(
            objective="binary:logistic",
            n_estimators=300,
            max_depth=4,
            learning_rate=0.05,
            subsample=0.9,
            colsample_bytree=0.8,
            reg_lambda=1.0,
            random_state=RANDOM_STATE,
            n_jobs=4,
            eval_metric="logloss",
        )
        model.fit(X_train, y_bin, sample_weight=sample_weight)
        models.append(model)
    return models


def _ordinal_to_class_probs(models: list[XGBClassifier], X: np.ndarray) -> np.ndarray:
    c2 = models[0].predict_proba(X)[:, 1]
    c3 = models[1].predict_proba(X)[:, 1]
    c4 = models[2].predict_proba(X)[:, 1]
    c5 = models[3].predict_proba(X)[:, 1]

    c3 = np.minimum(c3, c2)
    c4 = np.minimum(c4, c3)
    c5 = np.minimum(c5, c4)

    p1 = 1.0 - c2
    p2 = c2 - c3
    p3 = c3 - c4
    p4 = c4 - c5
    p5 = c5

    probs = np.vstack([p1, p2, p3, p4, p5]).T
    probs = np.clip(probs, 1e-9, None)
    probs = probs / probs.sum(axis=1, keepdims=True)
    return probs


def _distribution_from_classes(classes_1_to_5: np.ndarray) -> pd.Series:
    s = pd.Series(classes_1_to_5).value_counts(normalize=True).reindex([1, 2, 3, 4, 5], fill_value=0.0)
    s.index.name = "risk_level"
    s.name = "distribution"
    return s


def _evaluate_transfer(
    source_dist: pd.Series,
    target_dist: pd.Series,
) -> dict[str, Any]:
    diff = (source_dist - target_dist).abs()
    max_gap = float(diff.max())
    collapse = bool((target_dist == 0).any())
    represented = int((target_dist > 0).sum())
    return {
        "source_distribution": source_dist,
        "target_distribution": target_dist,
        "max_gap": max_gap,
        "collapse": collapse,
        "represented_classes": represented,
    }


def run_transfer_experiment(
    scf_df: pd.DataFrame,
    hfcs_df: pd.DataFrame,
    mapping: dict[str, dict[str, str | None]],
    apply_prior_shift: bool = True,
) -> dict[str, Any]:
    X_scf_raw, X_hfcs_raw = build_aligned_feature_frames(scf_df, hfcs_df, mapping)
    y_scf = build_source_risk_labels(X_scf_raw)

    Xs, Xt, prep = _fit_transform_source_target(X_scf_raw, X_hfcs_raw)
    source_prior = _distribution_from_classes(y_scf.values).values

    # Keep a source holdout only for sanity checks.
    X_train, X_test, y_train, y_test = train_test_split(
        Xs,
        y_scf.values,
        test_size=0.2,
        random_state=RANDOM_STATE,
        stratify=y_scf.values,
    )

    label_encoder = LabelEncoder()
    y_train_enc = label_encoder.fit_transform(y_train)
    y_test_enc = label_encoder.transform(y_test)

    class_to_encoded = {cls: idx for idx, cls in enumerate(label_encoder.classes_)}
    encoded_to_class = {idx: cls for cls, idx in class_to_encoded.items()}

    def enc_probs_to_class_probs(p_enc: np.ndarray) -> np.ndarray:
        out = np.zeros((p_enc.shape[0], 5), dtype=float)
        for enc_idx, cls in encoded_to_class.items():
            out[:, int(cls) - 1] = p_enc[:, enc_idx]
        out = np.clip(out, 1e-9, None)
        out = out / out.sum(axis=1, keepdims=True)
        return out

    results: dict[str, dict[str, Any]] = {}

    baseline_model = _fit_multiclass_xgb(X_train, y_train_enc)
    baseline_test_enc = baseline_model.predict(X_test)
    baseline_test_class = np.array([encoded_to_class[int(c)] for c in baseline_test_enc])
    baseline_test_proba_enc = baseline_model.predict_proba(X_test)
    baseline_hfcs_proba_enc = baseline_model.predict_proba(Xt)

    baseline_hfcs_probs = enc_probs_to_class_probs(baseline_hfcs_proba_enc)
    if apply_prior_shift:
        baseline_hfcs_probs = _prior_shift_correct(baseline_hfcs_probs, source_prior)
    baseline_hfcs_pred = baseline_hfcs_probs.argmax(axis=1) + 1

    source_dist = _distribution_from_classes(y_scf.values)
    baseline_eval = _evaluate_transfer(source_dist, _distribution_from_classes(baseline_hfcs_pred))
    baseline_eval["test_accuracy"] = float(accuracy_score(y_test, baseline_test_class))
    baseline_eval["test_f1_macro"] = float(f1_score(y_test, baseline_test_class, average="macro"))
    results["baseline_transfer"] = baseline_eval

    # Prior shift only (explicit baseline probabilities before/after).
    baseline_raw_pred = enc_probs_to_class_probs(baseline_hfcs_proba_enc).argmax(axis=1) + 1
    prior_eval = _evaluate_transfer(source_dist, _distribution_from_classes(baseline_hfcs_pred))
    prior_eval["raw_target_distribution"] = _distribution_from_classes(baseline_raw_pred)
    results["prior_shift_correction"] = prior_eval

    # Covariate shift weighting.
    domain_weights = _compute_domain_weights(X_train, Xt)
    weighted_model = _fit_multiclass_xgb(X_train, y_train_enc, sample_weight=domain_weights)
    weighted_hfcs_probs = enc_probs_to_class_probs(weighted_model.predict_proba(Xt))
    if apply_prior_shift:
        weighted_hfcs_probs = _prior_shift_correct(weighted_hfcs_probs, source_prior)
    weighted_hfcs_pred = weighted_hfcs_probs.argmax(axis=1) + 1
    results["covariate_shift_weighted"] = _evaluate_transfer(
        source_dist,
        _distribution_from_classes(weighted_hfcs_pred),
    )

    # Ordinal model.
    ordinal_models = _fit_ordinal_models(X_train, y_train, sample_weight=None)
    ordinal_hfcs_probs = _ordinal_to_class_probs(ordinal_models, Xt)
    if apply_prior_shift:
        ordinal_hfcs_probs = _prior_shift_correct(ordinal_hfcs_probs, source_prior)
    ordinal_hfcs_pred = ordinal_hfcs_probs.argmax(axis=1) + 1
    results["ordinal"] = _evaluate_transfer(source_dist, _distribution_from_classes(ordinal_hfcs_pred))

    # Combined ordinal + weighting.
    ordinal_models_w = _fit_ordinal_models(X_train, y_train, sample_weight=domain_weights)
    ordinal_weighted_probs = _ordinal_to_class_probs(ordinal_models_w, Xt)
    if apply_prior_shift:
        ordinal_weighted_probs = _prior_shift_correct(ordinal_weighted_probs, source_prior)
    ordinal_weighted_pred = ordinal_weighted_probs.argmax(axis=1) + 1
    results["combined_ordinal_covariate"] = _evaluate_transfer(
        source_dist,
        _distribution_from_classes(ordinal_weighted_pred),
    )

    summary_rows = []
    for name, res in results.items():
        notes = []
        if name == "baseline_transfer":
            notes.append("Direct model transfer")
        if name == "prior_shift_correction":
            notes.append("Posterior prior correction")
        if name == "covariate_shift_weighted":
            notes.append("Domain importance weighting")
        if name == "ordinal":
            notes.append("Cumulative binary formulation")
        if name == "combined_ordinal_covariate":
            notes.append("Ordinal + domain weighting")

        summary_rows.append(
            {
                "method": name,
                "max_gap": float(res["max_gap"]),
                "collapse": bool(res["collapse"]),
                "represented_classes": int(res["represented_classes"]),
                "notes": "; ".join(notes),
            }
        )

    summary_df = pd.DataFrame(summary_rows).sort_values("max_gap", ascending=True).reset_index(drop=True)
    best_method = str(summary_df.iloc[0]["method"])

    return {
        "mapping": mapping,
        "source_distribution": source_dist,
        "results": results,
        "summary_df": summary_df,
        "best_method": best_method,
        "best_result": results[best_method],
        "X_scf_shape": X_scf_raw.shape,
        "X_hfcs_shape": X_hfcs_raw.shape,
        "preprocessor": prep,
    }


def plot_transfer_distributions(
    source_distribution: pd.Series,
    results: dict[str, dict[str, Any]],
    save_path: str | Path = "data/outputs/scf_hfcs_transfer_comparison.png",
) -> Path:
    save = _resolve_path(save_path, "data/outputs/scf_hfcs_transfer_comparison.png")
    save.parent.mkdir(parents=True, exist_ok=True)

    methods = list(results.keys())
    n_methods = len(methods)
    ncols = 2
    nrows = int(np.ceil(n_methods / ncols))

    fig, axes = plt.subplots(nrows, ncols, figsize=(14, 4 * nrows), squeeze=False)
    axes_flat = axes.flatten()

    x = np.arange(1, 6)
    width = 0.35

    for i, method in enumerate(methods):
        ax = axes_flat[i]
        target_dist = results[method]["target_distribution"]

        ax.bar(x - width / 2, source_distribution.values, width=width, label="SCF source", color="#1f77b4")
        ax.bar(x + width / 2, target_dist.values, width=width, label="HFCS predicted", color="#ff7f0e")
        ax.set_title(f"{method} | max_gap={results[method]['max_gap']:.3f}")
        ax.set_xlabel("Risk class")
        ax.set_ylabel("Proportion")
        ax.set_xticks(x)
        ax.legend(loc="best")

    for j in range(i + 1, len(axes_flat)):
        axes_flat[j].axis("off")

    plt.tight_layout()
    plt.savefig(save, dpi=200, bbox_inches="tight")
    plt.close(fig)
    return save


def start_download_and_inspection(
    raw_dir: str | Path = "data/raw/scf_hfcs",
    hfcs_nrows: int = 50000,
) -> dict[str, Any]:
    """Step 1 helper: download + inspect SCF and HFCS inputs.

    HFCS microdata are restricted. If unavailable, this returns access artifacts and
    an actionable message while still inspecting SCF.
    """
    paths = download_scf_hfcs_data(raw_dir=raw_dir, force=False)
    scf_df = load_dataset(paths.scf_data_file)
    scf_info = inspect_dataset(scf_df, "SCF microdata")

    hfcs_info = None
    hfcs_df = None
    message = ""

    if paths.hfcs_microdata_file is not None:
        hfcs_df = load_dataset(paths.hfcs_microdata_file, nrows=hfcs_nrows)
        hfcs_info = inspect_dataset(hfcs_df, "HFCS microdata")
        message = "HFCS microdata found and loaded from HFCS_MICRODATA_PATH."
    else:
        message = (
            "HFCS microdata are access-restricted by ECB. "
            "Downloaded official access form and statistical tables. "
            "Set HFCS_MICRODATA_PATH in .env once access is granted."
        )

    return {
        "paths": paths,
        "scf_info": scf_info,
        "hfcs_info": hfcs_info,
        "scf_df": scf_df,
        "hfcs_df": hfcs_df,
        "message": message,
    }
