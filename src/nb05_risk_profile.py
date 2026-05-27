"""
nb05_risk_profile.py — Client risk profiling: feature extraction, scoring, classifier training.
"""
import json
import os
import warnings
from pathlib import Path
from typing import Dict, List, Tuple

import joblib
import numpy as np
import pandas as pd
import sys
import matplotlib
if 'ipykernel' not in sys.modules:
    matplotlib.use('Agg')
import matplotlib.pyplot as plt

warnings.filterwarnings("ignore")

FEATURE_MAP = {
    "S_Age":       "age_group",
    "S_Education": "education",
    "S_Income":    "income_group",
    "S_Gender2":   "gender",
    "G1":  "fin_literacy_objective",
    "G2":  "fin_knowledge_self",
    "B30": "risk_willingness",
    "B31": "investment_horizon",
    "B3":  "investment_approach",
    "B4":  "portfolio_diversity",
    "E5":  "emergency_fund",
    "E6":  "spending_vs_income",
    "D2":  "debt_confidence",
    "D3":  "too_much_debt",
}
LEVEL_NAMES = {0: "Very Conservative", 1: "Conservative", 2: "Moderate",
               3: "Aggressive", 4: "Very Aggressive"}
POS_RISK = ["risk_willingness", "investment_horizon", "portfolio_diversity",
            "fin_literacy_objective", "fin_knowledge_self", "income_group"]
NEG_RISK = ["age_group", "too_much_debt"]


def load_nfcs_data(raw_dir: Path) -> pd.DataFrame:
    """Load and clean NFCS 2021 investor survey."""
    from sklearn.preprocessing import MinMaxScaler
    nfcs_path = raw_dir / "investor_2021" / "NFCS 2021 Investor Data 221121.csv"
    df_raw = pd.read_csv(nfcs_path)
    print(f"NFCS dataset: {df_raw.shape[0]:,} respondents, {df_raw.shape[1]} variables")

    available = {k: v for k, v in FEATURE_MAP.items() if k in df_raw.columns}
    df = df_raw[list(available.keys())].rename(columns=available).copy()
    print(f"Features selected: {list(available.values())}")

    for col in df.columns:
        df[col] = pd.to_numeric(df[col], errors="coerce")
        df.loc[df[col].isin([98, 99]), col] = np.nan

    n_before = len(df)
    df = df.dropna(thresh=len(df.columns) // 2)
    print(f"Dropped {n_before - len(df)} rows with >50% missing")
    for col in df.columns:
        if df[col].isnull().any():
            df[col] = df[col].fillna(df[col].median())
    return df


def compute_risk_scores(df: pd.DataFrame) -> pd.DataFrame:
    """Add risk_score [0,1] and risk_level [0-4] columns."""
    from sklearn.preprocessing import MinMaxScaler
    feature_cols = [c for c in df.columns if c not in ["risk_score", "risk_level"]]
    scaler_mm = MinMaxScaler()
    df_sc = pd.DataFrame(scaler_mm.fit_transform(df[feature_cols]),
                         columns=feature_cols, index=df.index)

    pos_avail = [f for f in POS_RISK if f in df_sc.columns]
    neg_avail = [f for f in NEG_RISK if f in df_sc.columns]
    risk_score = pd.Series(0.0, index=df.index)
    if pos_avail: risk_score += df_sc[pos_avail].mean(axis=1)
    if neg_avail: risk_score -= df_sc[neg_avail].mean(axis=1) * 0.5
    risk_score = (risk_score - risk_score.min()) / (risk_score.max() - risk_score.min())

    df = df.copy()
    df["risk_score"] = risk_score
    df["risk_level"] = pd.qcut(risk_score, q=5, labels=[0, 1, 2, 3, 4],
                                duplicates="drop").astype(int)
    print("Risk level distribution:")
    for lvl in range(5):
        n = (df["risk_level"] == lvl).sum()
        print(f"  Level {lvl} ({LEVEL_NAMES[lvl]}): {n:,} ({n/len(df)*100:.1f}%)")
    return df


def train_classifiers(
    df: pd.DataFrame,
    feature_cols: List[str],
) -> Tuple[object, pd.DataFrame, object]:
    """Train LR, RF, XGBoost classifiers and return (best_model, metrics_df, scaler)."""
    from sklearn.model_selection import train_test_split, StratifiedKFold, GridSearchCV
    from sklearn.preprocessing import StandardScaler
    from sklearn.linear_model import LogisticRegression
    from sklearn.ensemble import RandomForestClassifier
    from xgboost import XGBClassifier
    from sklearn.metrics import accuracy_score, f1_score
    import seaborn as sns

    X = df[feature_cols].values
    y = df["risk_level"].values

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y)
    print(f"Train: {X_train.shape[0]:,} | Test: {X_test.shape[0]:,}")

    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)
    X_test_s  = scaler.transform(X_test)

    cv = StratifiedKFold(5, shuffle=True, random_state=42)

    print("Training Logistic Regression...")
    lr_gs = GridSearchCV(
        LogisticRegression(solver="lbfgs", random_state=42),
        {"C": [0.01, 0.1, 1.0, 10.0], "max_iter": [1000]},
        cv=cv, scoring="f1_weighted", n_jobs=-1)
    lr_gs.fit(X_train_s, y_train)

    print("Training Random Forest...")
    rf_gs = GridSearchCV(
        RandomForestClassifier(random_state=42, class_weight="balanced"),
        {"n_estimators": [100, 200], "max_depth": [5, 10, 15], "min_samples_leaf": [5, 10]},
        cv=cv, scoring="f1_weighted", n_jobs=-1)
    rf_gs.fit(X_train_s, y_train)

    print("Training XGBoost...")
    xgb_gs = GridSearchCV(
        XGBClassifier(random_state=42, eval_metric="mlogloss", verbosity=0),
        {"n_estimators": [100, 200], "max_depth": [3, 5], "learning_rate": [0.05, 0.1]},
        cv=cv, scoring="f1_weighted", n_jobs=-1)
    xgb_gs.fit(X_train_s, y_train)

    models = {"Logistic Regression": lr_gs.best_estimator_,
              "Random Forest":       rf_gs.best_estimator_,
              "XGBoost":             xgb_gs.best_estimator_}
    results = []
    for name, model in models.items():
        y_pred = model.predict(X_test_s)
        acc_te = accuracy_score(y_test, y_pred)
        acc_tr = accuracy_score(y_train, model.predict(X_train_s))
        f1_w   = f1_score(y_test, y_pred, average="weighted")
        mae    = np.mean(np.abs(y_pred.astype(float) - y_test.astype(float)))
        results.append({"model": name, "train_acc": acc_tr, "test_acc": acc_te,
                         "f1_weighted": f1_w, "mae_ordinal": mae, "overfit": acc_tr - acc_te})
        print(f"  {name}: Train={acc_tr:.4f} Test={acc_te:.4f} F1={f1_w:.4f} MAE={mae:.4f}")

    metrics_df = pd.DataFrame(results)
    best_idx   = metrics_df.sort_values(["f1_weighted", "mae_ordinal"],
                                         ascending=[False, True]).index[0]
    best_name  = metrics_df.loc[best_idx, "model"]
    best_model = models[best_name]
    print(f"\nBest model: {best_name}")

    # Store test data for confusion matrix
    best_model._test_data = (X_test_s, y_test, scaler, feature_cols)
    return best_model, metrics_df, scaler, feature_cols


def plot_evaluation(best_model, y_test: np.ndarray, X_test_s: np.ndarray,
                    best_name: str, fig_dir: Path) -> None:
    """Confusion matrix + per-class metrics plot."""
    import seaborn as sns
    from sklearn.metrics import confusion_matrix, classification_report

    y_pred = best_model.predict(X_test_s)
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    sns.heatmap(confusion_matrix(y_test, y_pred), annot=True, fmt="d",
                cmap="Blues", ax=axes[0],
                xticklabels=range(1, 6), yticklabels=range(1, 6))
    axes[0].set_xlabel("Predicho"); axes[0].set_ylabel("Real")
    axes[0].set_title(f"Matriz de confusión ({best_name})")
    rep = classification_report(y_test, y_pred, output_dict=True)
    pd.DataFrame(rep).T.iloc[:5][["precision", "recall", "f1-score"]] \
        .plot.bar(ax=axes[1], edgecolor="black", alpha=0.8)
    axes[1].set_title("Métricas por clase"); axes[1].set_ylim(0, 1.05)
    plt.tight_layout()
    out = fig_dir / "05_model_evaluation.png"
    plt.savefig(out, dpi=150)
    print(f"Saved: {out}")
    return fig


def build_predict_fn(best_model, scaler, feature_cols: List[str]):
    """Return a callable predict_risk_profile(client_features: dict) -> dict."""
    def predict_risk_profile(client_features: dict) -> dict:
        x = np.array([[client_features.get(f, 0.0) for f in feature_cols]])
        x_s   = scaler.transform(x)
        level = int(best_model.predict(x_s)[0])
        probas = best_model.predict_proba(x_s)[0]
        return {"risk_level": level, "risk_profile": LEVEL_NAMES.get(level, str(level)),
                "confidence": float(probas.max()),
                "probabilities": {int(c): float(p)
                                  for c, p in zip(best_model.classes_, probas)}}
    return predict_risk_profile


def save_outputs(
    best_model, scaler, feature_cols: List[str],
    best_name: str, metrics_df: pd.DataFrame,
    client_profiles: pd.DataFrame,
    processed_dir: Path,
) -> None:
    """Persist risk model, preprocessor, metrics, and client profiles."""
    joblib.dump(best_model, processed_dir / "risk_model.pkl")
    joblib.dump({"scaler": scaler, "feature_cols": feature_cols,
                 "model_name": best_name, "classes": list(best_model.classes_)},
                processed_dir / "risk_preprocessor.pkl")
    best_result = metrics_df[metrics_df["model"] == best_name].iloc[0].to_dict()
    with open(processed_dir / "risk_model_metrics.json", "w") as f:
        json.dump(best_result, f, indent=2, default=str)
    client_profiles.to_parquet(processed_dir / "client_profiles.parquet", index=False)
    print("Saved: risk_model.pkl | risk_preprocessor.pkl")
    print("       risk_model_metrics.json | client_profiles.parquet")


def load_nb06_results(results_dir: Path) -> Tuple[pd.DataFrame, str]:
    """Load nb06 model results (priority: final > v15 > v14 > v13)."""
    for tag, fname in [("final", "nb06_h22_model_results.parquet"),
                       ("v15",   "nb06v15_h22_model_results.parquet"),
                       ("v14",   "nb06v14_h22_model_results.parquet"),
                       ("v13",   "nb06v13_h22_model_results.parquet")]:
        p = results_dir / fname
        if p.exists():
            df = pd.read_parquet(p)
            print(f"Loaded nb06 {tag}: {df.shape}")
            return df, tag
    raise FileNotFoundError("No nb06 results found.")
