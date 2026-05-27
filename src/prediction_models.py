"""
Portfolio Growth Prediction Models (M1, M2, M3).

Implements three predictive models for forecasting portfolio returns:
- M1: Temporal Baseline (Linear/Ridge Regression)
- M2: XGBoost Regressor
- M3: LSTM Neural Network

Target: Future portfolio growth (%) at horizon h (default: 63 trading days = 3 months)

No data leakage: Uses only past information for features.
Time-based train/val/test split: 70% train, 15% val, 15% test.
"""

import logging
import pickle
from typing import Tuple, Dict, Any
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.linear_model import Ridge
from xgboost import XGBRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, mean_absolute_percentage_error

try:
    import tensorflow as tf
    from tensorflow import keras
    from tensorflow.keras.layers import LSTM, Dense, Dropout
    from tensorflow.keras.models import Sequential
    LSTM_AVAILABLE = True
except ImportError:
    LSTM_AVAILABLE = False
    logging.warning("TensorFlow/Keras not available. LSTM model will be skipped.")

logger = logging.getLogger(__name__)

RANDOM_STATE = 42
np.random.seed(RANDOM_STATE)


# ============================================================================
# DATA PREPARATION
# ============================================================================

def build_supervised_dataset(
    portfolio_returns: pd.DataFrame,
    hmm_states: pd.Series,
    hmm_probs: pd.DataFrame,
    horizon: int = 63,
    lookback: int = 60,
) -> pd.DataFrame:
    """
    Build supervised dataset for portfolio growth prediction.
    
    Parameters
    ----------
    portfolio_returns : pd.DataFrame
        Daily returns, shape (n_dates, n_portfolios).
        Index is DatetimeIndex.
    hmm_states : pd.Series
        HMM hidden state per date, indexed by date.
    hmm_probs : pd.DataFrame
        HMM state probabilities per date (n_dates, n_states).
        Index is DatetimeIndex.
    horizon : int
        Forecast horizon in trading days (default: 63 = 3 months).
    lookback : int
        Historical lookback window for rolling features (default: 60 days).
    
    Returns
    -------
    supervised_df : pd.DataFrame
        Rows = one row per date-portfolio observation
        Columns = features + target
        Columns:
        - date
        - portfolio_id
        - lagged_return_1, ..., lagged_return_10
        - rolling_mean_return
        - rolling_volatility
        - rolling_sharpe
        - hmm_state
        - hmm_prob_* (one per hidden state)
        - future_growth_h (TARGET)
    """
    logger.info(f"Building supervised dataset: horizon={horizon}, lookback={lookback}")
    
    n_portfolios = portfolio_returns.shape[1]
    n_dates = portfolio_returns.shape[0]
    
    dataset_rows = []
    
    for idx in range(lookback, n_dates - horizon):
        # This is the observation date t
        date_t = portfolio_returns.index[idx]
        
        # Future date t+h
        future_idx = idx + horizon
        date_future = portfolio_returns.index[future_idx]
        
        for pid in range(n_portfolios):
            port_name = portfolio_returns.columns[pid]
            
            # --- HISTORICAL FEATURES (using dates up to t) ---
            
            # Lagged returns (last 10 days)
            hist_returns = portfolio_returns.iloc[idx-10:idx, pid].values
            lagged_returns = np.pad(hist_returns, (10 - len(hist_returns), 0), mode='edge')[-10:]
            
            # Rolling features (last lookback days)
            window_returns = portfolio_returns.iloc[idx-lookback:idx, pid].values
            rolling_mean = np.mean(window_returns)
            rolling_vol = np.std(window_returns)
            rolling_sharpe = rolling_mean / rolling_vol if rolling_vol > 0 else 0.0
            
            # HMM state at time t
            hmm_state_t = hmm_states.iloc[idx] if pid == 0 else hmm_states.iloc[idx]  # Same for all portfolios
            
            # HMM probabilities at time t
            hmm_prob_t = hmm_probs.iloc[idx].values
            
            # --- TARGET: Future portfolio growth ---
            # growth_h = (V_{t+h} - V_t) / V_t ≈ sum of returns from t to t+h
            future_returns = portfolio_returns.iloc[idx+1:future_idx+1, pid].values
            future_growth = np.sum(future_returns)  # Cumulative log return ≈ % growth
            
            # Build row
            row = {
                'date': date_t,
                'portfolio_id': pid,
                'future_date': date_future,
            }
            
            # Lagged returns
            for lag in range(1, 11):
                row[f'lagged_return_{lag}'] = lagged_returns[lag-1]
            
            # Rolling features
            row['rolling_mean_return'] = rolling_mean
            row['rolling_volatility'] = rolling_vol
            row['rolling_sharpe'] = rolling_sharpe
            
            # HMM features
            row['hmm_state'] = hmm_state_t
            for state_idx in range(len(hmm_prob_t)):
                row[f'hmm_prob_{state_idx}'] = hmm_prob_t[state_idx]
            
            # Target
            row['future_growth'] = future_growth
            
            dataset_rows.append(row)
    
    supervised_df = pd.DataFrame(dataset_rows)
    logger.info(f"✓ Built dataset: shape {supervised_df.shape}")
    
    return supervised_df


def split_temporal(
    df: pd.DataFrame,
    test_frac: float = 0.15,
    val_frac: float = 0.15,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Time-aware train/val/test split.
    
    Ensures no data leakage by preserving temporal order.
    
    Parameters
    ----------
    df : pd.DataFrame
        Supervised dataset, sorted by date.
    test_frac : float
        Fraction of data for test set.
    val_frac : float
        Fraction of data for validation set.
    
    Returns
    -------
    train_df, val_df, test_df : tuple of pd.DataFrame
        Three data frames maintaining temporal order.
    """
    n = len(df)
    train_end = int(n * (1 - test_frac - val_frac))
    val_end = train_end + int(n * val_frac)
    
    train_df = df.iloc[:train_end]
    val_df = df.iloc[train_end:val_end]
    test_df = df.iloc[val_end:]
    
    logger.info(f"Time-based split:")
    logger.info(f"  Train: {train_df.shape[0]} samples ({len(train_df)/n*100:.1f}%)")
    logger.info(f"  Val:   {val_df.shape[0]} samples ({len(val_df)/n*100:.1f}%)")
    logger.info(f"  Test:  {test_df.shape[0]} samples ({len(test_df)/n*100:.1f}%)")
    
    return train_df, val_df, test_df


# ============================================================================
# M1: TEMPORAL BASELINE (RIDGE REGRESSION)
# ============================================================================

class M1_TemporalBaseline:
    """
    M1: Simple temporal baseline using Ridge Regression.
    
    Features: lagged returns + rolling statistics + HMM state.
    Simple, interpretable, fast to train.
    """
    
    def __init__(self, alpha: float = 1.0, random_state: int = RANDOM_STATE):
        self.alpha = alpha
        self.random_state = random_state
        self.model = Ridge(alpha=alpha, random_state=random_state)
        self.scaler = StandardScaler()
        self.feature_cols = None
        
    def fit(self, X: pd.DataFrame, y: np.ndarray) -> None:
        """Train on supervised data."""
        self.feature_cols = X.columns.tolist()
        X_scaled = self.scaler.fit_transform(X)
        self.model.fit(X_scaled, y)
        logger.info(f"✓ M1 fitted on {X.shape[0]} samples, {X.shape[1]} features")
        
    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """Make predictions."""
        X_scaled = self.scaler.transform(X)
        return self.model.predict(X_scaled)
    
    def save(self, path: Path) -> None:
        """Save model to disk."""
        with open(path, 'wb') as f:
            pickle.dump({'model': self.model, 'scaler': self.scaler, 'features': self.feature_cols}, f)
        logger.info(f"✓ M1 saved to {path}")
    
    @staticmethod
    def load(path: Path) -> 'M1_TemporalBaseline':
        """Load model from disk."""
        with open(path, 'rb') as f:
            data = pickle.load(f)
        m = M1_TemporalBaseline()
        m.model = data['model']
        m.scaler = data['scaler']
        m.feature_cols = data['features']
        return m


# ============================================================================
# M2: XGBOOST REGRESSOR
# ============================================================================

class M2_XGBoost:
    """
    M2: XGBoost for tabular growth prediction.
    
    Same features as M1, but non-linear gradient boosting.
    """
    
    def __init__(
        self,
        n_estimators: int = 100,
        max_depth: int = 5,
        learning_rate: float = 0.1,
        subsample: float = 0.8,
        random_state: int = RANDOM_STATE,
    ):
        self.hyperparams = {
            'n_estimators': n_estimators,
            'max_depth': max_depth,
            'learning_rate': learning_rate,
            'subsample': subsample,
        }
        self.random_state = random_state
        self.model = XGBRegressor(
            n_estimators=n_estimators,
            max_depth=max_depth,
            learning_rate=learning_rate,
            subsample=subsample,
            random_state=random_state,
            verbosity=0,
        )
        self.scaler = StandardScaler()
        self.feature_cols = None
    
    def fit(self, X: pd.DataFrame, y: np.ndarray) -> None:
        """Train on supervised data."""
        self.feature_cols = X.columns.tolist()
        X_scaled = self.scaler.fit_transform(X)
        self.model.fit(X_scaled, y, verbose=False)
        logger.info(f"✓ M2 fitted on {X.shape[0]} samples, {X.shape[1]} features")
    
    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """Make predictions."""
        X_scaled = self.scaler.transform(X)
        return self.model.predict(X_scaled)
    
    def save(self, path: Path) -> None:
        """Save model to disk."""
        with open(path, 'wb') as f:
            pickle.dump({
                'model': self.model,
                'scaler': self.scaler,
                'features': self.feature_cols,
                'hyperparams': self.hyperparams,
            }, f)
        logger.info(f"✓ M2 saved to {path}")
    
    @staticmethod
    def load(path: Path) -> 'M2_XGBoost':
        """Load model from disk."""
        with open(path, 'rb') as f:
            data = pickle.load(f)
        m = M2_XGBoost(**data['hyperparams'])
        m.model = data['model']
        m.scaler = data['scaler']
        m.feature_cols = data['features']
        return m


# ============================================================================
# M3: LSTM NEURAL NETWORK
# ============================================================================

class M3_LSTM:
    """
    M3: LSTM for sequence-based growth prediction.
    
    Input: rolling sequences of portfolio returns + features
    Output: future growth prediction
    """
    
    def __init__(
        self,
        sequence_length: int = 30,
        lstm_units: int = 64,
        dropout_rate: float = 0.2,
        batch_size: int = 32,
        epochs: int = 50,
        random_state: int = RANDOM_STATE,
    ):
        if not LSTM_AVAILABLE:
            raise RuntimeError("TensorFlow/Keras not available")
        
        self.sequence_length = sequence_length
        self.lstm_units = lstm_units
        self.dropout_rate = dropout_rate
        self.batch_size = batch_size
        self.epochs = epochs
        self.random_state = random_state
        
        # Set seeds for reproducibility
        np.random.seed(random_state)
        tf.random.set_seed(random_state)
        
        self.model = None
        self.scaler = MinMaxScaler()
        self.feature_cols = None
        self.history = None
    
    def _build_model(self, n_features: int) -> None:
        """Build LSTM architecture."""
        self.model = Sequential([
            LSTM(self.lstm_units, input_shape=(self.sequence_length, n_features),
                 return_sequences=False, activation='relu'),
            Dropout(self.dropout_rate),
            Dense(32, activation='relu'),
            Dropout(self.dropout_rate),
            Dense(1),  # Output: single continuous value (growth)
        ])
        self.model.compile(optimizer='adam', loss='mse', metrics=['mae'])
    
    def fit(
        self,
        X: pd.DataFrame,
        y: np.ndarray,
        validation_data: Tuple[pd.DataFrame, np.ndarray] = None,
        verbose: int = 0,
    ) -> None:
        """
        Train LSTM.
        
        X should be reshaped to (n_samples, sequence_length, n_features).
        For now, we create sequences from flat features.
        """
        self.feature_cols = X.columns.tolist()
        
        # Scale features
        X_scaled = self.scaler.fit_transform(X)
        
        # Reshape to sequences
        X_seq = self._reshape_to_sequences(X_scaled)
        
        if X_seq is None or len(X_seq) == 0:
            logger.warning("Not enough data to create sequences for LSTM")
            return
        
        # Build model
        self._build_model(X_seq.shape[2])
        
        # Prepare validation data if provided
        val_data = None
        if validation_data is not None:
            X_val, y_val = validation_data
            X_val_scaled = self.scaler.transform(X_val)
            X_val_seq = self._reshape_to_sequences(X_val_scaled)
            if X_val_seq is not None and len(X_val_seq) > 0:
                val_data = (X_val_seq, y_val[:len(X_val_seq)])
        
        # Train
        self.history = self.model.fit(
            X_seq,
            y[:len(X_seq)],
            batch_size=self.batch_size,
            epochs=self.epochs,
            validation_data=val_data,
            verbose=verbose,
        )
        logger.info(f"✓ M3 (LSTM) trained on {len(X_seq)} sequences")
    
    def _reshape_to_sequences(self, X: np.ndarray) -> np.ndarray:
        """
        Reshape flat feature array into sliding sequences.
        
        Input: (n_samples, n_features)
        Output: (n_valid_sequences, sequence_length, n_features)
        """
        n_samples, n_features = X.shape
        
        if n_samples < self.sequence_length:
            logger.warning(f"Not enough samples ({n_samples}) for sequence_length={self.sequence_length}")
            return None
        
        sequences = np.array([
            X[i:i+self.sequence_length]
            for i in range(n_samples - self.sequence_length + 1)
        ])
        
        return sequences
    
    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """Make predictions."""
        if self.model is None:
            raise RuntimeError("Model not fitted")
        
        X_scaled = self.scaler.transform(X)
        X_seq = self._reshape_to_sequences(X_scaled)
        
        if X_seq is None or len(X_seq) == 0:
            logger.warning("Not enough samples to make LSTM predictions")
            return np.array([])
        
        return self.model.predict(X_seq, verbose=0)
    
    def save(self, path: Path) -> None:
        """Save model to disk."""
        if self.model is None:
            logger.warning("No model to save")
            return
        
        # Save Keras model
        model_path = path.with_suffix('.h5')
        self.model.save(model_path)
        
        # Save metadata
        meta_path = path.with_suffix('.pkl')
        with open(meta_path, 'wb') as f:
            pickle.dump({
                'scaler': self.scaler,
                'features': self.feature_cols,
                'sequence_length': self.sequence_length,
            }, f)
        
        logger.info(f"✓ M3 saved to {model_path} and {meta_path}")
    
    @staticmethod
    def load(path: Path) -> 'M3_LSTM':
        """Load model from disk."""
        if not LSTM_AVAILABLE:
            raise RuntimeError("TensorFlow/Keras not available")
        
        model_path = path.with_suffix('.h5')
        meta_path = path.with_suffix('.pkl')
        
        m = M3_LSTM()
        m.model = keras.models.load_model(model_path)
        
        with open(meta_path, 'rb') as f:
            data = pickle.load(f)
        m.scaler = data['scaler']
        m.feature_cols = data['features']
        m.sequence_length = data['sequence_length']
        
        return m


# ============================================================================
# EVALUATION UTILITIES
# ============================================================================

def evaluate_predictions(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    model_name: str = "Model",
) -> Dict[str, float]:
    """
    Compute regression metrics.
    
    Returns
    -------
    metrics : dict
        MAE, RMSE, MAPE, R²
    """
    mae = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    mape = mean_absolute_percentage_error(y_true, y_pred) if np.all(y_true != 0) else np.nan
    
    # Simple R²
    ss_res = np.sum((y_true - y_pred) ** 2)
    ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
    r2 = 1 - (ss_res / ss_tot) if ss_tot > 0 else 0.0
    
    metrics = {
        'mae': mae,
        'rmse': rmse,
        'mape': mape,
        'r2': r2,
    }
    
    logger.info(f"{model_name} Performance:")
    for key, val in metrics.items():
        logger.info(f"  {key.upper():6} = {val:.6f}")
    
    return metrics


# ============================================================================
# RELIABILITY SCORING
# ============================================================================

def compute_model_reliability(mae: float, rmse: float) -> float:
    """
    Derive reliability score from prediction error.
    
    Parameters
    ----------
    mae : float
        Mean absolute error
    rmse : float
        Root mean squared error
    
    Returns
    -------
    reliability : float in [0, 1]
        Normalized reliability score (inverse of error).
        Higher score = more reliable.
    """
    # Avoid division by zero
    avg_error = (mae + rmse) / 2
    if avg_error == 0:
        return 1.0
    
    # Sigmoid-like mapping: lower error -> higher reliability
    # reliability = 1 / (1 + avg_error)
    reliability = 1.0 / (1.0 + avg_error)
    
    return np.clip(reliability, 0, 1)
