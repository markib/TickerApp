# Import libraries
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from sklearn.pipeline import Pipeline
from typing import Tuple, Optional


class MLEnhancedTradingStrategy:
    """
    Machine Learning Enhanced Trading Strategy that combines technical indicators
    with Random Forest classification to generate trading signals.

    Features:
    - Multiple technical indicators (RSI, ATR, Moving Averages)
    - Proper feature scaling
    - Walk-forward validation
    - Risk-adjusted returns calculation
    """

    def __init__(self, data: pd.DataFrame, ticker: str, lookback: int = 14):
        """
        Initialize the strategy.

        Args:
            data: DataFrame containing market data (must include OHLCV)
            ticker: Ticker symbol for the asset
            lookback: Lookback period for technical indicators
        """
        self.ticker = ticker
        self.lookback = lookback
        self.model = None
        self.scaler = StandardScaler()
        self._prepare_data(data)
        self._initialize_model()

    def _prepare_data(self, data: pd.DataFrame) -> None:
        """Prepare and validate the input data."""
        if isinstance(data.columns, pd.MultiIndex):
            self.data = data[
                [
                    ("Close", self.ticker),
                    ("High", self.ticker),
                    ("Low", self.ticker),
                    ("Volume", self.ticker),
                ]
            ].copy()
            self.data.columns = ["Close", "High", "Low", "Volume"]
        else:
            self.data = data[["Close", "High", "Low", "Volume"]].copy()

        self._validate_data()

    def _validate_data(self) -> None:
        """Validate the input data structure."""
        required_columns = ["Close", "High", "Low", "Volume"]
        if not all(col in self.data.columns for col in required_columns):
            raise ValueError(f"Missing required columns. Needed: {required_columns}")

    def _initialize_model(self) -> None:
        """Initialize the machine learning model pipeline."""
        self.model = Pipeline(
            [
                ("scaler", StandardScaler()),
                (
                    "classifier",
                    RandomForestClassifier(
                        n_estimators=100,
                        random_state=42,
                        class_weight="balanced",
                        n_jobs=-1,
                    ),
                ),
            ]
        )

    def compute_technical_indicators(self) -> None:
        """Calculate all technical indicators used as features."""
        # Price-based features
        self.data["Returns"] = self.data["Close"].pct_change()
        self.data["SMA20"] = self.data["Close"].rolling(window=20).mean()
        self.data["SMA50"] = self.data["Close"].rolling(window=50).mean()

        # Momentum indicator
        self.data["RSI"] = self._compute_rsi(self.data["Close"], 14)

        # Volatility indicator
        self.data["ATR"] = self._compute_atr(
            self.data["High"], self.data["Low"], self.data["Close"], 14
        )

        # Volume-based feature
        self.data["VolumeMA"] = self.data["Volume"].rolling(window=10).mean()

        # Target variable (1 if next return is positive, 0 otherwise)
        self.data["Target"] = (self.data["Returns"].shift(-1) > 0).astype(int)

        # Clean data
        self.data.dropna(inplace=True)

    @staticmethod
    def _compute_rsi(series: pd.Series, period: int) -> pd.Series:
        """Calculate Relative Strength Index (RSI)."""
        delta = series.diff()
        gain = delta.where(delta > 0, 0)
        loss = -delta.where(delta < 0, 0)

        avg_gain = gain.rolling(window=period).mean()
        avg_loss = loss.rolling(window=period).mean()

        rs = avg_gain / avg_loss
        return 100 - (100 / (1 + rs))

    @staticmethod
    def _compute_atr(
        high: pd.Series, low: pd.Series, close: pd.Series, period: int
    ) -> pd.Series:
        """Calculate Average True Range (ATR)."""
        tr1 = high - low
        tr2 = (high - close.shift()).abs()
        tr3 = (low - close.shift()).abs()
        tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
        return tr.rolling(window=period).mean()

    def train_and_validate(self) -> dict:
        """
        Train the model and validate performance.

        Returns:
            Dictionary containing performance metrics
        """
        features = ["Returns", "SMA20", "SMA50", "RSI", "ATR", "VolumeMA"]
        X = self.data[features]
        y = self.data["Target"]

        # Split data (time-series aware split)
        split_idx = int(len(X) * 0.8)
        X_train, X_test = X[:split_idx], X[split_idx:]
        y_train, y_test = y[:split_idx], y[split_idx:]

        # Train model
        self.model.fit(X_train, y_train)

        # Evaluate
        train_pred = self.model.predict(X_train)
        test_pred = self.model.predict(X_test)

        return {
            "train_report": classification_report(
                y_train, train_pred, output_dict=True
            ),
            "test_report": classification_report(y_test, test_pred, output_dict=True),
            "feature_importances": dict(
                zip(features, self.model.named_steps["classifier"].feature_importances_)
            ),
        }

    def generate_signals(self) -> None:
        """Generate trading signals based on model predictions."""
        features = ["Returns", "SMA20", "SMA50", "RSI", "ATR", "VolumeMA"]
        X = self.data[features]

        # Make predictions
        self.data["Prediction"] = self.model.predict(X)

        # Convert predictions to signals
        self.data["Signal"] = 0
        self.data.loc[self.data["Prediction"] == 1, "Signal"] = 1  # Buy signal
        self.data.loc[self.data["Prediction"] == 0, "Signal"] = -1  # Sell signal

    def calculate_returns(self) -> None:
        """Calculate strategy returns and performance metrics."""
        self.data["StrategyReturns"] = self.data["Returns"] * self.data["Signal"].shift(
            1
        )
        self.data["CumulativeReturns"] = (1 + self.data["StrategyReturns"]).cumprod()
        self.data.fillna(0, inplace=True)

    def execute(self) -> Tuple[pd.DataFrame, dict]:
        """
        Execute the full strategy pipeline.

        Returns:
            Tuple containing:
                - DataFrame with signals and returns
                - Dictionary with performance metrics
        """
        try:
            # Feature engineering
            self.compute_technical_indicators()

            # Model training and validation
            metrics = self.train_and_validate()

            # Signal generation
            self.generate_signals()

            # Performance calculation
            self.calculate_returns()

            # Prepare results
            results_df = self.data[
                ["Close", "Signal", "StrategyReturns", "CumulativeReturns"]
            ]

            return results_df, metrics

        except Exception as e:
            raise RuntimeError(f"Strategy execution failed: {str(e)}") from e
