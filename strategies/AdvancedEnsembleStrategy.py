import numpy as np
import pandas as pd
import warnings
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.calibration import CalibratedClassifierCV
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from scipy.stats import norm


class AdvancedTradingStrategy:
    """
    An advanced ensemble trading strategy with:
    - Walk-forward validation
    - Realistic slippage modeling
    - Minimum trade duration
    - Equity-based position sizing
    - Comprehensive risk metrics
    """

    def __init__(self, data, ticker, initial_capital=100000, **params):
        """
        Initialize strategy with data and parameters.
        
        Parameters:
        -----------
        data : pd.DataFrame
            Input price data with at least 'Close' and 'Volume'
        ticker : str
            Ticker symbol for the asset
        initial_capital : float
            Starting capital for backtesting
        params : dict
            Strategy parameters including:
            - short_window: Short moving average window (default 10)
            - medium_window: Medium MA window (default 50)
            - long_window: Long MA window (default 200)
            - volatility_window: Volatility lookback (default 20)
            - rsi_window: RSI period (default 14)
            - min_hold_days: Minimum holding period (default 3)
            - max_position_size: Maximum allocation (default 0.25)
            - slippage_pct: Base slippage percentage (default 0.0005)
        """
        self.data = self._prepare_data(data, ticker)
        self._validate_init_data()
        self.initial_capital = initial_capital

        # Set parameters with defaults
        self.short_window = params.get('short_window', 10)
        self.medium_window = params.get('medium_window', 50) 
        self.long_window = params.get('long_window', 200)
        self.volatility_window = params.get('volatility_window', 20)
        self.rsi_window = params.get('rsi_window', 14)
        self.min_hold_days = params.get('min_hold_days', 3)
        self.max_position_size = params.get('max_position_size', 0.25)
        self.slippage_pct = params.get('slippage_pct', 0.0005)

        # Initialize models and scaler
        self.models = self._initialize_models()
        self.scaler = StandardScaler()
        self.feature_importances = {}
        self.ensemble_weights = {}

        # Prepare data
        self._initialize_returns()
        self._initialize_volatility()

    def _prepare_data(self, data, ticker):
        """Prepare and validate input data structure."""
        if isinstance(data.columns, pd.MultiIndex):
            required_columns = [("Close", ticker), ("Volume", ticker)]
            df = data[required_columns].copy()
            df.columns = ["Close", "Volume"]
        else:
            required_columns = ["Close", "Volume"]
            df = data[required_columns].copy()

        # Ensure numeric data
        df = df.apply(pd.to_numeric, errors="coerce")
        return df.dropna()

    def _validate_init_data(self):
        """Validate we have required data columns."""
        required = ["Close", "Volume"]
        if not all(col in self.data.columns for col in required):
            raise ValueError(f"Missing required columns. Needed: {required}")

    def _initialize_returns(self):
        """Initialize returns if not present."""
        if "Returns" not in self.data.columns:
            # self.data["Returns"] = self.data["Close"].pct_change()
            self.data["Returns"] = self.data["Close"].pct_change(fill_method=None)
            self.data.dropna(subset=["Returns"], inplace=True)

    def _initialize_volatility(self):
        """Initialize volatility-related columns."""
        if "Log_Returns" not in self.data.columns:
            self.data["Log_Returns"] = np.log(
                self.data["Close"] / self.data["Close"].shift(1)
            )
            self.data["Volatility"] = self.data["Log_Returns"].rolling(
                self.volatility_window).std() * np.sqrt(252)
            self.data.dropna(subset=["Volatility"], inplace=True)

    def _initialize_models(self):
        """Initialize ensemble models with probability calibration."""
        return {
            "random_forest": Pipeline([
                ("scaler", StandardScaler()),
                ("model", RandomForestClassifier(n_estimators=100, random_state=42))
            ]),
            "gradient_boosting": Pipeline([
                ("scaler", StandardScaler()),
                ("model", GradientBoostingClassifier(n_estimators=100, random_state=42))
            ]),
            "svm": Pipeline([
                ("scaler", StandardScaler()),
                ("model", CalibratedClassifierCV(SVC(kernel="rbf", probability=True)))
            ])
        }

    def create_features(self):
        """Create complete feature set with validation."""
        required_columns = ["Returns", "Log_Returns", "Volatility"]
        if not all(col in self.data.columns for col in required_columns):
            raise ValueError("Missing required columns for feature creation")

        # Moving Averages
        for window, name in [
            (self.short_window, "SMA10"),
            (self.medium_window, "SMA50"),
            (self.long_window, "SMA200"),
        ]:
            self.data[name] = self.data["Close"].rolling(window).mean()

        # MA Cross
        self.data["MA_Cross"] = (self.data["SMA10"] > self.data["SMA50"]).astype(int)

        # Bollinger Bands
        self.data["Upper_Band"] = self.data["SMA50"] + 2 * self.data["Volatility"]
        self.data["Lower_Band"] = self.data["SMA50"] - 2 * self.data["Volatility"]

        # Momentum Indicators
        self.data["RSI"] = self._compute_rsi(self.data["Close"], self.rsi_window)
        self.data["MACD"] = self.data["SMA10"] - self.data["SMA50"]

        # Volume Features
        self.data["Volume_MA"] = self.data["Volume"].rolling(10).mean()
        self.data["Volume_Spike"] = self.data["Volume"] / self.data["Volume_MA"] - 1
        self.data["Volume_ZScore"] = (
            self.data["Volume"] - self.data["Volume"].rolling(20).mean()
        ) / (self.data["Volume"].rolling(20).std() + 1e-9)
        self.data["Volatility_Spike"] = (
            self.data["Volatility"] / self.data["Volatility"].rolling(50).mean() - 1
        )

        # Target Variable
        self.data["Target"] = (
            self.data["Returns"].shift(-1) > self.data["Returns"].std() * 0.5
        ).astype(int)

    @staticmethod
    def _compute_rsi(series, period):
        """Calculate Relative Strength Index."""
        delta = series.diff()
        gain = delta.where(delta > 0, 0)
        loss = -delta.where(delta < 0, 0)

        avg_gain = gain.rolling(period).mean()
        avg_loss = loss.rolling(period).mean()

        rs = avg_gain / (avg_loss + 1e-9)  # Avoid division by zero
        return 100 - (100 / (1 + rs))

    def train_ensemble(self, X, y, n_splits=3):
        """Walk-forward training with expanding window."""
        # Add class_weight parameter
        self.models["random_forest"].named_steps["model"].set_params(
            class_weight="balanced",
            max_features="sqrt",  # Better for high-dimensional data
            min_samples_leaf=5,  # Prevent overfitting
        )

        # Add early stopping for GBM
        self.models["gradient_boosting"].named_steps["model"].set_params(
            n_iter_no_change=10,
            validation_fraction=0.2,
            tol=1e-4,  # Stricter tolerance
            max_depth=3,  # Prevent overfitting
        )
        # Add feature names for importance tracking
        features = X.columns.tolist()
        metrics = []
        oof_predictions = {}  # Out-of-fold predictions for meta-model  

        # Time-based splits
        split_points = np.linspace(0, len(X), n_splits + 1, dtype=int)[1:-1]

        metrics = []
        for i, split in enumerate(split_points):
            train_idx = slice(None, split)
            test_idx = slice(
                split, min(split + int(len(X) * 0.2), len(X))
            )  # Ensure test index is within bounds

            X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
            y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]

            if len(X_train) < 100 or len(X_test) < 20:  # Minimum data requirements
                warnings.warn(f"Skipping split {i+1} due to insufficient data.")
                continue

            # Track predictions for stacking/ensemble weighting
            fold_predictions = {}

            # Train models
            for name, model in self.models.items():
                # Train with early stopping callback if available
                if hasattr(model.named_steps["model"], "fit"):
                    model.fit(X_train, y_train)
                # Get probabilistic predictions
                if hasattr(model.named_steps["model"], "predict_proba"):
                    pred = model.predict_proba(X_test)[:, 1]
                else:
                    pred = model.predict(X_test)

                fold_predictions[name] = pred                    
                # Calculate comprehensive metrics
                acc = accuracy_score(y_test, (pred > 0.5).astype(int))
                precision = precision_score(y_test, (pred > 0.5).astype(int))
                recall = recall_score(y_test, (pred > 0.5).astype(int))
                f1 = f1_score(y_test, (pred > 0.5).astype(int))

                metrics.append(
                    {
                        "split": i + 1,
                        "model": name,
                        "accuracy": acc,
                        "precision": precision,
                        "recall": recall,
                        "f1_score": f1,
                        "train_size": len(X_train),
                        "test_size": len(X_test),
                        "train_start": X_train.index[0],
                        "train_end": X_train.index[-1],
                        "test_start": X_test.index[0],
                        "test_end": X_test.index[-1],
                    }
                )

                # Store predictions for potential meta-model training
                oof_predictions[f"fold_{i}"] = fold_predictions

        # Final training on full dataset with feature importance
        best_model = None
        best_score = -np.inf

        # Retrain on full data (after dropping NaNs)
        for name, model in self.models.items():
            model.fit(X, y)

            # Score on full data to select best model
            if hasattr(model.named_steps["model"], "score"):
                score = model.score(X, y)
                if score > best_score:
                    best_score = score
                    best_model = model

        # Store best model# Calculate feature importances from best model
        if best_model is not None and hasattr(
            best_model.named_steps["model"], "feature_importances_"
        ):
            importances = best_model.named_steps["model"].feature_importances_
            self.feature_importances = dict(zip(features, importances))

            # Calculate ensemble weights based on cross-validation performance
            metrics_df = pd.DataFrame(metrics)
            model_scores = metrics_df.groupby("model")["f1_score"].mean()
            self.ensemble_weights = (model_scores / model_scores.sum()).to_dict()

        # Get feature importances
        # best_model = self.models["random_forest"]
        # importances = best_model.named_steps["model"].feature_importances_
        # self.feature_importances = dict(zip(features, importances))

        return pd.DataFrame(metrics)

    def generate_signals(self, X):
        """Generate trading signals with minimum holding period."""
        # Initialize signals
        self.data["Signal"] = 0

        # Get predictions from all models
        predictions = {
            name: model.predict_proba(X)[:, 1] 
            for name, model in self.models.items()
        }

        # Weighted average prediction
        weights = {"random_forest": 0.5, "gradient_boosting": 0.3, "svm": 0.2}
        weighted_probs = sum(predictions[name] * weight for name, weight in weights.items())

        # Generate signals with confidence threshold
        confidence_threshold = 0.15
        self.data.loc[:, "Model_Confidence"] = weighted_probs

        # Vectorized signal generation with np.select
        conditions = [
            weighted_probs > (0.5 + confidence_threshold),
            weighted_probs < (0.5 - confidence_threshold)
        ]
        choices = [1, -1]
        self.data.loc[:, "Signal"] = np.select(conditions, choices, default=0)

        # Apply minimum holding period with proper indexing
        signals = self.data["Signal"].copy()

        # Apply minimum holding period
        last_trade_day = None
        for i in range(len(self.data)):
            if signals.iat[i] != 0:
                if last_trade_day is None or (i - last_trade_day) >= self.min_hold_days:
                    last_trade_day = i
                else:
                    signals.iat[i] = 0

        self.data.loc[:, "Signal"] = signals

        # Vectorized volatility-normalized position sizing
        vol_ma = self.data["Volatility"].rolling(20, min_periods=5).mean()
        norm_vol = self.data["Volatility"] / (vol_ma + 1e-9)

        base_size = 0.05  # Base 10% allocation
        confidence_multiplier = np.abs(self.data["Model_Confidence"] - 0.5) * 2

        raw_size = np.where(
            self.data["Signal"] != 0,
            base_size / (norm_vol.clip(0.8, 1.2) + 0.01) * confidence_multiplier,
            0
        )

        self.data.loc[:, "PositionSize"] = np.clip(
            raw_size, -self.max_position_size, self.max_position_size
        )

        return self.data[["Signal", "PositionSize", "Model_Confidence"]]

    def calculate_returns(self):
        """Dynamic position sizing with slippage modeling using proper pandas operations."""
        # Initialize columns with .loc
        self.data.loc[:, "PortfolioValue"] = float(self.initial_capital)
        self.data.loc[:, "SharesHeld"] = 0.0
        self.data.loc[:, "TradeCost"] = 0.0

        # Pre-calculate rolling average volume
        avg_volumes = self.data["Volume"].rolling(20, min_periods=1).mean()

        for i in range(1, len(self.data)):
            prev_val = self.data["PortfolioValue"].iat[i - 1]
            current_price = self.data["Close"].iat[i]

            # Calculate target position using .iat for scalar access
            dollar_position = prev_val * self.data["PositionSize"].iat[i]
            target_shares = dollar_position / current_price
            shares_traded = target_shares - self.data["SharesHeld"].iat[i - 1]

            # Dynamic slippage calculation
            liquidity_factor = np.sqrt(1e6 / (avg_volumes.iat[i] + 1e-9)).clip(0.8, 1.2)
            dynamic_slippage = self.slippage_pct * liquidity_factor

            # Calculate trade cost
            trade_value = abs(shares_traded) * current_price
            trade_cost = trade_value * dynamic_slippage + (abs(shares_traded) > 0) * 0.5

            # Update portfolio with .iat
            self.data.loc[self.data.index[i], "SharesHeld"] = target_shares
            self.data.loc[self.data.index[i], "TradeCost"] = trade_cost

            # Calculate new portfolio value
            cash_flow = -shares_traded * current_price - trade_cost
            market_movement = (
                target_shares
                * self.data["Close"].iat[i]
                * (self.data["Close"].pct_change(fill_method=None).iat[i])
            )

            self.data.loc[self.data.index[i], "PortfolioValue"] = (
                prev_val + cash_flow + market_movement
            )

        # Calculate returns without fill_method warning
        self.data.loc[:, "StrategyReturns"] = self.data["PortfolioValue"].pct_change(
            fill_method=None
        )
        self.data.loc[:, "CumulativeReturns"] = (
            1 + self.data["StrategyReturns"].fillna(0)
        ).cumprod()

        return self.data

    def _calculate_drawdown_stats(self, cum_returns):
        """Calculate comprehensive drawdown statistics with robust error handling.

        Args:
            cum_returns (pd.Series): Series of cumulative returns

        Returns:
            dict: Dictionary containing:
                - max_drawdown: Maximum drawdown percentage
                - avg_drawdown: Average drawdown percentage
                - drawdown_duration: Average duration of drawdowns in periods
                - max_drawdown_duration: Longest drawdown duration
                - ulcer_index: Ulcer Performance Index
                - pain_index: Average percentage drawdown
                - recovery_duration: Time to recover from max drawdown
        """
        try:
            # Initialize default return values
            stats = {
                "max_drawdown": 0.0,
                "avg_drawdown": 0.0,
                "drawdown_duration": 0,
                "max_drawdown_duration": 0,
                "ulcer_index": 0.0,
                "pain_index": 0.0,
                "recovery_duration": 0,
            }

            if len(cum_returns) == 0:
                return stats

            # Calculate peak and drawdown
            peak = cum_returns.cummax()
            drawdown = (peak - cum_returns) / (
                peak + 1e-9
            )  # Add small constant to avoid division by zero
            drawdown_pct = drawdown * 100

            # Basic stats
            stats["max_drawdown"] = drawdown_pct.max()
            stats["avg_drawdown"] = drawdown_pct.mean()
            stats["ulcer_index"] = np.sqrt((drawdown_pct**2).mean())
            stats["pain_index"] = drawdown_pct.mean()

            # Calculate drawdown periods with proper boolean handling
            in_drawdown = (drawdown > 0).astype(bool)

            # Explicitly convert to boolean dtype before shift operations
            shifted = in_drawdown.shift(1).fillna(False)
            if shifted.dtype == object:
                shifted = shifted.infer_objects(copy=False)

            drawdown_start = in_drawdown & ~shifted
            drawdown_end = ~in_drawdown & shifted

            # Track durations
            durations = []
            current_start = None

            for i in range(len(drawdown)):
                if drawdown_start.iloc[i]:
                    current_start = i
                elif drawdown_end.iloc[i] and current_start is not None:
                    durations.append(i - current_start)
                    current_start = None

            # If we're still in a drawdown at the end
            if current_start is not None:
                durations.append(len(drawdown) - 1 - current_start)

            # Calculate duration stats
            if durations:
                stats["drawdown_duration"] = np.mean(durations)
                stats["max_drawdown_duration"] = max(durations)

            # Calculate recovery duration from max drawdown
            if stats["max_drawdown"] > 0:
                trough_idx = drawdown_pct.idxmax()
                recovery_mask = cum_returns[trough_idx:] >= peak[trough_idx]
                if recovery_mask.any():
                    recovery_idx = recovery_mask.idxmax()
                    stats["recovery_duration"] = recovery_idx - trough_idx
                else:
                    stats["recovery_duration"] = len(cum_returns) - trough_idx - 1

            return stats

        except Exception as e:
            warnings.warn(f"Error calculating drawdown stats: {str(e)}")
            return {
                "max_drawdown": 0.0,
                "avg_drawdown": 0.0,
                "drawdown_duration": 0,
                "max_drawdown_duration": 0,
                "ulcer_index": 0.0,
                "pain_index": 0.0,
                "recovery_duration": 0,
            }

    def calculate_common_sense_ratio(self, returns):
        """
        Calculate Common Sense Ratio (CSR)
        CSR = (Gross Winning Trades / Gross Losing Trades)

        A ratio > 1 means strategy makes more on winners than loses on losers
        """
        winning_trades = returns[returns > 0]
        losing_trades = returns[returns < 0]

        if len(losing_trades) == 0:
            return np.nan  # Avoid division by zero if no losing trades

        gross_profits = winning_trades.sum()
        gross_losses = abs(losing_trades.sum())

        return gross_profits / gross_losses

    def calculate_advanced_metrics(self):
        """Calculate comprehensive performance metrics."""
        returns = self.data["StrategyReturns"]
        cum_returns = self.data["CumulativeReturns"]

        # Initialize metrics with default values
        metrics = {
            # Return metrics
            "cumulative_return": 0.0,
            "annualized_return": 0.0,
            "annualized_volatility": 0.0,
            # Risk metrics
            "sharpe_ratio": 0.0,
            "sortino_ratio": 0.0,
            "max_drawdown": 0.0,
            "drawdown_duration": 0,
            "recovery_duration": 0,
            "calmar_ratio": 0.0,
            "var_95": 0.0,  # 95% Value at Risk
            "es_95": 0.0,  # 95% Expected Shortfall (CVaR)
            "var_99": 0.0,  # 99% Value at Risk
            "es_99": 0.0,  # 99% Expected Shortfall
            "cvar_95": 0.0,  # Conditional VaR at 95%
            "cvar_99": 0.0,  # Conditional VaR at 99%
            "skewness": 0.0,
            "kurtosis": 0.0,
            "omega_ratio": 0.0,
            "tail_ratio": 1.0,
            # Trade statistics
            "winning_trades": 0,
            "losing_trades": 0,
            "total_trades": 0,
            "win_rate": 0.0,
            "profit_factor": 0.0,
            "common_sense_ratio": 0.0,
            "avg_winning_trade": 0.0,
            "avg_losing_trade": 0.0,
            "best_trade": 0.0,
            "worst_trade": 0.0,
            # Position metrics
            "max_leverage": 0.0,
            "avg_leverage": 0.0,
            "long_short_ratio": 0.0,
            "long_short_value_ratio": 0.0,
            # Model metrics
            "feature_importances": getattr(self, "feature_importances", {}),
            "ensemble_weights": getattr(self, "ensemble_weights", {}),
            "num_models": len(getattr(self, "ensemble_weights", {})),
        }

        positive_returns = returns[returns > 0]
        negative_returns = returns[returns < 0]

        # Calculate trade statistics
        winning_trades = len(positive_returns)
        losing_trades = len(negative_returns)
        total_trades = winning_trades + losing_trades

        # Calculate long/short metrics if Position data exists
        long_count = short_count = long_value = short_value = 0
        if "Position" in self.data.columns:
            long_positions = self.data[self.data["Position"] > 0]
            short_positions = self.data[self.data["Position"] < 0]
            long_count = len(long_positions)
            short_count = len(short_positions)
            long_value = long_positions["Position"].sum() if long_count > 0 else 0
            short_value = abs(short_positions["Position"].sum()) if short_count > 0 else 0

        # Return metrics
        metrics["cumulative_return"] = cum_returns.iloc[-1] - 1 if len(cum_returns) > 0 else 0

        if len(returns) > 252:
            metrics["annualized_return"] = (cum_returns.iloc[-1] ** (252/len(returns))) - 1
            metrics["annualized_volatility"] = returns.std() * np.sqrt(252)
        else:
            metrics["annualized_return"] = cum_returns.iloc[-1] - 1
            metrics["annualized_volatility"] = returns.std() * np.sqrt(len(returns))

        # Risk metrics
        metrics["sharpe_ratio"] = (
            returns.mean() / (returns.std() + 1e-9) * np.sqrt(252)
        )
        metrics["sortino_ratio"] = self._calculate_sortino(returns)
        metrics.update(self._calculate_drawdown_stats(cum_returns))

        max_dd = abs(metrics["max_drawdown"])
        metrics["calmar_ratio"] = (
            metrics["annualized_return"] / max_dd if max_dd > 0 else 0
        )

        # Calculate VaR and other risk metrics
        if len(returns) >= 100:  # Minimum data points for meaningful calculations
            try:
                # Historical VaR
                metrics["var_95"] = np.percentile(returns, 5) * 100  # As percentage
                metrics["es_95"] = returns[returns <= np.percentile(returns, 5)].mean() * 100
                metrics["var_99"] = np.percentile(returns, 1) * 100
                metrics["es_99"] = (
                    returns[returns <= np.percentile(returns, 1)].mean() * 100
                )

                # Conditional VaR (Expected Shortfall)
                metrics["cvar_95"] = (
                    returns[returns <= np.percentile(returns, 5)].mean() * 100
                )
                metrics["cvar_99"] = (
                    returns[returns <= np.percentile(returns, 1)].mean() * 100
                )

                # Distribution characteristics
                metrics["skewness"] = returns.skew()
                metrics["kurtosis"] = returns.kurtosis()

                # Parametric VaR (assuming normal distribution)
                mu, sigma = returns.mean(), returns.std()
                metrics["parametric_var_95"] = (mu + sigma * norm.ppf(0.05)) * 100
                metrics["parametric_var_99"] = (mu + sigma * norm.ppf(0.01)) * 100

            except Exception as e:
                print(f"Error calculating risk metrics: {str(e)}")

        # Additional metrics
        metrics.update(
            {
                "omega_ratio": self._calculate_omega_ratio(returns),
                "tail_ratio": self._calculate_tail_ratio(returns),
                "win_rate": self._calculate_win_rate(returns),
                "winning_trades": winning_trades,
                "losing_trades": losing_trades,
                "total_trades": total_trades,
                "profit_factor": self._calculate_profit_factor(returns),
                "common_sense_ratio": self.calculate_common_sense_ratio(returns),
                "avg_winning_trade": (
                    positive_returns.mean() if len(positive_returns) > 0 else 0
                ),
                "avg_losing_trade": (
                    negative_returns.mean() if len(negative_returns) > 0 else 0
                ),
                "best_trade": returns.max() if len(returns) > 0 else 0,
                "worst_trade": returns.min() if len(returns) > 0 else 0,
                "long_short_ratio": long_count
                / (short_count + 1e-9),  # Count-based ratio
                "long_short_value_ratio": long_value
                / (short_value + 1e-9),  # Value-based ratio
                "max_leverage": self.data["PositionSize"].abs().max(),
                "avg_leverage": self.data["PositionSize"].abs().mean(),
                "total_trades": (self.data["Signal"].diff().abs() > 0).sum(),
                "feature_importances": self.feature_importances,
                "ensemble_weights": getattr(self, "ensemble_weights", {}),
            }
        )

        return metrics

    @staticmethod
    def _calculate_sortino(returns, risk_free=0):
        """Calculate Sortino ratio."""
        downside_returns = returns[returns < risk_free]
        downside_std = downside_returns.std()
        return (
            (returns.mean() - risk_free) / (downside_std + 1e-9) * np.sqrt(252)
            if len(downside_returns) > 1 else 0
        )

    @staticmethod
    def _calculate_omega_ratio(returns, threshold=0):
        """Calculate Omega ratio."""
        excess = returns - threshold
        return (
            excess[excess > 0].sum() / -excess[excess < 0].sum()
            if excess[excess < 0].sum() < 0 else float('inf')
        )

    @staticmethod
    def _calculate_tail_ratio(returns):
        """Calculate Tail ratio (95th vs 5th percentile)."""
        if len(returns) < 10:
            return 0
        right_tail = np.percentile(returns, 95)
        left_tail = abs(np.percentile(returns, 5))
        return right_tail / (left_tail + 1e-9)

    @staticmethod
    def _calculate_win_rate(returns):
        """Calculate percentage of winning trades."""
        winning = returns[returns > 0]
        total = returns[returns != 0]
        return len(winning) / len(total) if len(total) > 0 else 0

    @staticmethod
    def _calculate_profit_factor(returns):
        """Calculate profit factor (gross profit/gross loss)."""
        gross_profit = returns[returns > 0].sum()
        gross_loss = abs(returns[returns < 0].sum())
        return gross_profit / (gross_loss + 1e-9)

    def execute(self):
        """Execute full strategy pipeline with error handling."""
        try:
            # Data preparation
            self._initialize_returns()
            self._initialize_volatility()

            # Feature engineering
            self.create_features()
            self.data.dropna(inplace=True)  # Drop rows with NaN values
            if len(self.data) == 0:
                raise ValueError("No data remaining after dropping NaN values")

            # Model training
            X = self.data[
                [
                    "SMA10",
                    "SMA50",
                    "SMA200",
                    "Volatility",
                    "RSI",
                    "MACD",
                    "Volume_Spike",
                    "Log_Returns",
                ]
            ]
            y = self.data["Target"]  # Ensure you have the target variable
            wf_results = self.train_ensemble(X=X, y=y)  # Pass X and y to train_ensemble

            # Signal generation
            self.generate_signals(X)

            # Performance calculation
            self.calculate_returns()
            metrics = self.calculate_advanced_metrics()

            # Prepare results
            results = self.data[
                [
                    "Close",
                    "Signal",
                    "PositionSize",
                    "StrategyReturns",
                    "CumulativeReturns",
                    "Model_Confidence",
                    "Volatility",
                ]
            ]

            return results, metrics, wf_results

        except Exception as e:
            warnings.warn(f"Strategy execution failed: {str(e)}")
            return pd.DataFrame(), {}, pd.DataFrame()
