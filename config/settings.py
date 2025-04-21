# Configuration settings for stock analysis app

# API Settings
YFINANCE_API_URL = "https://query1.finance.yahoo.com/v7/finance/download/{ticker}"

# Machine Learning model settings
MODEL_CONFIG = {
    "linear_regression": {"learning_rate": 0.01, "epochs": 100},
    "xgboost": {"n_estimators": 100, "max_depth": 3},
    "lstm": {"epochs": 50, "batch_size": 32}
}
