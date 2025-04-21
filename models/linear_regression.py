import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from datetime import timedelta

class LinearRegressionModel:
    def __init__(self, data: pd.DataFrame):
        self.data = data
        self.model = LinearRegression()

    def preprocess_data(self):
        # Using "Close" prices for prediction
        self.data['Date'] = pd.to_datetime(self.data.index)
        self.data['Days'] = (self.data['Date'] - self.data['Date'].min()).dt.days
        X = self.data[['Days']]
        y = self.data['Close']
        return X, y

    def train_model(self, X, y):
        self.model.fit(X, y)

    def predict_next_days(self,n_days=5):
        X, y = self.preprocess_data()
        self.train_model(X, y)

        # Predicting next 5 days
        future_days = np.array([[i] for i in range(self.data['Days'].max() + 1, self.data['Days'].max() + 6)])
        preds = self.model.predict(future_days)

        # Generate future dates starting after the last date in the data
        last_date = self.data.index[-1]
        future_dates = [last_date + timedelta(days=i) for i in range(1, n_days + 1)]

        # Convert to DataFrame
        predictions = pd.DataFrame({
            'Date': future_dates,
            'Predicted_Close': preds.flatten()  # Flatten (n, 1) to (n,)
        })
        
        return predictions
    
    def evaluate(self, X_test, y_test):
        predictions = self.predict(X_test)
        return mean_squared_error(y_test, predictions)
