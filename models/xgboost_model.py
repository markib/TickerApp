import xgboost as xgb
from sklearn.metrics import mean_squared_error

class XGBoostModel:
    def __init__(self, params):
        self.model = xgb.XGBRegressor(**params)

    def train(self, X_train, y_train):
        self.model.fit(X_train, y_train)

    def predict(self, X_test):
        return self.model.predict(X_test)

    def evaluate(self, X_test, y_test):
        predictions = self.predict(X_test)
        return mean_squared_error(y_test, predictions)
