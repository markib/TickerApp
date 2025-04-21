import tensorflow as tf
from sklearn.metrics import mean_squared_error

class LSTMModel:
    def __init__(self, epochs=50, batch_size=32):
        self.epochs = epochs
        self.batch_size = batch_size
        self.model = None

    def build_model(self, input_shape):
        model = tf.keras.Sequential()
        model.add(tf.keras.layers.LSTM(50, return_sequences=True, input_shape=input_shape))
        model.add(tf.keras.layers.LSTM(50))
        model.add(tf.keras.layers.Dense(1))
        model.compile(optimizer='adam', loss='mean_squared_error')
        self.model = model

    def train(self, X_train, y_train):
        self.model.fit(X_train, y_train, epochs=self.epochs, batch_size=self.batch_size)

    def predict(self, X_test):
        return self.model.predict(X_test)

    def evaluate(self, X_test, y_test):
        predictions = self.predict(X_test)
        return mean_squared_error(y_test, predictions)
