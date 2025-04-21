import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler

def preprocess_data(df):
    """ Preprocess stock data by scaling features """
    df = df[['Open', 'High', 'Low', 'Close', 'Volume']]
    scaler = StandardScaler()
    scaled_data = scaler.fit_transform(df)
    df_scaled = pd.DataFrame(scaled_data, columns=df.columns)
    return df_scaled
