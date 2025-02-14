from fastapi import FastAPI
import yfinance as yf
import pandas as pd
import numpy as np
import tensorflow as tf
from sklearn.preprocessing import MinMaxScaler
from pydantic import BaseModel
from keras.metrics import MeanAbsoluteError
from keras.models import load_model

app = FastAPI()

WINDOW_SIZE = 20
FEATURES = 5

class StockRequest(BaseModel):
    ticker: str
    start_date: str
    end_date: str

def dataframe_prep(stock_data: pd.DataFrame) -> pd.DataFrame:

    scaler = MinMaxScaler(feature_range=(0, 1))
    df = stock_data.copy()
    close_df = df.xs('Close', level=0, axis=1)
    high_df = df.xs('High', level=0, axis=1)
    low_df = df.xs('Low', level=0, axis=1)
    open_df = df.xs('Open', level=0, axis=1)
    volume_df = df.xs('Volume', level=0, axis=1)

    dates = close_df.index
    close = close_df.values
    high = high_df.values
    low = low_df.values
    open = open_df.values
    volume = volume_df.values

    dates = np.array(dates).flatten()
    open = np.array(open).flatten()
    high = np.array(high).flatten()
    low = np.array(low).flatten()
    close = np.array(close).flatten()
    volume = np.array(volume).flatten()

    # 'Date': dates, 'Close': close, 'High': high, 'Low': low, 'Open': open, 'Volume': volume
    data = pd.DataFrame({'Date': dates, 'Close': close, 'High': high, 'Low': low, 'Open': open, 'Volume': volume})
    data.index = data.pop('Date')

    data = scaler.fit_transform(data)
    data = pd.DataFrame(data, columns=['Close', 'High', 'Low', 'Open', 'Volume'])	

    return data 

def rolling_window(dataframe: pd.DataFrame, window_size: int, features: int) -> pd.DataFrame:
    """
    Create a rolling window of the dataframe
    window_size: the size of the window (the amount of previous data to consider for each prediction)
    Probably will drop volume from the present day
    """
    df = dataframe.copy()
    for i in range(1, window_size + 1):
        for col in ['Close', 'High', 'Low', 'Open', 'Volume']:	
            df[f'{col}_shifted_{i}'] = df[col].shift(i)
    df.dropna(inplace=True)
    cols = df.columns.to_list()
    col = cols[features:] + cols[:1]
    df = df[col]
    data_reset = df.reset_index()
    data_reset.head()
    
    return data_reset

from typing import Tuple

def window_x_y_shape(dataframe: pd.DataFrame, window_size: int, features: int) -> Tuple[np.array, np.array, np.array]:
    """
    Split the dataframe into X and y
    X: the input data
    y: the output data
    """
    df_as_np = dataframe.to_numpy()
    dates = df_as_np[:, 0]
    middle_matrix = df_as_np[:, 1:-1]
    
    # Ensure the reshape is possible
    expected_elements = len(dates) * window_size * features
    actual_elements = middle_matrix.size
    
    if actual_elements != expected_elements:
        raise ValueError(f"Cannot reshape array of size {actual_elements} into shape ({len(dates)}, {window_size}, {features})")
    
    X = middle_matrix.reshape((len(dates), window_size, features))
    y = df_as_np[:, -1]
    
    return dates, X.astype(np.float32), y.astype(np.float32)

@app.get("/")
async def root():
    return {"message": "API up and running!"}

@app.post("/LSTM_Predict")

async def LSTM_Predict(stock_request: StockRequest):
    scaler_reverse = MinMaxScaler(feature_range=(0, 1))
    stock_data = yf.download(stock_request.ticker, start=stock_request.start_date, end=stock_request.end_date)
    scaler_reverse.fit(stock_data[['Close']])

    print(stock_data)

    data_prep = dataframe_prep(stock_data)

    print(data_prep)

    data_window = rolling_window(data_prep, WINDOW_SIZE, FEATURES)

    print(data_window)

    dates, X, y = window_x_y_shape(data_window, WINDOW_SIZE, FEATURES)

    print(X)

    mae = MeanAbsoluteError()
    model = load_model('model/MSFT_2020_20.h5', custom_objects={'mae': mae})
    predicted_prices = model.predict(X)

    print(predicted_prices)

    dates = dates.tolist()
    predicted_prices = scaler_reverse.inverse_transform(predicted_prices).flatten().tolist()
    

    return {"predicted_price": predicted_prices, "dates": dates}