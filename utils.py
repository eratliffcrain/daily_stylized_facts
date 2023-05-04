import pandas as pd
import numpy as np
import matplotlib.pylab as plt
from datetime import date
from statsmodels.tsa.seasonal import seasonal_decompose, STL
from statsmodels.tsa.stattools import adfuller
from scipy.stats import norm
import seaborn as sns

symbols = ['AAPL', 'AMZN', 'BRK-A', 'BRK-B', 'JNJ', 'MSFT', 'NVDA',
           'TSLA', 'V', 'XOM']
#symbols = ['Shuffled', 'Rand. Walk']
#symbols = ['RAND']

def natural_log_returns(prices: pd.Series) -> pd.Series:
    """calculates log returns as the difference of the log price series."""
    return prices.apply(np.log).diff().dropna()

def returns(prices: pd.Series) -> pd.Series:
    """
    calculates returns as the difference of the log price series.
    assumes prices already log-transformed
    """
    return prices.diff().dropna()

def read_data(sym, stl=False, period=20):
    if sym == 'Shuffled':
        df = read_data('AAPL', tsl=True)
        sh = df['price'].values
        np.random.seed(5)
        np.random.shuffle(sh)
        df['price'] = sh
        return df
    elif sym == 'Rand. Walk':
        df = random_walk()
    else:
        df = pd.read_parquet(f"./data/updated/HistoricalData_{sym}.parquet")

    df.set_index(['Date'], inplace=True)
    df['price'] = np.log(df['price'])
    
    if stl:
        decomposition = STL(df['price'], period = period).fit()
        p_0 = df['price'][0]
        df['price'] = decomposition.resid + p_0  
        
    df.dropna(inplace=True)
    
    return df


def update():
    for sym in symbols:
        df = pd.read_csv(f"./data/HistoricalData_{sym}.csv")
        if sym != 'SNP':
            df['price'] = df['Close/Last'].str.replace('$', '', regex=True).astype(float)
        else:
            df['price'] = df['Close']
            df['Volume'] = 1
            
        df['Date']= pd.to_datetime(df['Date'])
        df.to_parquet(f"./data/updated/HistoricalData_{sym}.parquet")

def random_walk():
    # Probability to move up or down
    prob = 0.5
     
    # statically defining the starting position
    aapl = read_data('AAPL')
    start = aapl['price'][0]
    rets = returns(aapl['price'])
    mu, std = norm.fit(rets)
    
    normal_rets = np.random.normal(loc=mu,scale=std,size=len(rets))
    
    prices = [start]
    p = start
    for i in range(len(rets)):
        r = normal_rets[i]
        if np.random.rand() < prob:
            r = normal_rets[i] * -1

        p = p + r
        prices.append(p)
    
    df = aapl.copy(deep=True)[['Volume']]
    df['price'] = prices
    df.reset_index(inplace=True)
    return df


def consolidate_data_vwap(sym, period='M', stl=False):
    df = read_data(sym, stl=stl)
    if period=='D':
        return df
    df['price'] = df['price'] * df['Volume']
    df = (
        df.resample(period).agg({"price": "sum", "Volume": "sum"})
    )
    df['price'] = df['price'] / df['Volume']
    
    return df

def consolidate_data(sym, period='M', stl=False):
    df = read_data(sym, stl=stl)
    if period=='D':
        return df
    df = (
        df.resample(period).agg({"price": "last", "Volume": "sum"})
    )
    
    return df

