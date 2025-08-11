# -*- coding:utf-8 -*-

import numpy as np
import pandas as pd

import time
from datetime import date


def generate_trade_df():

    #today = date.fromtimestamp(time.time()).strftime('%Y%m%d')
    #daily_fn = '/home/guochenglin/xproject-data/data/%s.data.ochl' % (today)
    #trade_fn = '/home/guochenglin/xproject-data/data/%s.result.txt' % (today)
    #trade_df_fn = '/home/guochenglin/xproject-data/data/%s.trade.pickle' % (today)

    names = ['date', 'stock', 'open', 'close', 'high', 'low', 'avg', 'volume', 'fhzs', 'limit_up', 'limit_down', 'last_close']
    dtype = {'stock': np.str_, 'date': np.str_, 'open': np.float32, 'close': np.float32, 'high': np.float32, 'low': np.float32, 'avg': np.float32, 'volume': np.float32, 'fhzs': np.str_, 'limit_up': np.float32, 'limit_down': np.float32, 'last_close': np.float32}
    daily_df = pd.read_csv('data/tushare.data', header=None, sep='\t', names=names, index_col=False, dtype=dtype)
    daily_df['stock'] = daily_df['stock'].str[:6]
    daily_df['sd'] = daily_df['date'] + daily_df['stock']
    print(daily_df)

    trade_fn = '/home/t-guochenglin-cj/result.txt.v7'
    names = ['date', 'stock', 'h_pred',  'h_return']
    dtype = {'date': np.str_, 'stock': np.str_, 'h_return': np.float32,  'h_pred': np.float32}
    high_df = pd.read_csv(trade_fn, header=None, sep=' ', names=names, index_col=False, dtype=dtype)
    high_df['sd'] = high_df['date'] + high_df['stock']
    high_df = high_df.drop(columns=['date', 'stock'])
    daily_df = daily_df.join(high_df.set_index('sd'), on='sd')
    print(daily_df)

    daily_df_fn = '/da1/public/duyimin/trade_back/daily.df'
    d = daily_df.set_index(keys=['stock', 'date'], drop=False) 
    d.to_pickle(daily_df_fn)

    trade_fn = '/da1/public/guochenglin/trade/trade.stk'
    trade_fn = '/home/t-guochenglin-cj/trade.txt'
    trade_fn = 'trade.f'
    #names = ['date', 'stock', 'return',  'pred']
    #dtype = {'date': np.str_, 'stock': np.str_, 'return': np.float32,  'pred': np.float32}
    names = ['date', 'stock', 'pred',  'return']
    dtype = {'date': np.str_, 'stock': np.str_, 'pred': np.float32,  'return': np.float32}
    df = pd.read_csv(trade_fn, header=None, sep=' ', names=names, index_col=False, dtype=dtype)
    df['stock'] = df['stock'].str[:6]

    print(df)

    df['sd'] = df['date'] + df['stock']
    daily_df = daily_df.drop(columns=['date', 'stock'])

    df = df.join(daily_df.set_index('sd'), on='sd')

    print('df')
    print(df)
    

    df.set_index(keys=['stock', 'date'], drop=False,inplace=True)
    trade_df_fn = '/da1/public/duyimin/trade_back/trade.df'
    df.to_pickle(trade_df_fn)

if __name__ == '__main__':
    generate_trade_df()
