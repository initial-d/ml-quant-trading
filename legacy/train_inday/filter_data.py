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

    names = ['date', 'stock', 'open', 'close', 'high', 'low', 'avg', 'volume', 'fhzs', 'limit_up', 'limit_down']
    dtype = {'stock': np.str_, 'date': np.str_, 'open': np.float32, 'close': np.float32, 'high': np.float32, 'low': np.float32, 'avg': np.float32, 'volume': np.float32, 'fhzs': np.str_, 'limit_up': np.float32, 'limit_down': np.float32}
    daily_df = pd.read_csv('../train_classify/tushare.data', header=None, sep='\t', names=names, index_col=False, dtype=dtype)
    daily_df['stock'] = daily_df['stock'].str[:6]
    daily_df['sd'] = daily_df['date'] + daily_df['stock']
    daily_df.sort_values(by=['stock', 'date'], ascending=True, inplace=True)
    daily_df['fhzs_1'] = daily_df['fhzs'].shift(-1)
    #print(daily_df)


    trade_fn = 'tushare_day.v12'
    names = ['date', 'stock', 'pred',  'return']
    dtype = {'date': np.str_, 'stock': np.str_, 'pred': np.float32,  'return': np.float32}
    df = pd.read_csv(trade_fn, header=None, sep=' ', names=names, index_col=False, dtype=dtype)
    df['stock'] = df['stock'].str[:6]

    #print(df)

    df['sd'] = df['date'] + df['stock']
    daily_df = daily_df.drop(columns=['date', 'stock'])

    df = df.join(daily_df.set_index('sd'), on='sd')
    #df = df.loc[ ( ((df.limit_up / df.close - 1) > 0.01)) & (df.avg < 300)] 
    df = df.loc[ ((df.limit_up / df.close - 1) > 0.01) & (df.volume * df.avg * 100 > 50000000) ] 
    #df = df.loc[ ((df.limit_up / df.close - 1) > 0.02) & (df.volume  * 100 > 60000000) ] 
    df = df.loc[ df.fhzs_1 != "1"] 
    #df = df.loc[ ((df.limit_up / df.close - 1) > 0.01)] 
    #df = df.loc[ (  ((df.volume * df.avg * 100) > 100000000) )] 

    #print(df)
    #df.set_index(keys=['stock', 'date'], drop=False,inplace=True)
    df.sort_values(by=['date', 'pred'], ascending=[True, False], inplace=True)
    df.to_csv('trade.txt', sep=' ', header=False, index=False)


if __name__ == '__main__':
    generate_trade_df()
