# -*- coding:utf-8 -*-

import sys
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
    #daily_df = pd.read_csv('../newest_data', header=None, sep='\t', names=names, index_col=False, dtype=dtype)
    daily_df = pd.read_csv('newest_data', header=None, sep='\t', names=names, index_col=False, dtype=dtype)
    daily_df['stock'] = daily_df['stock'].str[:6]
    daily_df.sort_values(by=['stock', 'date'], ascending=True, inplace=True)
    daily_df = daily_df.loc[daily_df.date > '20220101']
    res = []
    for k, g_df in daily_df.groupby(['stock']):

        g_df['last_9_lowest'] = g_df['low'].shift(1).rolling(8, min_periods=1).min()
        g_df['last_8_lowest'] = g_df['low'].shift(1).rolling(7, min_periods=1).min()
        g_df['last_7_lowest'] = g_df['low'].shift(1).rolling(6, min_periods=1).min()
        g_df['last_6_lowest'] = g_df['low'].shift(1).rolling(5, min_periods=1).min()
        g_df['last_5_lowest'] = g_df['low'].shift(1).rolling(4, min_periods=1).min()
        g_df['last_4_lowest'] = g_df['low'].shift(1).rolling(3, min_periods=1).min()
        g_df['last_3_lowest'] = g_df['low'].shift(1).rolling(2, min_periods=1).min()
        g_df['last_2_lowest'] = g_df['low'].shift(1).rolling(1, min_periods=1).min()

        g_df['last_close'] = g_df['close'].shift(1)

        g_df['end_close'] = g_df['close'].shift(-8)

        g_df['during_9_lowest'] = g_df['last_9_lowest'].shift(-8)
        g_df['down_ratio_9'] = g_df['during_9_lowest'] / g_df['last_close'] - 1
        g_df['down_ratio_mean_9'] = g_df['down_ratio_9'].mean()

        g_df['during_8_lowest'] = g_df['last_8_lowest'].shift(-7)
        g_df['down_ratio_8'] = g_df['during_8_lowest'] / g_df['last_close'] - 1
        g_df['down_ratio_mean_8'] = g_df['down_ratio_8'].mean()

        g_df['during_7_lowest'] = g_df['last_7_lowest'].shift(-6)
        g_df['down_ratio_7'] = g_df['during_7_lowest'] / g_df['last_close'] - 1
        g_df['down_ratio_mean_7'] = g_df['down_ratio_7'].mean()

        g_df['during_6_lowest'] = g_df['last_6_lowest'].shift(-5)
        g_df['down_ratio_6'] = g_df['during_6_lowest'] / g_df['last_close'] - 1
        g_df['down_ratio_mean_6'] = g_df['down_ratio_6'].mean()

        g_df['during_5_lowest'] = g_df['last_5_lowest'].shift(-4)
        g_df['down_ratio_5'] = g_df['during_5_lowest'] / g_df['last_close'] - 1
        g_df['down_ratio_mean_5'] = g_df['down_ratio_5'].mean()

        g_df['during_4_lowest'] = g_df['last_4_lowest'].shift(-3)
        g_df['down_ratio_4'] = g_df['during_4_lowest'] / g_df['last_close'] - 1
        g_df['down_ratio_mean_4'] = g_df['down_ratio_4'].mean()

        g_df['during_3_lowest'] = g_df['last_3_lowest'].shift(-2)
        g_df['down_ratio_3'] = g_df['during_3_lowest'] / g_df['last_close'] - 1
        g_df['down_ratio_mean_3'] = g_df['down_ratio_3'].mean()

        g_df['during_2_lowest'] = g_df['last_2_lowest'].shift(-1)
        g_df['down_ratio_2'] = g_df['during_2_lowest'] / g_df['last_close'] - 1
        g_df['down_ratio_mean_2'] = g_df['down_ratio_2'].mean()

        
        res.append(g_df)
    daily_df = pd.concat(res, axis=0)
    print(daily_df)

    #daily_df['buy_price'] = np.where(daily_df['last_close'] < daily_df['open'], daily_df['last_close'], daily_df['open'])
    #daily_df['down_ratio'] = daily_df['during_9_lowest'] / daily_df['last_close'] - 1
    #daily_df['down_ratio_mean'] = daily_df['down_ratio'].mean()


    #df.sort_values(by=['stock', 'date'], ascending=True, inplace=True)
    #res = []
    #for k, g_df in df.groupby(['stock']):
    #    #g_df['v'] = g_df['return'].shift(1).rolling(30, min_periods=1).std()
    #    #g_df['v60'] = g_df['return'].shift(1).rolling(60, min_periods=1).std()
    #    #g_df['v20'] = g_df['return'].shift(1).rolling(20, min_periods=1).std()
    #    #g_df['flag'] = np.where(g_df['return'] > 0, 1, 0)
    #    g_df['d'] = g_df['flag'].rolling(10, min_periods=1).sum()
    #    res.append(g_df)
    #df = pd.concat(res, axis=0)

    df = daily_df
    #df.sort_values(by=['date'], ascending=[True], inplace=True)
    df = df.dropna(subset=['end_close'])


    df['g_down_mean_2'] = df['down_ratio_2'].mean()
    df['g_down_mean_3'] = df['down_ratio_3'].mean()
    df['g_down_mean_4'] = df['down_ratio_4'].mean()
    df['g_down_mean_5'] = df['down_ratio_5'].mean()
    df['g_down_mean_6'] = df['down_ratio_6'].mean()
    df['g_down_mean_7'] = df['down_ratio_7'].mean()
    df['g_down_mean_8'] = df['down_ratio_8'].mean()
    df['g_down_mean_9'] = df['down_ratio_9'].mean()

    df = df[['date', 'stock', 'open', 'close', 'high', 'low', 'avg', 'volume', 'fhzs', 'limit_up', 'limit_down', 'last_2_lowest', 'last_close', 'end_close', 'during_2_lowest', 'down_ratio_2', 'down_ratio_mean_2', 'g_down_mean_2']]


    df.to_csv(sys.stdout, sep=' ', header=True, index=False)


if __name__ == '__main__':
    generate_trade_df()
