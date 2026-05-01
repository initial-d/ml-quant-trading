# -*- coding:utf-8 -*-

import os
import sys

import numpy as np
import pandas as pd
#import modin.pandas as pd

MAX_DAILY_TRADE_CNT = 100  # 天级别交易股票的上限
TRADE_COST = 0.00125      # 交易成本


def evaluate(fn):

    zz500 = []
    for line in open('zz500.txt'):
        zz500.append(line.rstrip('\n\r'))


    #names = ['pred', 'return']
    #dtype  = {'pred':np.float32, 'return':np.float32}
    #df = pd.read_csv(sys.stdin, header=None, sep=' ', names=names, index_col=False, dtype=dtype, engine='c', na_filter=False, low_memory=False)
    #df['return'] = df['return'].shift(1)
    #df.dropna(inplace=True)
    #x = df['pred'].corr(df['return'])
    #print(x)
    #return

    names = ['date', 'stock', 'pred', 'return']
    dtype = {'date': np.str_, 'stock': np.str_, 'return': np.float32,  'pred': np.float32}
    dtype = {'date': 'category', 'stock': 'category', 'return': np.float32,  'pred': np.float32}
    dtype = {'date': np.str_, 'stock': np.str_, 'return': np.float32,  'pred': np.float32}
    df = pd.read_csv(sys.stdin, header=None, sep=' ', names=names, index_col=False, dtype=dtype, engine='c', na_filter=False, low_memory=False)

    names = ['index', 'stock', 'date', 'close', 'open', 'high', 'low', 'pre_close', 'change', 'pct_chg', 'vol', 'amount']
    dtype = {'date': np.str_, 'stock': np.str_, 'close': np.float32,  'pre_close': np.float32, 'change': np.float32}
    hs300_df = pd.read_csv('../000905.SH.csv.index', header=None, sep=',', names=names, dtype=dtype, engine='c', na_filter=False, low_memory=False)
    #hs300_df = pd.read_csv('../../000300.SH.csv.index', header=None, sep=',', names=names, engine='c', na_filter=False, low_memory=False)
    hs300_df['stock'] = hs300_df['stock'].str[:6]
    hs300_df['up'] = hs300_df['close'].shift(-1) / hs300_df['close'] - 1
    #print(hs300_df)


    tushare_fn = 'tushare.data'

    names = ['date', 'stock', 'open', 'close', 'high', 'low', 'avg', 'volume', 'fhzs', 'limit_up', 'limit_down', 'last_close', 'total_share']
    dtype = {'stock': np.str_, 'date': np.str_, 'open': np.float32, 'close': np.float32, 'high': np.float32, 'low': np.float32, 'avg': np.float32, 'volume': np.float32, 'fhzs': np.str_, 'limit_up': np.float32, 'limit_down': np.float32, 'last_close': np.float32, 'total_share':np.float32}
    daily_df = pd.read_csv(tushare_fn, header=None, sep='\t', names=names, index_col=False, dtype=dtype)
    daily_df['stock'] = daily_df['stock'].str[:6]
    daily_df['sd'] = daily_df['date'] + daily_df['stock']

    df['sd'] = df['date'] + df['stock']
    daily_df = daily_df.drop(columns=['date', 'stock'])

    df = df.join(daily_df.set_index('sd'), on='sd')
    print(df)
    #df['pred1'] = df['pred'] * np.log(df['volume'] * df['avg'])
    df['pred1'] = df['pred'] * np.log(df['total_share'] * df['avg']) 
    df['pred2'] = df['return'] * np.log(df['total_share'] * df['avg']) 
#
    th = 0.006
    #trade_df = df.loc[((df['pred'] > th) & (df['stocks'].str.startswith('600') ) )]
    #df['return'] = - df['return']
    #df['pred'] = - df['pred']
    trade_df = df
    #trade_df = df.loc[(df.pred > th)]
    trade_df = df.loc[(df['total_share'] * df['avg'] * 100 > 500000000)]
    #trade_df = df.loc[df['date'] < '20210901']
    #trade_df = trade_df.loc[trade_df['return'] < 0.22]
    #trade_df = trade_df.loc[trade_df['stock'].isin(zz500)]
    #trade_df['cnt'] = (trade_df['pred'] * 1000).astype(np.int32)
    dt = []
    rr1 = []
    rr2 = []
    rr3 = []
    rr4 = []
    rr5 = []
    rr6 = []
    for k, g_df in trade_df.groupby(by='date'):
        c1 = len(g_df)
        #g_df = g_df.loc[(g_df['pred'] > th)]
        gx_df = g_df.loc[(g_df['return'] > 0)]
        c2 = len(g_df)
        c3 = len(gx_df)
        #print(c1, c2)
        p = c2 / c1 
        #print(c2, c1+c2)
        p1 = 0
        if c2 > 0:
            p1 = c3 / c2
        #g_df = g_df.loc[(g_df['pred'] > th)]
        rets = g_df['return'].head(MAX_DAILY_TRADE_CNT).mean() ##- g_df['return'].tail(MAX_DAILY_TRADE_CNT).mean()
        preds = g_df['pred'].head(MAX_DAILY_TRADE_CNT).mean()
        #rpreds = g_df['pred1'].head(MAX_DAILY_TRADE_CNT).mean()
        rpreds = g_df['pred1'].mean()
        preturns = g_df['pred2'].mean()
        #dt.append((k, rets, p))
        dt.append(k)
        #if p < 0.1: rets = 0
        rr1.append(rets)
        rr2.append(p)
        rr3.append(preds)
        rr4.append(p1)
        rr5.append(rpreds)
        rr6.append(preturns)

    d = {'date': dt, 'return': rr1, 'pred1': rr2, 'pred': rr3, 'winr': rr4, 'rpreds':rr5, 'preturns':rr6}
    df = pd.DataFrame(d)
    #hs300_df['sd'] = hs300_df['date'] + df['stock']
    df = df.join(hs300_df.set_index('date'), on='date')
    df.dropna(inplace=True)
    print(df)
    x = df['return'].corr(df['pred'])
    x1 = df['return'].corr(df['pred'], method='spearman')
    #x = df['return'].corr(df['pred'])
    print('pred vs return',x, x1)
    x = df['return'].corr(df['pred1'])
    x1 = df['return'].corr(df['pred1'], method='spearman')
    print('ratio vs return',x, x1)
    x = df['return'].corr(df['rpreds'])
    x1 = df['return'].corr(df['rpreds'], method='spearman')
    print('pred * np.log(avg * total_shres) vs return',x, x1)
    
    x = df['up'].corr(df['return'])
    print('up return:', x)
    x = df['up'].corr(df['pred'])
    print('up pred:', x)
    x = df['up'].corr(df['preturns'])
    print('up return1:', x)
    x = df['up'].corr(df['rpreds'])
    print(x)
    x = df['up'].corr(df['pred1'])
    print(x)
    print('return:', df['return'].mean())
    df['net'] = df['return'] - df['up']
    print('net mean:',df['net'].mean())
    print(df['net'].std())
    print(df['return'].std())
    print(df['up'].std())
    df['winr1'] = df['winr'].shift(1)
    df.dropna(inplace=True)
    x = df['return'].corr(df['winr1'])
    print(x)
    df['return1'] = df['return'].shift(1)
    df.dropna(inplace=True)
    x = df['return1'].corr(df['return'])
    print(x)


if __name__ == '__main__':
    evaluate('../../result.txt.192')
