# -*- coding:utf-8 -*-

import os
import sys

import numpy as np
import pandas as pd
from datetime import date, timedelta


def evaluate(fn):

    for fn in ['index_weights_000300.XSHG', 'index_weights_000902.XSHG', 'index_weights_000905.XSHG', 'index_weights_000907.XSHG']:
        fn = 'data/' + fn
        names = ['stock', 'date', 'weight']
        dtype = {'stock':np.str_, 'date':np.str_, 'weight':np.float32}
        w_df = pd.read_csv(fn, header=None, sep=',', names=names, index_col=False, dtype=dtype, engine='c', na_filter=False, low_memory=False)
        w_df['stock'] = w_df['stock'].str[:6]
        w_df['date'] = w_df['date'].str.replace('-', '')
        last_k = ''
        last_v = ''
        res = []
        for k, g_df in w_df.groupby(by='date'):
            if last_k != '':
                sdate = date(int(last_k[:4]), int(last_k[4:6]), int(last_k[6:8]))             
                edate = date(int(k[:4]), int(k[4:6]), int(k[6:8])) -  timedelta(days=1)            
                
                g_df['weight'] = g_df['weight']/ g_df['weight'].sum()
                res.append(g_df)
                while sdate < edate:
                    sdate = sdate + timedelta(days=1)
                    day = sdate.strftime("%Y%m%d")
                    t_df = g_df.reset_index()
                    t_df['date'] = day
                    t_df['weight'] = t_df['weight']/ t_df['weight'].sum()
                    res.append(t_df)
            last_k = k
        w_df = pd.concat(res, axis=0)
        w_df.sort_values(by=['stock', 'date'], ascending=True, inplace=True)
        w_df.ffill(axis=0, inplace=True)
        w_df.to_pickle(fn + '.df.pickle')
        

    tushare_fn = 'data/tushare.data'
    tushare_fn_pickle = 'data/tushare.data.df.pickle'

    names = ['date', 'stock', 'open', 'close', 'high', 'low', 'avg', 'volume', 'fhzs', 'limit_up', 'limit_down', 'last_close', 'total_share']
    dtype = {'stock': np.str_, 'date': np.str_, 'open': np.float32, 'close': np.float32, 'high': np.float32, 'low': np.float32, 'avg': np.float32, 'volume': np.float32, 'fhzs': np.str_, 'limit_up': np.float32, 'limit_down': np.float32, 'last_close': np.float32, 'total_share':np.float32}
    tushare_df = pd.read_csv(tushare_fn, header=None, sep='\t', names=names, index_col=False, dtype=dtype)
    tushare_df['stock'] = tushare_df['stock'].str[:6]
    tushare_df.to_pickle(tushare_fn_pickle)

    
    index_fn = 'data/000300.SH.csv.index'
    index_fn_pickle = 'data/000300.SH.csv.index.df.pickle'

    names = ['index', 'stock', 'date', 'close', 'open', 'high', 'low', 'pre_close', 'change', 'pct_chg', 'vol', 'amount']
    dtype = {'date': np.str_, 'stock': np.str_, 'close': np.float32,  'pre_close': np.float32, 'change': np.float32}
    hs300_df = pd.read_csv(index_fn, header=None, sep=',', names=names, dtype=dtype, engine='c', na_filter=False, low_memory=False)
    #hs300_df = pd.read_csv('data/data/000300.SH.csv.index', header=None, sep=',', names=names, engine='c', na_filter=False, low_memory=False)
    hs300_df['stock'] = hs300_df['stock'].str[:6]
    hs300_df['ireturn'] = hs300_df['close'].shift(-1) / hs300_df['close'] - 1
    hs300_df['cireturn'] = hs300_df['close'] / hs300_df['close'].shift() - 1
    hs300_df.to_pickle(index_fn_pickle)

   
    index_fn = 'data/000905.SH.csv.index'
    index_fn_pickle = 'data/000905.SH.csv.index.df.pickle'

    names = ['index', 'stock', 'date', 'close', 'open', 'high', 'low', 'pre_close', 'change', 'pct_chg', 'vol', 'amount']
    dtype = {'date': np.str_, 'stock': np.str_, 'close': np.float32,  'pre_close': np.float32, 'change': np.float32}
    hs300_df = pd.read_csv(index_fn, header=None, sep=',', names=names, dtype=dtype, engine='c', na_filter=False, low_memory=False)
    #hs300_df = pd.read_csv('data/data/000300.SH.csv.index', header=None, sep=',', names=names, engine='c', na_filter=False, low_memory=False)
    hs300_df['stock'] = hs300_df['stock'].str[:6]
    hs300_df['ireturn'] = hs300_df['close'].shift(-1) / hs300_df['close'] - 1
    hs300_df['cireturn'] = hs300_df['close'] / hs300_df['close'].shift() - 1
    hs300_df.to_pickle(index_fn_pickle)

   
    index_fn = 'data/000904.SH.csv.index'
    index_fn_pickle = 'data/000904.SH.csv.index.df.pickle'

    names = ['index', 'stock', 'date', 'close', 'open', 'high', 'low', 'pre_close', 'change', 'pct_chg', 'vol', 'amount']
    dtype = {'date': np.str_, 'stock': np.str_, 'close': np.float32,  'pre_close': np.float32, 'change': np.float32}
    hs300_df = pd.read_csv(index_fn, header=None, sep=',', names=names, dtype=dtype, engine='c', na_filter=False, low_memory=False)
    #hs300_df = pd.read_csv('data/data/000300.SH.csv.index', header=None, sep=',', names=names, engine='c', na_filter=False, low_memory=False)
    hs300_df['stock'] = hs300_df['stock'].str[:6]
    hs300_df['ireturn'] = hs300_df['close'].shift(-1) / hs300_df['close'] - 1
    hs300_df['cireturn'] = hs300_df['close'] / hs300_df['close'].shift() - 1
    hs300_df.to_pickle(index_fn_pickle)

if __name__ == '__main__':
    evaluate('data/data/result.txt.192')
