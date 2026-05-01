# -*- coding:utf-8 -*-

import os
import sys

import numpy as np
import pandas as pd
from datetime import date, timedelta
#import modin.pandas as pd

MAX_DAILY_TRADE_CNT = 2  # 天级别交易股票的上限
TRADE_COST = 0.00125      # 交易成本
import math


def evaluate(fn):

    w_df = pd.read_pickle('data/000905.SH.csv.w.df.pickle')    
    zz500_index = pd.read_pickle('data/000905.SH.csv.index.df.pickle')
    z_index = {}
    for index,row in zz500_index.iterrows():
        z_index[row['date']] = row['ireturn']


    names = ['date', 'stock', 'pred', 'return']
    dtype = {'date': np.str_, 'stock': np.str_, 'return': np.float32,  'pred': np.float32}
    #dtype = {'date': 'category', 'stock': 'category', 'return': np.float32,  'pred': np.float32}
    df = pd.read_csv(sys.stdin, header=None, sep=' ', names=names, index_col=False, dtype=dtype, engine='c', na_filter=False, low_memory=False)
    #d = df['date'].unique().tolist()
    #df_date = pd.DataFrame({'date':d}) 
    #print(len(w_df))
    #print(df_date)
    #w_df = df.merge(w_df, on=['stock', 'date'], how='outer')    
    #print('#####')
    #print(w_df)
   

    th = 0.002
    #trade_df = df.loc[((df['pred'] > th) & (df['stocks'].str.startswith('600') ) )]
    #df['return'] = - df['return']
    #df['pred'] = - df['pred']
    #trade_df = df.loc[(df.pred > th)]
    trade_df = df
    df['sd'] = df['stock'] + df['date']
    w_df['sd'] = w_df['stock'] + w_df['date']
    #w_df  = w_df.drop(columns=['date', 'stock'])
    trade_df = df.merge(w_df, on=['date', 'stock'], how='inner') 
    #trade_df = df.join(w_df.set_index('sd'), on='sd')
    #print(trade_df)
    #trade_df = trade_df.loc[trade_df['return'] > -0.22]
    #trade_df = trade_df.loc[trade_df['stock'].isin(zz500)]
    #trade_df['cnt'] = (trade_df['pred'] * 1000).astype(np.int32)
    #trade_df['weight'] = trade_df['weight'].mask(trade_df['weight']>0.003, 0.003)
    trade_df['return'] = trade_df['return'] * trade_df['weight']
    #trade_df['return'] = trade_df['return'] 
    #trade_df['weight'] = 0
    ret = 1
    cash = 0.0
    for k, g_df in trade_df.groupby(by='date'):
        c1 = len(g_df)
        #g_df = g_df.loc[(g_df['pred'] > th)]
        c2 = len(g_df)
        #print(c1, c2)
        p = c2 / (c1 + c2) 
        #if  p < 0.09:
        if c2 <= 0:
            print('%s\t%s\t%s\t%s\t%s\t%s\t[wait]' % (k, len(g_df), ret, '', '', p))
            continue

        #c = g_df.groupby(by='cnt')['cnt'].count()
        # print(c)
        #c = g_df['cnt'].sum()
        #lz = g_df.loc[g_df['cnt'] < -3]['cnt'].count()
        #gz = g_df.loc[g_df['cnt'] > 3]['cnt'].count()
        #print(c, lz, gz)
        #rets = (g_df.nlargest(MAX_DAILY_TRADE_CNT, 'pred')['return']).mean()

        #rets = (g_df['return'].head(MAX_DAILY_TRADE_CNT).mean() - g_df['return'].tail(MAX_DAILY_TRADE_CNT).mean() ) /2
        #rets = -g_df['return'].tail(MAX_DAILY_TRADE_CNT).mean()
        #if rets == np.nan : continue
        
        #rets = (g_df['return'].head(MAX_DAILY_TRADE_CNT).sum() - g_df['return'].tail(MAX_DAILY_TRADE_CNT).sum() )
        th1 = 0.01
        th2 = -0.01
        rets = (g_df.loc[g_df['pred'] > th1]['return'].sum() - g_df.loc[g_df['pred'] < th2]['return'].sum() )
        #rets = g_df.loc[g_df['pred'] > th1 ]['return'].sum()
        #rets =  -g_df.loc[g_df['pred'] < th2]['return'].sum() 
        #g1_df = g_df.groupby(['name', 'country'])['score'].transform('quantile', 0.60)
        #g = g_df.groupby('stock')
        #g.apply(lambda r: r[r.score >= r.score.quantile(0.95)])
        
        w1 = g_df['weight'].head(MAX_DAILY_TRADE_CNT).sum()
        w2 = g_df['weight'].tail(MAX_DAILY_TRADE_CNT).sum()
        w1 = g_df.loc[g_df['pred'] > th1]['weight'].sum()
        w2 = g_df.loc[g_df['pred'] < th2]['weight'].sum()
        #w1 = 0
        #ww = min(w1, w2)
        w = (w1 + w2)/ 2
        cash = cash + w1 - w2
        #print(g_df['weight'].head(MAX_DAILY_TRADE_CNT).mean(), g_df['weight'].tail(MAX_DAILY_TRADE_CNT).mean())
        #rets = -g_df['return'].tail(MAX_DAILY_TRADE_CNT).mean()
        #rets = g_df['return'].head(MAX_DAILY_TRADE_CNT).mean()
        #for row i
	#rets = g_df['return'].sum()
        #w = w2
        ret = ret * (1 + rets - TRADE_COST * w)
        print('%s\t%s\t%s\t%s\t%s\t%s\t%.9f\t%.5f\t%.5f' % (k, len(g_df), ret, rets - TRADE_COST * w, rets, z_index[k], w1-w2, w1, w2))

if __name__ == '__main__':
    evaluate('../../result.txt.192')
