# -*- coding:utf-8 -*-

import os
import sys

import numpy as np
import pandas as pd
from datetime import date, timedelta

MAX_DAILY_TRADE_CNT = 100  # 天级别交易股票的上限
TRADE_COST = 0.00125      # 交易成本

def evaluate(fn):

    names = ['stock', 'date', 'weight']
    dtype = {'stock':np.str_, 'date':np.str_, 'weight':np.float32}
    w_df = pd.read_csv('../000905.SH.csv.w', header=None, sep=',', names=names, index_col=False, dtype=dtype, engine='c', na_filter=False, low_memory=False)
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

    names = ['date', 'stock', 'pred', 'return']
    dtype = {'date': np.str_, 'stock': np.str_, 'return': np.float32,  'pred': np.float32}
    df = pd.read_csv(sys.stdin, header=None, sep=' ', names=names, index_col=False, dtype=dtype, engine='c', na_filter=False, low_memory=False)

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
    trade_df['weight'] = trade_df['weight'].mask(trade_df['weight']>0.001, 0.001)
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
        #if rets == np.nan : continue
        
        rets = (g_df['return'].head(MAX_DAILY_TRADE_CNT).sum() - g_df['return'].tail(MAX_DAILY_TRADE_CNT).sum() )
        th1 = 0.005
        th2 = -0.005
        #rets = (g_df.loc[g_df['pred'] > th1]['return'].sum() - g_df.loc[g_df['pred'] < th2]['return'].sum() )
        #g1_df = g_df.groupby(['name', 'country'])['score'].transform('quantile', 0.60)
        #g = g_df.groupby('stock')
        #g.apply(lambda r: r[r.score >= r.score.quantile(0.95)])
        
        w1 = g_df['weight'].head(MAX_DAILY_TRADE_CNT).sum()
        w2 = g_df['weight'].tail(MAX_DAILY_TRADE_CNT).sum()
        #w1 = g_df.loc[g_df['pred'] > th1]['weight'].sum()
        #w2 = g_df.loc[g_df['pred'] < th2]['weight'].sum()
        #ww = min(w1, w2)
        w = (w1 + w2)/ 2
        cash = cash + w1 - w2
        #print(g_df['weight'].head(MAX_DAILY_TRADE_CNT).mean(), g_df['weight'].tail(MAX_DAILY_TRADE_CNT).mean())
        #rets = -g_df['return'].tail(MAX_DAILY_TRADE_CNT).mean()
        #rets = g_df['return'].head(MAX_DAILY_TRADE_CNT).mean()
        #for row i
	#rets = g_df['return'].sum()
        #w = 1
        ret = ret * (1 + rets - TRADE_COST * w)
        print('%s\t%s\t%s\t%s\t%s\t%.9f\t%.5f\t%.5f' % (k, len(g_df), ret, rets - TRADE_COST * w, '', w1-w2, w1, w2))

if __name__ == '__main__':
    evaluate('../../result.txt.192')
