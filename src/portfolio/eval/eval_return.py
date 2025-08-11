# -*- coding:utf-8 -*-

import os
import sys
import numpy as np
import pandas as pd


MAX_DAILY_TRADE_CNT = 100  # 天级别交易股票的上限
TRADE_COST = 0.00125      # 交易成本
P = 1000


def search_thres(df, ratio=0.2):
    s = 0
    e = 5
    th = 0
    w = 0
    while e - s > 0.000001 and int(w*P) != int(ratio*P):
        t = (s + e) / 2
        th = t
        #w = df.mask(df['pred'] > t, t)['pred'].mean()
        #w = len(df.loc[df['pred'] > th]) / len(df)
        w = len(df.loc[df['pred'] > th])/len(df)
        #print(w, th, ratio, s, e, w> ratio)

        if w < ratio:
            e = t
        else:
            s = t
    return th


def evaluate(fn):

    names = ['date', 'stock', 'pred', 'return']
    dtype = {'date': np.str_, 'stock': np.str_, 'return': np.float32,  'pred': np.float32}
    df = pd.read_csv(sys.stdin, header=None, sep='\t', names=names, index_col=False, dtype=dtype, engine='c', na_filter=False, low_memory=False)
    df['net_return'] = df['return'] - 0.00125
    df['creturn'] = (df['return'] * 100).astype(np.int32).abs()
    df['cpred'] = (df['pred'] * 100).astype(np.int32).abs()

    #zz500_index = pd.read_pickle('data/000905.SH.csv.index.df.pickle')
    #z_index = {}
    #for index, row in zz500_index.iterrows():
    #    z_index[row['date']] = row['ireturn']

    #tushare_df = pd.read_pickle('data/tushare.data.df.pickle')
    #df = df.merge(tushare_df, on=['date', 'stock'], how='inner')
    df.sort_values(by=['date', 'pred'], ascending=[True, False], inplace=True)
    #df['market_cap'] = df['avg'] * df['total_share'] * 100
    stats_df = pd.DataFrame()
    g_date_df = df.groupby(by='date')

    #stats_df['pred'] = g_date_df.apply(lambda d: (d['pred']).mean())
    #stats_df['return'] = g_date_df.apply(lambda d: d.head(MAX_DAILY_TRADE_CNT)['return'].mean())
    #stats_df['pred1'] = g_date_df.apply(lambda d: len(d.loc[d['pred'] > 0.006])/len(d))
    #th = search_thres(stats_df, ratio=0.2)
    #print(stats_df.loc[ stats_df['pred'] > th]['return'].mean())
    #print(stats_df.loc[ stats_df['pred'] < th]['return'].mean())
    #th =stats_df['pred'].mean()
    df = df.loc[df.pred >= 0.5]
    #df['pred'] = df['pred'] ** 3
    df['pred'] = 1

    stats_df['pred'] = g_date_df.apply(lambda d: d.head(MAX_DAILY_TRADE_CNT)['pred'].sum())
    # print(stats_df)
    th = search_thres(stats_df, ratio=0.7)
    #th = stats_df['pred'].mean()

    ret = 1
    for k, g_df in df.groupby(by='date'):
        w = g_df['pred'].mean()
        #w1= len(g_df.loc[g_df['pred'] > 0.006])/len(g_df)
        x = 1
        #idx = z_index[k]
        d = g_df.head(MAX_DAILY_TRADE_CNT)
        #d = g_df.loc[g_df['pred'] > 0.005 ** 3]
        rets = (d['return'] * d['pred'] / d['pred'].sum()).sum()
        t = g_df.head(MAX_DAILY_TRADE_CNT)['pred'].sum()
        # print(d['pred']/d['pred'].sum())
        th = -100
        if t < th:
            th1 = th * 1
            # print(th1)
            rets = (d['return'] * d['pred'] / th1).sum()
            x = (d['pred'] / th1).sum()

        #rets = g_df['return'].head(MAX_DAILY_TRADE_CNT).mean()
        #rets = (1 + rets) / ( 1+idx * 2 ) - 1
        n = len(g_df['return'].head(MAX_DAILY_TRADE_CNT))
        # if w < -0.006:
        #    x = (w + 0.016) * 100
        #    if x < 0: x = 0
        #    pass
        #    #ret = ret * (1 + rets - TRADE_COST)
        #    print('%s\t%s\t%s\t%s\t[WAIT]' % (k, n, ret, rets))
        #    #continue

        ret = ret * ((1+rets - TRADE_COST) * x + (1-x))
        print('%s\t%s\t%s\t%s\t[BUY]' % (k, n, ret, rets))


if __name__ == '__main__':
    evaluate('../trade.txt')
