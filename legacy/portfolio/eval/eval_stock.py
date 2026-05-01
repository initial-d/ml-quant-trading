# -*- coding:utf-8 -*-

import os
import sys
import numpy as np
import pandas as pd

MAX_DAILY_TRADE_CNT = 20  # 天级别交易股票的上限
TRADE_COST = 0.00125      # 交易成本


def evaluate(fn):

    names = ['date', 'stock', 'pred', 'return']
    dtype = {'date': np.str_, 'stock': np.str_, 'return': np.float32,  'pred': np.float32}
    df = pd.read_csv(sys.stdin, header=None, sep=' ', names=names, index_col=False, dtype=dtype, engine='c', na_filter=False, low_memory=False)
    df['net_return'] = df['return'] - 0.00125

    #print(df['pred'].corr(df['return']))
    #df.sort_values(by=['date', 'pred'], ascending=[True, False], inplace=True)
    #df.loc[df['stock']== '000001'].to_csv(sys.stdout, sep='\t', index=False)
    #df.to_csv(sys.stdout, sep='\t', index=False)
    #df['pred'] = df['pred'] ** 2.6
    #df = df.dropna()
    #df = df.loc[df['corr'] > 0.2]
    ret = 1
    for k, g_df in df.groupby(by='date'):
        w = g_df['pred'].mean()
        x = 1
        d = g_df.head(MAX_DAILY_TRADE_CNT)
        rets = (d['return'] * d['pred'] / d['pred'].sum()).sum()
        n = len(g_df['return'].head(MAX_DAILY_TRADE_CNT))
        ret = ret * ((1+rets - TRADE_COST) * x + (1-x))
        print('%s\t%s\t%s\t%s\t[BUY]' % (k, n, ret, rets))


if __name__ == '__main__':
    evaluate('./trade.txt')
