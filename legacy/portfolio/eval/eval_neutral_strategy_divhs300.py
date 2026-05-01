# -*- coding:utf-8 -*-

import os
import sys

import numpy as np
import pandas as pd
from datetime import date, timedelta

MAX_DAILY_TRADE_CNT = 120  # 天级别交易股票的上限
TRADE_COST = 0.0012      # 交易成本
P = 200


def search_thres(df, ratio=0.2):
    s = 0
    e = 0.01
    th = 0
    w = 0
    while e - s > 0.000001 and int(w*P) != int(ratio*P):
        t = (s + e) / 2
        th = t
        w = df.mask(df['weight'] > t, t)['weight'].sum()
        if w > ratio:
            e = t
        else:
            s = t
    return th


def evaluate(fn):

    #w_df = pd.read_pickle('data/000905.SH.csv.w.df.pickle')
    w_df = pd.read_pickle('data/index_weights_000905.XSHG.df.pickle')
    zz500_index = pd.read_pickle('data/000905.SH.csv.index.df.pickle')
    z_index = {}
    for index, row in zz500_index.iterrows():
        z_index[row['date']] = row['ireturn']

    names = ['date', 'stock', 'pred', 'return']
    dtype = {'date': np.str_, 'stock': np.str_, 'return': np.float32,  'pred': np.float32}
    #dtype = {'date': 'category', 'stock': 'category', 'return': np.float32,  'pred': np.float32}
    df = pd.read_csv(sys.stdin, header=None, sep=' ', names=names, index_col=False, dtype=dtype, engine='c', na_filter=False, low_memory=False)

    trade_df = df.merge(w_df, on=['date', 'stock'], how='inner')

    th = 0.008
    ret = 1
    for k, g_df in trade_df.groupby(by='date'):

        head_df = g_df.head(MAX_DAILY_TRADE_CNT)
        #head_df = g_df[g_df.pred > th]
        tail_df = g_df.tail(MAX_DAILY_TRADE_CNT)
        #tail_df = g_df[g_df.pred < -th]

        head_th = search_thres(head_df, ratio=0.50)
        tail_th = search_thres(tail_df, ratio=0.50)
        w1 = head_df.mask(head_df['weight'] > head_th, head_th)['weight'].sum()
        w2 = tail_df.mask(tail_df['weight'] > tail_th, tail_th)['weight'].sum()
        m_ratio = min(w1, w2)
        head_th = search_thres(head_df, ratio=m_ratio)
        tail_th = search_thres(tail_df, ratio=m_ratio)

        head_ret = (head_df.mask(head_df['weight'] > head_th, head_th)['weight'] * head_df['return']).sum()
        tail_ret = (tail_df.mask(tail_df['weight'] > tail_th, tail_th)['weight'] * tail_df['return']).sum()
        rets = head_ret - tail_ret
        w1 = head_df.mask(head_df['weight'] > head_th, head_th)['weight'].sum()
        w2 = tail_df.mask(tail_df['weight'] > tail_th, tail_th)['weight'].sum()
        w = (w1 + w2) / 2

        ret = ret * (1 + rets - TRADE_COST * w)
        print('%s\t%s\t%s\t%s\t%s\t%s\t%.9f\t%.5f\t%.5f' % (k, len(g_df), ret, rets - TRADE_COST * w, rets, z_index[k], w1-w2, w1, w2))


if __name__ == '__main__':
    evaluate('../../result.txt.192')
