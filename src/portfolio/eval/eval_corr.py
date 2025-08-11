# -*- coding:utf-8 -*-

import os
import sys
import numpy as np
import pandas as pd


MAX_DAILY_TRADE_CNT = 50  # 天级别交易股票的上限
TRADE_COST = 0.00125      # 交易成本


def evaluate(fn):

    names = ['date', 'stock', 'pred', 'return']
    dtype = {'date': np.str_, 'stock': np.str_, 'return': np.float32,  'pred': np.float32}
    df = pd.read_csv(sys.stdin, header=None, sep=' ', names=names, index_col=False, dtype=dtype, engine='c', na_filter=False, low_memory=False)
    df['net_return'] = df['return'] - 0.00125
    df['creturn'] = (df['return'] * 100).astype(np.int32).abs()
    df['cpred'] = (df['pred'] * 100).astype(np.int32).abs()

    hs300_df = pd.read_pickle('data/000905.SH.csv.index.df.pickle')
    tushare_df = pd.read_pickle('data/tushare.data.df.pickle')

    df = df.merge(tushare_df, on=['date', 'stock'], how='inner')
    df.sort_values(by=['date', 'pred'], ascending=[True, False], inplace=True)
    df['market_cap'] = df['avg'] * df['total_share'] * 100 
    g_date_df = df.groupby(by='date')
    stats_df = pd.DataFrame()
    stats_df['pred'] = g_date_df.apply(lambda d: (d['pred']).mean())
    stats_df['predr'] = g_date_df.apply(lambda d: len(d.loc[d['pred'] > 0.006])/len(d))
    stats_df['wpred'] = g_date_df.apply(lambda d: (d['pred'] * np.log(d['avg'] * d['volume'])).mean())
    stats_df['return'] = g_date_df.apply(lambda d: (d['return'].head(MAX_DAILY_TRADE_CNT)).mean())
    #stats_df['pred'] = g_date_df.apply(lambda d: (d.loc[d['pred'] > 0.003]).mean())
    stats_df = stats_df.reset_index()

    print(stats_df)
    #stats_df.to_csv(sys.stdout, sep='\t', index=False)
    stats_df = stats_df.join(hs300_df.set_index('date'), on='date')
    #for th1, th2 in  ((i*0.001, (i+1)*0.001)  for i in range(10)):
    total_cnt = len(df)
    for i in range(12):
        th = i * 0.001
        t_df =  df.loc[ df['pred'] > th ]
        w1 = t_df['net_return'].mean()
        cnt = t_df['return'].count()
        print('%.3f\t%.5f\t%d\t%.3f' % (th, w1, cnt, cnt/total_cnt) )
        #d = t_df.groupby(['creturn'])['return'].transform('sum')
        d = t_df.groupby(['creturn'])['net_return'].sum()
        d = d.reset_index()
        d['count'] = t_df.groupby(['creturn'])['net_return'].count()
        d['a_return'] = d['net_return']/ d['count']
        d.to_csv(sys.stdout, sep='\t', index=False)

    for i in range(20):
        th = i * 0.001
        t_df =  df.loc[ df['pred'] > th ]
        w1 = t_df['net_return'].mean()
        cnt = t_df['return'].count()
        print('%.3f\t%.5f\t%d\t%.3f' % (th, w1, cnt, cnt/total_cnt) )

#    for j in range(20):
#        th1 = j * 0.001
#        for i in range(50):
#            th = i * 10000000 + 60000000
#            t_df =  df.loc[ (df['market_cap'] > th) & (df['pred'] > th1) ]
#            w1 = t_df['net_return'].mean()
#            cnt = t_df['return'].count()
#            if cnt > 10000:
#                print('%.3f\t%.3f\t%.5f\t%d\t%.3f' % (th1, th, w1, cnt, cnt/total_cnt) )

    print('pred corr return:', stats_df['return'].corr(stats_df['pred'], method='pearson'), stats_df['return'].corr(stats_df['pred'], method='spearman'))
    print('predr corr return:', stats_df['return'].corr(stats_df['predr'], method='pearson'), stats_df['return'].corr(stats_df['predr'], method='spearman'))
    print('wpred corr return:', stats_df['return'].corr(stats_df['wpred'], method='pearson'), stats_df['return'].corr(stats_df['wpred'], method='spearman'))
    print('pred corr hs300 index return:', stats_df['ireturn'].corr(stats_df['pred'], method='pearson'), stats_df['ireturn'].corr(stats_df['pred'], method='spearman'))
    print('wpred corr hs300 index return:', stats_df['ireturn'].corr(stats_df['wpred'], method='pearson'), stats_df['ireturn'].corr(stats_df['wpred'], method='spearman'))
    print('return corr hs300 index return:', stats_df['ireturn'].corr(stats_df['return'], method='pearson'), stats_df['ireturn'].corr(stats_df['return'], method='spearman'))
    #stats_df['pred'] = stats_df['pred']-stats_df['pred'].mean()
    print(stats_df['return'].mean(), len(stats_df))
    d = stats_df.loc[stats_df['predr'] > stats_df['predr'].mean()]
    print(d['return'].mean(), len(d))
    d = stats_df.loc[stats_df['pred'] > stats_df['pred'].mean()]
    print(d['return'].mean(), len(d))
    #stats_df['return'] = stats_df['return']-stats_df['return'].mean()
    #stats_df.to_csv(sys.stdout, sep='\t', index=False, columns=['date', 'stock', 'pred', 'return'])


if __name__ == '__main__':
    evaluate('../../result.txt.192')
