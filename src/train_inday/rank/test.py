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
    #daily_df = pd.read_csv('../newest_data', header=None, sep='\t', names=names, index_col=False, dtype=dtype)
    daily_df = pd.read_csv('newest_data', header=None, sep='\t', names=names, index_col=False, dtype=dtype)
    daily_df['stock'] = daily_df['stock'].str[:6]
    daily_df.sort_values(by=['stock', 'date'], ascending=True, inplace=True)
    daily_df['open_date'] = daily_df['date'].shift(2)
    daily_df['open_close'] = daily_df['close'].shift(2)
    daily_df['sd'] = daily_df['open_date'] + daily_df['stock']
    daily_df['fhzs_1'] = daily_df['fhzs'].shift(-1)
    daily_df = daily_df.loc[daily_df.date >= '20220104']
    #print(daily_df)


    trade_fn = 'sta'
    names = ['date', 'stock', 'name', 'buy_price', 'sell_price', 'acc', 'id', 'sell_order_price', 'sell_avg_price', 'sell_time', 'state']
    dtype = {'date': np.str_, 'stock': np.str_, 'name':np.str_, 'buy_price': np.float32, 'sell_price': np.float32, 'acc':np.str_, 'id':np.str_, 'sell_order_price': np.float32, 'sell_avg_price': np.float32, 'sell_time':np.str_, 'state':np.str_}
    df = pd.read_csv(trade_fn, header=None, sep=',', names=names, index_col=False, dtype=dtype)
    df['stock'] = df['stock'].str[:6]
    df['date'] = df['date'].str.replace('-', '')
    df['sell_price'] = df['sell_price'] * 1.00112
    df = df.loc[df.sell_price > 0.0]
    df = df.loc[df.buy_price > 0.0]
    df['sd'] = df['date'] + df['stock']
    df = df.drop(columns=['date', 'stock'])
    
    #df.sort_values(by=['stock', 'date'], ascending=[True, True], inplace=True)
    #df['pred_lst'] = df['pred'].shift(1)
    #df['return_lst'] = df['return'].shift(1)
    print(df)
    #df = df.loc[df.return_lst > 0]
    #df = df.loc[df.return_lst_2 < 0]

    #names = ['stock']
    #dtype = {'stock': np.str_}
    #gz2000_df = pd.read_csv('gz2000', header=None, sep='\t', names=names, index_col=False, dtype=dtype)


    #print(df)
    #names = ['datetime', 'code', 'total_mv']
    #dtype = {'datetime': np.str_, 'code': np.str_, 'total_mv': np.float32}
    #df_mv = pd.read_csv('cap.txt', header=None, sep='\t', names=names, index_col=False, dtype=dtype)
    #df_mv.code = df_mv.code.str[:6]
    #df_mv.datetime = df_mv.datetime.str.replace('-', '')
    #df_mv = df_mv.loc[(df_mv.datetime > '20210101') & (df_mv.datetime < '20220901')]
    #df_mv['sd'] = df_mv['datetime'] + df_mv['code']
    #df_mv = df_mv.drop(columns=['datetime', 'code'])



    daily_df = daily_df.drop(columns=['date', 'stock'])

    df = df.join(daily_df.set_index('sd'), on='sd', how='right')


    #df = df.join(gz2000_df.set_index('stock'), on='stock', how='right')


    #df = df.loc[ ( ((df.limit_up / df.close - 1) > 0.01)) & (df.avg < 300)] 
    #df = df.loc[ ((df.limit_up / df.close - 1) > 0.01) & (df.volume * df.avg * 100 > 30000000) & (df.volume * df.avg * 100 < 40000000)] 
    #df = df.loc[ ((df.limit_up / df.close - 1) > 0.01) & (df.volume * df.avg * 100 < 50000000) & (df.volume * df.avg * 100 > 10000000)] 
    #df = df.loc[ ((df.limit_up / df.close - 1) > 0.01) & (df.volume * df.avg * 100 < 30000000) ] 
    #df = df.loc[ (df.close - df.open) / df.open > 0.02 ]
    #df = df.loc[ ((df.limit_up / df.close - 1) > 0.01) & (df.volume < 50000)] 
    #df = df.loc[ ((df.limit_up / df.close - 1) > 0.02) & (df.volume  * 100 > 60000000) ] 
    #df = df.loc[ ((df.limit_up / df.close - 1) > 0.01)] 
    #df = df.loc[ (  ((df.volume * df.avg * 100) > 100000000) )] 
    #df = df.loc[(df.close - df.low) / (df.high - df.close) > 1]

    #trade_fn = 'd2.stk.trade'
    trade_fn = 'stk.order.d2.100'
    names = ['date', 'stock', 'ordpoint', 'ret']
    dtype = {'date': np.str_, 'stock': np.str_, 'ordpoint': np.float32, 'ret':np.float32}
    total_df = pd.read_csv(trade_fn, header=None, sep=' ', names=names, index_col=False, dtype=dtype)
    total_df['sd'] = total_df['date'] + total_df['stock']
    #total_df = total_df.drop(columns=['date', 'stock'])
    total_df = total_df.join(df.set_index('sd'), on='sd', how='left')

    #df = df.join(df_mv.set_index('sd'), on='sd')
    #df = df.loc[df.total_mv < 200]
    total_df = total_df.fillna(0)
    total_df.sort_values(by=['stock', 'date'], ascending=True, inplace=True)
    res = []
    for k, g_df in total_df.groupby(['date']):
        g_df['buy_rate'] = float(len(g_df.loc[g_df.buy_price != 0.0])) / float(len(g_df))
        g_df['ord_cnt'] = len(g_df.loc[g_df.ret>=0])
        g_df['buy_cnt'] = len(g_df.loc[(g_df.buy_price>0.0) & (g_df.sell_price >0.0)])
        res.append(g_df)
    total_df = pd.concat(res, axis=0)

    #print(df)
    #df.set_index(keys=['stock', 'date'], drop=False,inplace=True)
    #df['return'] = -df['return']
    #df.sort_values(by=['date', 'pred'], ascending=[True, False], inplace=True)
    total_df = total_df[['date', 'stock', 'buy_price', 'sell_price', 'open', 'low', 'close', 'ordpoint', 'buy_rate', 'ord_cnt', 'buy_cnt', 'sell_order_price', 'sell_avg_price', 'sell_time', 'state', 'open_close']]
    print('############统计##############')
    total_df.to_csv('test.txt', sep=' ', header=False, index=False)


if __name__ == '__main__':
    generate_trade_df()
