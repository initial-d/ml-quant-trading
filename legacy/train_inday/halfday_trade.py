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
    daily_df = pd.read_csv('../train_classify/tushare.data', header=None, sep='\t', names=names, index_col=False, dtype=dtype)
    #daily_df = pd.read_csv('./newest_data', header=None, sep='\t', names=names, index_col=False, dtype=dtype)
    daily_df['stock'] = daily_df['stock'].str[:6]
    daily_df['sd'] = daily_df['date'] + daily_df['stock']
    daily_df.sort_values(by=['stock', 'date'], ascending=True, inplace=True)
    daily_df['fhzs_1'] = daily_df['fhzs'].shift(-1)
    daily_df['after_open'] = daily_df['open'].shift(-1)
    daily_df['last_volume'] = daily_df['volume'].shift(1)
    daily_df['last_avg'] = daily_df['avg'].shift(1)
    #print(daily_df)


    #trade_fn = 'tushare_day.v12.cl'
    #names = ['date', 'stock', 'pred',  'return']
    #dtype = {'date': np.str_, 'stock': np.str_, 'pred': np.float32,  'return': np.float32}
    #df = pd.read_csv(trade_fn, header=None, sep=' ', names=names, index_col=False, dtype=dtype)
    #df['stock'] = df['stock'].str[:6]

    #print(df)


    #trade_fn = './day_halfday/tushare_day.86'
    #trade_fn = './tushare_day.82'
    #trade_fn = './tushare_day.999'
    #trade_fn = '../train_classify/tushare_day.26'
    #trade_fn = './tushare_day.1462'
    trade_fn = 'tushare_day'
    #trade_fn = './test/tushare_day_1130' #kl
    #trade_fn = './test/tushare_day_stoa_1130'
    #trade_fn = '../train_half/lr00001/tushare_day.315'
    #trade_fn = './tushare_day.668.new'
    names = ['stock', 'date', 'hreturn',  'hpred', 'today_close', 'halfday_close']
    dtype = {'stock': np.str_, 'date': np.str_, 'hreturn': np.float32,  'hpred': np.float32, 'today_close':np.float32, 'halfday_close':np.float32}
    half_df = pd.read_csv(trade_fn, header=None, sep='\t', names=names, index_col=False, dtype=dtype)
    half_df['date'] = half_df['date'].str.replace('-', '')
    half_df.sort_values(by=['stock', 'date'], ascending=True, inplace=True)
    #half_df['buy_date'] = half_df['date'].shift(1)
    #print(half_df)
    half_df['sd'] = half_df['date'] + half_df['stock']

    half_df = half_df.drop(columns=['date', 'stock'])
    
    


    #daily_df = daily_df.drop(columns=['date', 'stock'])

    df = half_df.join(daily_df.set_index('sd'), on='sd')
    #df = df.loc[ ( ((df.limit_up / df.close - 1) > 0.01)) & (df.avg < 300)] 
    df = df.loc[ ((df.limit_up / df.halfday_close - 1) > 0.01) & (df.last_volume * df.last_avg * 100 > 50000000) ] 
    #df = df.loc[ ((df.limit_up / df.close - 1) > 0.02) & (df.volume  * 100 > 60000000) ] 
    df = df.loc[ df.fhzs_1 != "1"] 
    #df = df.loc[ ((df.limit_up / df.close - 1) > 0.01)] 
    #df = df.loc[ (  ((df.volume * df.avg * 100) > 100000000) )] 

    #print(df)

    #df['fix_return'] = np.where(df['hpred'] > 0.00, df['return'], (df['halfday_close'] - df['close']) / df['close'] )
    
    #df['fix_return'] = (df['after_open'] - df['halfday_close']) / df['halfday_close']
    df['fix_return'] = (df['today_close'] - df['halfday_close']) / df['halfday_close']
    #df['fix_return'] = np.where((df['hpred'] > 0.00), df['return'], (df['halfday_close'] - df['close']) / df['close'] )
    #df.loc[(df['morning_return'] > 0.05), 'fix_return'] =  df.loc[(df['morning_return'] > 0.05), 'morning_return']
    #df.loc[(df['morning_return'] < -0.03), 'fix_return'] =  df.loc[(df['morning_return'] < -0.03), 'morning_return']

    #df.loc[(df['pred'] > 0.01) & (df['hpred'] > 0.01), 'fix_return'] =  df.loc[(df['pred'] > 0.01) & (df['hpred'] > 0.01), 'return']
    #df.loc[(df['morning_return'] < -0.03), 'fix_return'] = df.loc[(df['morning_return'] < -0.03), 'morning_return']



    #df.set_index(keys=['stock', 'date'], drop=False,inplace=True)
    #df['hpred'] = -df['hpred']
    #df['fix_return'] = -df['hreturn']
    df.sort_values(by=['date', 'hpred'], ascending=[True, False], inplace=True)
    df.to_csv('trade.txt', sep=' ', header=True, index=False)


if __name__ == '__main__':
    generate_trade_df()
