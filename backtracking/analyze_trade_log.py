# -*- utf-8 -*-


import sys
import pandas as pd
import numpy as np
from datetime import date, timedelta


log_fn = 'log.tmp'


names = ['day', 'stock', 'ff', 'NO', 'order_type', 'buy_or_sell', 'o_price', 's_amount', 's_deal_amount', 's_deal_chn', 's_date', 'account', 'strategy',
         'wz', 's_price', 'cost', 'status', 'f1', 'f2', 'f3', 'f4', 'f5', 'f6', 'f7', 'f8', 'f9', 'b_price', 'b_amount', 'b_deal_amount', 'b_date', 'profit']
dtype = {'day': np.str_, 'stock': np.str_, 'account': np.int32}

cols = ['day', 'stock', 'account', 'b_amount', 'b_price', 's_amount', 's_price', 'profit', 'cost', 'b_deal_amount', 's_deal_amount', 'b_date']
df = pd.read_csv(log_fn, sep=',', names=names, header=None, index_col=False, dtype=dtype, usecols=cols)

print(df)

day = '2019-07-08'
f = map(int, day.split('-'))
d = date(*f) - timedelta(days=1)

df = df.loc[df.account == 1000048]
df = df.loc[df.day == day]

df.to_csv(sys.stdout, sep='\t', columns=cols, index=False)

r_return = (df.b_amount * df.s_price).sum() / (df.b_amount * df.b_price).sum() - 1
print(df.profit.sum())
profit = df.profit.sum()

day = df.b_date.iloc[-1].split(' ')[0].replace('-', '')
print(day)
trade_df = pd.read_pickle('/da1/public/guochenglin/trade/trade.df')
trade_df = trade_df.loc[trade_df.date == day]

trade_df.to_csv(sys.stdout, sep='\t', index=False)

p_return = trade_df['pred'].mean()
print((df.b_amount * df.s_price).sum(), profit, profit/(df.b_amount * df.s_price).sum(), r_return, p_return)
