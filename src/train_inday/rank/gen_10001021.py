
import sys
import glob
import pandas as pd
import numpy as np

from datetime import datetime, timedelta

MAX_DAILY_TRADE_CNT = 50  # 天级别交易股票的上限
TH = 0.5
TRADE_COST = 0.00125      # 交易成本
def my_rank(x):
   return pd.Series(x).rank(pct=True).iloc[-1]

trade_fn = 'tushare_day.v12'
names = ['date', 'stock', 'pred',  'return']
dtype = {'date': np.str_, 'stock': np.str_, 'pred': np.float32,  'return': np.float32}
df = pd.read_csv(trade_fn, header=None, sep=' ', names=names, index_col=False, dtype=dtype)
df['stock'] = df['stock'].str[:6]

#print(df)

today = datetime.today().strftime("%Y%m%d")
fn_pickle = './ochl.pickle.all'
ochl_df = pd.read_pickle(fn_pickle)
ochl_df = ochl_df.loc[ochl_df.date >= '20210101']
ochl_df.sort_values(by=['stock', 'date'], ascending=[True, True], inplace=True)
#print(ochl_df)

f = ochl_df.groupby(by='stock').apply(lambda e: (e['close'] / e['close'].shift() - 1).shift(-1))
f = f.reset_index()
f = f.set_index('level_1')
#print(f)
ochl_df['return1'] = f['close']

df = df.merge(ochl_df, on=['date', 'stock'], how='inner')
df['return'] = df['return1']
#print(df)

t_df = df.groupby('date').apply(lambda e:e.loc[ e['pred'] > 0].head(MAX_DAILY_TRADE_CNT)['return'].mean() - TRADE_COST)
r_df = df.groupby('date').apply(lambda e: len(e.loc[ e['pred'] > TH]) / len(e))
#r_df = df.groupby('date').apply(lambda e: e.loc[ e['pred'] < 0]['pred'].mean())
#r_df = df.groupby('date').apply(lambda e: e['pred'].mean())
r_df = df.groupby('date').apply(lambda e: e.head(MAX_DAILY_TRADE_CNT)['pred'].mean())
x_df = r_df.rolling(3).apply(my_rank)
a_df = pd.concat([t_df, r_df, x_df], axis=1)
#print(a_df)
#a_df = a_df.loc[ a_df[1] > a_df[2]]
a_df = a_df.mask(a_df[2] < 0.5, 0)
rets = (a_df[0] + 1 ).prod()

w_df =  (a_df[0] + 1 ).cumprod()

#a_df.to_csv(sys.stdout, sep=' ', columns=[ 0, 1, 2], header=False)
res = pd.concat([w_df, w_df, a_df], axis=1).dropna()
res = res.reset_index()
res.columns = ['date', 'cret', 'cret1', 'cdret', 'cpred', 'crank']
#res.to_csv(sys.stdout, sep='\t',  header=False)
df = df.merge(res, on=['date'], how='inner')
df = df.loc[df.crank > 0.6]
df.sort_values(by=['date', 'pred'], ascending=[True, False], inplace=True)

df.to_csv(sys.stdout, sep=' ', columns=['date', 'stock', 'pred', 'return1'], header=False, index=False)
