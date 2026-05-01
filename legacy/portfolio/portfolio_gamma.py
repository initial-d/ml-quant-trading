# -*- coding:utf-8 -*-

import numpy as np
import pandas as pd

import time
from datetime import date

import sklearn.covariance as skcov
from sksparse.cholmod import cholesky
import scipy
import mosek
import sys
import math

# x = Return[股票数量, 历史天数]

def get_gt(x: np.ndarray):
    n, n_t = x.shape
    sigma = skcov.shrunk_covariance(np.matmul(x, x.transpose()) / n_t)
    factor = cholesky(scipy.sparse.csc_matrix(sigma))
    return factor.L().todense().T.tolist()


def streamprinter(text):
    sys.stdout.write("%s" % text),

def cal(n, gamma, mu, GT, x0, w, dt, stock, pred, ret, trade_file):

    #n = 3
    #gamma = 0.05
    #mu = [0.1073, 0.0737, 0.0627]
    #GT = [[0.1667, 0.0232, 0.0013],
    #      [0.0000, 0.1033, -0.0022],
    #      [0.0000, 0.0000, 0.0338]]
    #x0 = [0.0, 0.0, 0.0]
    #w = 1.0

    inf = 0.0 # This value has no significance

    with mosek.Env() as env:
        with env.Task(0, 0) as task:
            task.set_Stream(mosek.streamtype.log, streamprinter)

            # Constraints.
            task.appendcons(1 + n)

            # Total budget constraint - set bounds l^c = u^c
            rtemp = w + sum(x0)
            task.putconbound(0, mosek.boundkey.fx, rtemp, rtemp)
            task.putconname(0, "budget")

            # The remaining constraints GT * x - t = 0 - set bounds l^c = u^c
            task.putconboundlist(range(1 + 0, 1 + n), [mosek.boundkey.fx] * n, [0.0] * n, [0.0] * n)
            for j in range(1, 1 + n):
                task.putconname(j, "GT[%d]" % j)

            # Variables.
            task.appendvars(1 + 2 * n)

            # Offset of variables into the API variable.
            offsetx = 0
            offsets = n
            offsett = n + 1

            # x variables.
            # Returns of assets in the objective 
            task.putclist(range(offsetx + 0, offsetx + n), mu)
            # Coefficients in the first row of A
            task.putaijlist([0] * n, range(offsetx + 0, offsetx + n), [1.0] * n)
            # No short-selling - x^l = 0, x^u = inf 
            #task.putvarboundslice(offsetx, offsetx + n, [mosek.boundkey.lo] * n, [0.0] * n, [inf] * n)
            task.putvarboundslice(offsetx, offsetx + n, [mosek.boundkey.ra] * n, [0.0] * n, [0.05] * n)
            for j in range(0, n):
                task.putvarname(offsetx + j, "x[%d]" % (1 + j))

            # s variable is a constant equal to gamma
            #task.putvarbound(offsets + 0, mosek.boundkey.fx, gamma, gamma)
            task.putvarbound(offsets + 0, mosek.boundkey.ra, 0.0, gamma)
            task.putvarname(offsets + 0, "s")

            # t variables (t = GT*x).
            # Copying the GT matrix in the appropriate block of A
            for j in range(0, n):
                task.putaijlist(
                    [1 + j] * n, range(offsetx + 0, offsetx + n), GT[j])
            # Diagonal -1 entries in a block of A
            task.putaijlist(range(1, n + 1), range(offsett + 0, offsett + n), [-1.0] * n)
            # Free - no bounds
            task.putvarboundslice(offsett + 0, offsett + n, [mosek.boundkey.fr] * n, [-inf] * n, [inf] * n)
            for j in range(0, n):
                task.putvarname(offsett + j, "t[%d]" % (1 + j))

            # Define the cone spanned by variables (s, t), i.e. dimension = n + 1 
            task.appendcone(mosek.conetype.quad, 0.0, [offsets] + list(range(offsett, offsett + n)))
            task.putconename(0, "stddev")

            task.putobjsense(mosek.objsense.maximize)

            # Dump the problem to a human readable OPF file.
            #task.writedata("dump.opf")

            task.optimize()

            # Display solution summary for quick inspection of results.
            task.solutionsummary(mosek.streamtype.msg)

            # Retrieve results
            xx = [0.] * (n + 1)
            task.getxxslice(mosek.soltype.itr, offsetx + 0, offsets + 1, xx)
            expret = sum(mu[j] * xx[j] for j in range(offsetx, offsetx + n))
            stddev = xx[offsets]
            
            for j in range(offsetx, offsetx + n):
                trade_file.write("%s %s %f %f %e %e %e\n" % (dt[j], stock[j], mu[j], ret[j], xx[j], expret, stddev))
                 

            print("\nExpected return %e for gamma %e\n" % (expret, stddev))
            
    sys.stdout.flush()

def generate_trade_df():

    #today = date.fromtimestamp(time.time()).strftime('%Y%m%d')
    #daily_fn = '/home/guochenglin/xproject-data/data/%s.data.ochl' % (today)
    #trade_fn = '/home/guochenglin/xproject-data/data/%s.result.txt' % (today)
    #trade_df_fn = '/home/guochenglin/xproject-data/data/%s.trade.pickle' % (today)

    names = ['date', 'stock', 'open', 'close', 'high', 'low', 'avg', 'volume', 'fhzs', 'limit_up', 'limit_down']
    dtype = {'stock': np.str_, 'date': np.str_, 'open': np.float32, 'close': np.float32, 'high': np.float32, 'low': np.float32, 'avg': np.float32, 'volume': np.float32, 'fhzs': np.str_, 'limit_up': np.float32, 'limit_down': np.float32}
    daily_df = pd.read_csv('/da1/public/duyimin/trade_predict_217/evalate/data/tushare.data', header=None, sep='\t', names=names, index_col=False, dtype=dtype)
    daily_df['stock'] = daily_df['stock'].str[:6]
    daily_df['sd'] = daily_df['date'] + daily_df['stock']
    daily_df.sort_values(by=['stock', 'date'], ascending=True, inplace=True)
    daily_df['fhzs_1'] = daily_df['fhzs'].shift(-1)
    close_df = daily_df.pivot(index='date', columns=['stock'], values='close')
    return_df = (close_df.shift(-1) - close_df) / close_df
    #daily_df['return'] = (daily_df['close'].shift(-1) - daily_df['close']) / daily_df['close']
    
    #print(daily_df)
    #print(close_df)
    print(return_df)

    trade_fn = 'trade.txt'
    names = ['date', 'stock', 'pred',  'return']
    dtype = {'date': np.str_, 'stock': np.str_, 'pred': np.float32,  'return': np.float32}
    df = pd.read_csv(trade_fn, header=None, sep=' ', names=names, index_col=False, dtype=dtype)
    df['stock'] = df['stock'].str[:6]
    df = df.loc[df['pred'] > 0.006]
    #df['pred'] = np.sqrt(df['pred'])
    
    #print(df)
    return_df.reset_index(inplace=True)
    print(return_df)
    trade_file = open("trade_with_risk.txt", "w")
    for name, group in df.groupby('date'):
        #print(name)
        #print(group)
        #print(group['stock'].tolist())
        id = return_df.loc[return_df['date'] == name].index.tolist()[0]
        sub_return_df = return_df.iloc[id - 100 : id][group['stock'].tolist()[0:100]]
        sub_return_df.replace([np.inf, -np.inf], np.nan, inplace=True)
        sub_return_df.fillna(0, inplace=True)
        #print(sub_return_df.to_numpy())
        #print(sub_return_df.to_numpy().T)
        #print(get_gt(sub_return_df.to_numpy().T))
        n = len(group['stock'].tolist()[0:100])
        #print(n)
        gamma = 0.03
        mu = group['pred'].tolist()[0:100]
        #print(len(mu))
        GT = get_gt(sub_return_df.to_numpy().T)
        #print(GT)
        x0 = [0.0] * n
        w = 1.0
        #x0 = [1.0 / n] * n
        #w = 0
        dt = group['date'].tolist()[0:100]
        stock = group['stock'].tolist()[0:100]
        pred = group['pred'].tolist()[0:100]
        ret = group['return'].tolist()[0:100]
        
        cal(n, gamma, mu, GT, x0, w, dt, stock, pred, ret, trade_file)
        
        

    #df['sd'] = df['date'] + df['stock']
    #daily_df = daily_df.drop(columns=['date', 'stock'])

    #df = df.join(daily_df.set_index('sd'), on='sd')

    #df.set_index(keys=['stock', 'date'], drop=False,inplace=True)
    df.sort_values(by=['date', 'pred'], ascending=[True, False], inplace=True)
    #df.to_csv('trade.txt', sep=' ', header=False, index=False)


if __name__ == '__main__':
    generate_trade_df()
