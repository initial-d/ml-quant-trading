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
from sklearn.preprocessing import normalize

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
    #alphas = [0.0, 0.25, 0.5, 0.75, 1.0, 1.5, 2.0, 3.0, 4.0, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20]
    alphas = [0.0, 0.1, 0.5, 1.0, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100, 110, 120, 130, 140, 150, 160, 170, 180, 190, 200]
    #alphas = [1.0]
    #alphas = list(map(lambda x:x/10, list(range(-50,50))))

    inf = 0.0 # This value has no significance

    with mosek.Env() as env:
        with env.Task(0, 0) as task:
            task.set_Stream(mosek.streamtype.log, streamprinter)

            rtemp = w + sum(x0)

            # Constraints.
            task.appendcons(1 + n)
            task.putconbound(0, mosek.boundkey.fx, rtemp, rtemp)
            task.putconname(0, "budget")

            task.putconboundlist(range(1 + 0, 1 + n), n *
                                 [mosek.boundkey.fx], n * [0.0], n * [0.0])
            for j in range(1, 1 + n):
                task.putconname(j, "GT[%d]" % j)

            # Variables.
            task.appendvars(2 + 2 * n)

            offsetx = 0   # Offset of variable x into the API variable.
            offsets = n   # Offset of variable s into the API variable.
            offsett = n + 1 # Offset of variable t into the API variable.
            offsetu = 2*n + 1 # Offset of variable u into the API variable.

            # x variables.
            task.putclist(range(offsetx + 0, offsetx + n), mu)
            task.putaijlist(
                n * [0], range(offsetx + 0, offsetx + n), n * [1.0])
            for j in range(0, n):
                task.putaijlist(
                    n * [1 + j], range(offsetx + 0, offsetx + n), GT[j])

            task.putvarboundsliceconst(offsetx, offsetx + n, mosek.boundkey.lo, 0.0, inf)
            #task.putvarboundsliceconst(offsetx, offsetx + n, mosek.boundkey.ra, 0.0, 0.05)

            for j in range(0, n):
                task.putvarname(offsetx + j, "x[%d]" % (1 + j))

            # s variable.
            task.putvarbound(offsets + 0, mosek.boundkey.fr, -inf, inf)
            task.putvarname(offsets + 0, "s")

            # u variable.
            task.putvarbound(offsetu + 0, mosek.boundkey.fx, 0.5, 0.5)
            task.putvarname(offsetu + 0, "u")

            # t variables.
            task.putaijlist(range(1, n + 1), range(offsett +
                                                   0, offsett + n), n * [-1.0])
            task.putvarboundsliceconst(offsett, offsett + n, mosek.boundkey.fr, -inf, inf)
            for j in range(0, n):
                task.putvarname(offsett + j, "t[%d]" % (1 + j))

            task.appendcone(mosek.conetype.rquad, 0.0, 
                            [offsets, offsetu] + list(range(offsett, offsett + n)))
            task.putconename(0, "variance")

            task.putobjsense(mosek.objsense.maximize)

            # Turn all log output off.
            task.putintparam(mosek.iparam.log, 0)

            for alpha in alphas:
                # Dump the problem to a human readable OPF file.
                #task.writedata("dump.opf")

                task.putcj(offsets + 0, -alpha)

                task.optimize()

                # Display the solution summary for quick inspection of results.
                # task.solutionsummary(mosek.streamtype.msg)

                solsta = task.getsolsta(mosek.soltype.itr)

                if solsta in [mosek.solsta.optimal]:
                    expret = 0.0
                    x = [0.] * n
                    task.getxxslice(mosek.soltype.itr,
                                    offsetx + 0, offsetx + n, x)
                    for j in range(0, n):
                        expret += mu[j] * x[j]

                    stddev = [0.]
                    task.getxxslice(mosek.soltype.itr,
                                    offsets + 0, offsets + 1, stddev)

                    for j in range(0, n):
                        trade_file.write("%s %s %f %f %e %e %e %f\n" % (dt[j], stock[j], mu[j], ret[j], x[j], expret, stddev[0], alpha))


                    print("alpha = {0:.2e} exp. ret. = {1:.3e}, variance {2:.3e}".format(alpha, expret, stddev[0]))
                else:
                    print("An error occurred when solving for alpha=%e\n" % alpha)




    #with mosek.Env() as env:
    #    with env.Task(0, 0) as task:
    #        task.set_Stream(mosek.streamtype.log, streamprinter)

    #        # Constraints.
    #        task.appendcons(1 + n)

    #        # Total budget constraint - set bounds l^c = u^c
    #        rtemp = w + sum(x0)
    #        task.putconbound(0, mosek.boundkey.fx, rtemp, rtemp)
    #        task.putconname(0, "budget")

    #        # The remaining constraints GT * x - t = 0 - set bounds l^c = u^c
    #        task.putconboundlist(range(1 + 0, 1 + n), [mosek.boundkey.fx] * n, [0.0] * n, [0.0] * n)
    #        for j in range(1, 1 + n):
    #            task.putconname(j, "GT[%d]" % j)

    #        # Variables.
    #        task.appendvars(2 + 2 * n)

    #        # Offset of variables into the API variable.
    #        offsetx = 0
    #        offsets = n
    #        offsett = n + 1
    #        offsetu = 2 * n + 1

    #        # x variables.
    #        # Returns of assets in the objective 
    #        task.putclist(range(offsetx + 0, offsetx + n), mu)
    #        # Coefficients in the first row of A
    #        task.putaijlist([0] * n, range(offsetx + 0, offsetx + n), [1.0] * n)
    #        # No short-selling - x^l = 0, x^u = inf 
    #        #task.putvarboundslice(offsetx, offsetx + n, [mosek.boundkey.lo] * n, [0.0] * n, [inf] * n)
    #        task.putvarboundslice(offsetx, offsetx + n, [mosek.boundkey.ra] * n, [0.0] * n, [0.05] * n)
    #        for j in range(0, n):
    #            task.putvarname(offsetx + j, "x[%d]" % (1 + j))

    #        # s variable is a constant equal to gamma
    #        #task.putvarbound(offsets + 0, mosek.boundkey.fx, gamma, gamma)
    #        task.putvarbound(offsets + 0, mosek.boundkey.ra, 0.0, gamma)
    #        task.putvarname(offsets + 0, "s")

    #        # t variables (t = GT*x).
    #        # Copying the GT matrix in the appropriate block of A
    #        for j in range(0, n):
    #            task.putaijlist(
    #                [1 + j] * n, range(offsetx + 0, offsetx + n), GT[j])
    #        # Diagonal -1 entries in a block of A
    #        task.putaijlist(range(1, n + 1), range(offsett + 0, offsett + n), [-1.0] * n)
    #        # Free - no bounds
    #        task.putvarboundslice(offsett + 0, offsett + n, [mosek.boundkey.fr] * n, [-inf] * n, [inf] * n)
    #        for j in range(0, n):
    #            task.putvarname(offsett + j, "t[%d]" % (1 + j))

    #        # Define the cone spanned by variables (s, t), i.e. dimension = n + 1 
    #        task.appendcone(mosek.conetype.quad, 0.0, [offsets] + list(range(offsett, offsett + n)))
    #        task.putconename(0, "stddev")

    #        task.putobjsense(mosek.objsense.maximize)

    #        # Dump the problem to a human readable OPF file.
    #        #task.writedata("dump.opf")

    #        task.optimize()

    #        # Display solution summary for quick inspection of results.
    #        task.solutionsummary(mosek.streamtype.msg)

    #        # Retrieve results
    #        xx = [0.] * (n + 1)
    #        task.getxxslice(mosek.soltype.itr, offsetx + 0, offsets + 1, xx)
    #        expret = sum(mu[j] * xx[j] for j in range(offsetx, offsetx + n))
    #        stddev = xx[offsets]
    #        
    #        for j in range(offsetx, offsetx + n):
    #            trade_file.write("%s %s %f %f %e %e %e\n" % (dt[j], stock[j], mu[j], ret[j], xx[j], expret, stddev))
    #             

    #        print("\nExpected return %e for gamma %e\n" % (expret, stddev))
    #        
    #sys.stdout.flush()

def generate_trade_df():

    #today = date.fromtimestamp(time.time()).strftime('%Y%m%d')
    #daily_fn = '/home/guochenglin/xproject-data/data/%s.data.ochl' % (today)
    #trade_fn = '/home/guochenglin/xproject-data/data/%s.result.txt' % (today)
    #trade_df_fn = '/home/guochenglin/xproject-data/data/%s.trade.pickle' % (today)

    #names = ['date', 'stock', 'open', 'close', 'high', 'low', 'avg', 'volume', 'fhzs', 'limit_up', 'limit_down']
    #dtype = {'stock': np.str_, 'date': np.str_, 'open': np.float32, 'close': np.float32, 'high': np.float32, 'low': np.float32, 'avg': np.float32, 'volume': np.float32, 'fhzs': np.float32, 'limit_up': np.float32, 'limit_down': np.float32}
    #daily_df = pd.read_csv('tushare_data_newest', header=None, sep='\t', names=names, index_col=False, dtype=dtype)
    #daily_df['stock'] = daily_df['stock'].str[:6]
    #daily_df['sd'] = daily_df['date'] + daily_df['stock']
    #daily_df.sort_values(by=['stock', 'date'], ascending=True, inplace=True)
    #daily_df['fhzs_1'] = daily_df['fhzs'].shift(-1)
    #mask_df = daily_df.pivot(index='date', columns=['stock'], values='fhzs_1')
    #close_df = daily_df.pivot(index='date', columns=['stock'], values='close')
    #return_df = (close_df.shift(-1) - close_df) / close_df
    ##daily_df['return'] = (daily_df['close'].shift(-1) - daily_df['close']) / daily_df['close']
    #mask_df.fillna(0, inplace=True)
    #print(mask_df)
    #
    #return_df = return_df * (1 - mask_df)
    #
    ##print(daily_df)
    ##print(close_df)
    #print(return_df)


    names = ['stock1', 'stock2', 'rel', 'cnt']
    dtype = {'stock1': np.str_, 'stock2': np.str_, 'rel': np.float32, 'cnt': np.float32}
    rel_df = pd.read_csv('result.txt.pair', header=None, sep=' ', names=names, index_col=False, dtype=dtype)
    #rel_df.drop(rel_df[rel_df.cnt < 100].index, inplace=True)
    rel_df.loc[rel_df.cnt < 100, 'rel'] = -1.0
    matrix_df = rel_df.pivot(index='stock1', columns=['stock2'], values='rel')



    tmp_df = matrix_df[['000001']]
    tmp_df.reset_index(inplace=True)
    tmp_df['stock'] = tmp_df['stock1']


    #trade_fn = 'trade_with_risk_16_17_alpha5_xub0.1.txt'
    #trade_fn = 'trade_with_risk_18_19_alpha5_xub0.1.txt'
    trade_fn = 'trade_with_risk_20_21_alpha200_xub0.05.txt'
    #trade_fn = 'trade_vir.txt'
    names = ['date', 'stock', 'pred',  'return', 'weight', 'expret', 'std', 'alpha']
    dtype = {'date': np.str_, 'stock': np.str_, 'pred': np.float32,  'return': np.float32, 'weight': np.float32, 'expret': np.float32, 'std':np.float32, 'alpha': np.float32}
    df = pd.read_csv(trade_fn, header=None, sep=' ', names=names, index_col=False, dtype=dtype)
    df['stock'] = df['stock'].str[:6]
    df = df.loc[df['pred'] > 0.008]
    #df = df.loc[df['alpha'] == 0.1]
    df = df.loc[abs(df['alpha']-20) < 1e-8]
    #df['pred'] = np.sqrt(df['pred'])
    df = pd.merge(df, tmp_df, how='right', on='stock')
    df.drop(columns=['stock1', '000001'], inplace=True)
    df.sort_values(by=['date', 'pred'], ascending=[True, False], inplace=True)
    
    

    #return_df.reset_index(inplace=True)
    #print(return_df)
    trade_file = "trade_with_risk.txt"
    for name, group in df.groupby('date'):
        #print(name)
        #print(group['stock'].tolist()[0:200])
        #ids = return_df.loc[return_df['date'] == name].index.tolist()
        #id = 0
        #if(len(ids) > 0):
        #    id = ids[0]
        #else:
        #    print(name)
        #    continue
        #sub_return_df = return_df.iloc[id - 500 : id][group['stock'].tolist()[0:200]]
        #sub_return_df.replace([np.inf, -np.inf], np.nan, inplace=True)
        #sub_return_df.fillna(0, inplace=True)
        #sub_return_df = normalize(sub_return_df, norm = 'l2', axis=0)
        #print(sub_return_df.to_numpy())
        #print(sub_return_df.to_numpy().T)
        #print(get_gt(sub_return_df.to_numpy().T))
        sub_df = matrix_df.loc[matrix_df.index.isin(group['stock'].tolist()[0:200])][group['stock'].tolist()[0:200]]
        wl = group.loc[group['weight'] > 0.01]['stock'].tolist()
        group.set_index(keys=['stock'], drop=False, inplace=True)
        print(group)
        for l in wl:
            sl = sub_df.loc[l].sort_values(ascending = False,inplace = False).index.tolist()
            if(len(sl) > 1):
                for s in sl:
                    if (s != l) and (s not in wl):
                        if sub_df.loc[l, s] < 0.5:
                            break
                        if abs(group.loc[l, 'pred'] - group.loc[s, 'pred']) > 0.03:
                            continue
                        val = group.loc[l, 'weight'] * (1 - sub_df.loc[l, s])
                        group.loc[s, 'weight'] += val
                        group.loc[l, 'weight'] -= val
                        print(l,s)
                        print(group.loc[s, 'weight'])
                        print(group.loc[l, 'weight'])
                        break
        print(group) 
        group.to_csv(trade_file, mode='a', sep=' ', header=False, index=False)
        
        
        

    #df['sd'] = df['date'] + df['stock']
    #daily_df = daily_df.drop(columns=['date', 'stock'])

    #df = df.join(daily_df.set_index('sd'), on='sd')

    #df.set_index(keys=['stock', 'date'], drop=False,inplace=True)
    df.sort_values(by=['date', 'pred'], ascending=[True, False], inplace=True)
    #df.to_csv('trade.txt', sep=' ', header=False, index=False)


if __name__ == '__main__':
    generate_trade_df()
