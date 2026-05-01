# -*- coding:utf-8 -*-


import pandas as pd
import numpy as np
from conf import config_all as conf
from Feature import Func as func
from multiprocessing import Pool
import datetime


def condlimit(CLO):
    #return abs(CLO/CLO.shift() - 1) > 0
    return abs(CLO/CLO.shift() - 1) <= 0.098

def calc_feature(df, stock, cols):
    print(stock)


    df['datetime'] = pd.to_datetime(df['TRADE_DT'])
    #print(df)
    df['day_gap'] = df['datetime'] - df['datetime'].shift(1)
    df['day_gap'] = df['day_gap'].map(lambda x:x.days)


   
    split_date = df.loc[df['day_gap'] > 10]['TRADE_DT']
    #split_gap = df.loc[df['day_gap'] > 10]['day_gap']
    #print(split_gap)
    #print(split_date.tolist())
    split_df_list = []
    res_df = None
    if(len(split_date) > 0):
        for i in range(len(split_date) + 1):
            split_df = None
            scols = cols.copy()
            if i == 0:
                split_df = df.loc[df['TRADE_DT'] < split_date[i]]
            elif i > 0 and i < len(split_date):
                split_df = df.loc[(df['TRADE_DT'] >= split_date[i - 1]) & (df['TRADE_DT'] < split_date[i])]
            elif i == len(split_date):
                split_df = df.loc[df['TRADE_DT'] >= split_date[i - 1]]
            fea = []
            for f in scols:
                tmp = eval('func.' + f + '(split_df)')
                cond = condlimit(split_df.S_FWDS_ADJCLOSE)
                #print(f)
                #print(cond)
                #print(tmp[cond])
                fea.append(tmp[cond])

            scols.append('date')
            scols.append('stocks')
            scols.append('rank')

            #split_df['rank-1'] = split_df['rank_1'].shift(-1)

            fea.append(split_df['TRADE_DT'])
            fea.append(split_df['S_INFO_WINDCODE'])
            fea.append(split_df['rank'])

            fea_df =  pd.concat(fea, axis=1, keys=scols)
            split_df_list.append(fea_df)
            #print(fea_df)

        res_df = pd.concat(split_df_list, axis=0)


    #print(res_df)
    else:
        fea = []
        for f in cols:
            tmp = eval('func.' + f + '(df)')
            cond = condlimit(df.S_FWDS_ADJCLOSE)
            fea.append(tmp[cond])

        #df['rank-1'] = df['rank_1'].shift(-1)

        fea.append(df['TRADE_DT'])
        fea.append(df['S_INFO_WINDCODE'])
        fea.append(df['rank'])
    
        cols.append('date')
        cols.append('stocks')
        cols.append('rank')

        res_df =  pd.concat(fea, axis=1, keys=cols)

    return res_df


class OCHLData(object):

    def __init__(self, ochl_fn, bin_fn):
        self.load_ochl(ochl_fn)
        self.get_feature()

    def date_rank(self, x):
        x['rank'] = x['S_DQ_AMOUNT'].rank(ascending=False, pct=True)
        x['rank_1'] = x['lift_inday'].rank(ascending=False, pct=True)
        #print(x)
        return x

    def load_ochl(self, ochl_fn):
        #names = ['TRADE_DT', 'S_INFO_WINDCODE', 'S_FWDS_ADJOPEN', 'S_FWDS_ADJCLOSE', 'S_FWDS_ADJHIGH', 'S_FWDS_ADJLOW', 'S_DQ_AVGPRICE', 'S_DQ_VOLUME',  'FHZS_FLAG', 'LIMIT_UP', 'LIMIT_DOWN']
        #dtype = {'TRADE_DT': np.str_, 'S_INFO_WINDCODE': np.str_, 'S_FWDS_ADJOPEN': np.float32, 'S_FWDS_ADJCLOSE': np.float32, 'S_FWDS_ADJHIGH': np.float32,
        #         'S_FWDS_ADJLOW': np.float32, 'S_DQ_AVGPRICE': np.float32, 'S_DQ_VOLUME': np.float32, 'FHZS_FLAG': np.str_, 'LIMIT_UP': np.float32, 'LIMIT_DOWN': np.float32}

        names = ['TRADE_DT', 'S_INFO_WINDCODE', 'S_FWDS_ADJOPEN', 'S_FWDS_ADJCLOSE', 'S_FWDS_ADJHIGH', 'S_FWDS_ADJLOW', 'S_DQ_AVGPRICE', 'S_DQ_VOLUME']
        dtype = {'TRADE_DT': np.str_, 'S_INFO_WINDCODE': np.str_, 'S_FWDS_ADJOPEN': np.float32, 'S_FWDS_ADJCLOSE': np.float32, 'S_FWDS_ADJHIGH': np.float32,
                 'S_FWDS_ADJLOW': np.float32, 'S_DQ_AVGPRICE': np.float32, 'S_DQ_VOLUME': np.float32}

        df = pd.read_csv(ochl_fn, header=None, sep='\t', names=names, index_col=False, dtype=dtype)

        df.S_INFO_WINDCODE = df.S_INFO_WINDCODE.str[:6]
        df['S_DQ_AMOUNT'] = df.S_DQ_VOLUME * df.S_DQ_AVGPRICE

        df['lift_inday'] = (df.S_FWDS_ADJCLOSE - df.S_FWDS_ADJOPEN) / df.S_FWDS_ADJOPEN
        df = df.groupby(['TRADE_DT']).apply(self.date_rank)
        #print(df.groupby(['TRADE_DT']))
        #df.sort_values(by=['TRADE_DT'], ascending=True, inplace=True)
        #print(df)


        df.sort_values(by=['S_INFO_WINDCODE', 'TRADE_DT'], ascending=True, inplace=True)
        df.set_index(keys=['S_INFO_WINDCODE', 'TRADE_DT'], drop=False, inplace=True)

        self.stocks = df['S_INFO_WINDCODE'].unique().tolist()
        self.df = df
        self.fealist = []
        with open('allfea.ini') as file:
            lines = file.readlines()
            self.fealist = [line.rstrip() for line in lines]

    def get_feature(self):

        p = Pool(30)
        results = [p.apply_async(calc_feature, args=(self.df.loc[self.df.S_INFO_WINDCODE == stock], stock, self.fealist)) for stock in self.stocks]
        p.close()
        p.join()
        res = [r.get() for r in results]
        df = pd.concat(res, axis=0)
        print(df)
        df.to_pickle(conf.FEATURE_DF_FN)

    def get_feature_test(self):
        res = []
        for stock in self.stocks[:10]:
            print(stock)
            fea = []
            df = self.df.loc[self.df.S_INFO_WINDCODE == stock]
            for f in cols:
                data = eval('func.'+f + '(df)')
                fea.append(data)
            stock_df = pd.concat(self.fealist, axis=1, keys=cols)
            res.append(stock_df)
        stock_df = pd.concat(res, axis=0)
        print(stock_df)


if __name__ == '__main__':
    data = OCHLData(conf.DATA_FN, conf.FEATURE_DF_FN)
    #data = OCHLData('20_21', "Ashares2train_2020_2021.pickle")

