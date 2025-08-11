import torch.utils.data as data
import json
import numpy as np
import pandas as pd
import torch
import time
import torch.nn.functional as F
import os
from torch.utils.data.dataloader import default_collate
import base64
import pickle
import time
import configparser
from sklearn.utils import shuffle

def collate_fn_filter(batch):
    batch = list(filter(lambda x: len(x) == 9, batch))
    if len(batch) == 0:
        return torch.Tensor()
    return default_collate(batch)


class StockData_mini_batch_day_inday_tensor(data.Dataset):
    def __init__(self, fn_day, fn_halfday, begindate, enddate, flag, batch_size):

        if flag == 'init':
            cf = configparser.ConfigParser()
            cf.read('fea.ini')
            feaitems = cf.items('allfeatures')
            tmpfeature = dict((i[0],i[1]) for i in feaitems)
            fealist = tmpfeature['fealist']
            fealist = fealist.split(',')
            print("fealist训练特征如下：", fealist)
            self.col = fealist.copy()
            fp_day = open(fn_day, 'rb')
            df_day = pickle.load(fp_day)
            df_day = df_day.drop(columns=['date', 'stock'])


            print(fn_halfday) 
            fp_halfday = open(fn_halfday, 'rb')
            df_halfday = pd.read_pickle(fn_halfday)
            df_halfday = df_halfday.drop(columns=['date', 'stock'])
            
            #for all
            #df_halfday = df_halfday.loc[df_halfday['TRADE_DT'].str.contains('11:30:00', na=False) == True] 
            #df_halfday['TRADE_DT'] = df_halfday['TRADE_DT'].str[:8]
    
            df_day.sort_values(by=['S_INFO_WINDCODE', 'TRADE_DT'], ascending=True, inplace=True)
            print(df_day[['S_INFO_WINDCODE', 'TRADE_DT', 'S_FWDS_ADJCLOSE']])
            df_day['today_close'] = df_day['S_FWDS_ADJCLOSE'].shift(-1)
            df_day['TRADE_DT'] = df_day['TRADE_DT'].shift(-1)
            
            print(df_day[['S_INFO_WINDCODE', 'TRADE_DT', 'S_FWDS_ADJCLOSE', 'today_close']])
    
            df_day['sd'] = df_day['S_INFO_WINDCODE'] + df_day['TRADE_DT']
            df_halfday['sd'] = df_halfday['S_INFO_WINDCODE'] + df_halfday['TRADE_DT']

            
            #df_day = df_day.loc[ ((df_day.LIMIT_UP / df_day.S_FWDS_ADJCLOSE - 1) > 0.01) & (df_day.S_DQ_VOLUME * df_day.S_DQ_AVGPRICE * 100 > 50000000) ]
            
            df = df_day.join(df_halfday.set_index('sd'), on='sd', lsuffix='_day', rsuffix='_halfday')
    
    
            #df.dropna(subset=['date', 'stocks'],inplace=True)
            self.df = df.loc[ (df['TRADE_DT_day'] >= begindate) & (df['TRADE_DT_day'] <= enddate) ]
            self.df.sort_values(by=['S_INFO_WINDCODE_day', 'TRADE_DT_day'], ascending=True, inplace=True)
            self.df['target_01'] = (self.df.today_close - self.df.S_FWDS_ADJCLOSE_halfday) / self.df.S_FWDS_ADJCLOSE_halfday
            
            print(self.df[['S_INFO_WINDCODE_day', 'TRADE_DT_day', 'S_FWDS_ADJCLOSE_day', 'S_FWDS_ADJCLOSE_halfday', 'today_close', 'target_01']])
            print(self.df)
            #print(self.df['target_01'])
            #print(self.df['S_FWDS_ADJOPEN'])
            #print(self.df['S_FWDS_ADJCLOSE'])
            self.df.dropna(axis=0, subset=['target_01'], inplace=True)
            self.df = self.df.dropna(thresh=2)
            self.df.fillna(0, inplace=True)
    
            #self.df = self.df.loc[ ((self.df.LIMIT_UP / self.df.S_FWDS_ADJCLOSE - 1) > 0.01) & (self.df.S_DQ_VOLUME * self.df.S_DQ_AVGPRICE * 100 > 50000000) ]	
            #self.df = self.df.loc[ ((self.df.LIMIT_DOWN / self.df.S_FWDS_ADJCLOSE - 1) < -0.02)]	
            #self.df.target_01 = self.df.target_01.where(self.df.target_01 <= 0.08, 0.08)
            print(self.df)

            cf = configparser.ConfigParser()
            cf.read('fea_day_halfday.ini')
            feaitems = cf.items('allfeatures')
            tmpfeature = dict((i[0],i[1]) for i in feaitems)
            fealist = tmpfeature['fealist']
            fealist = fealist.split(',')
            print("fealist训练特征如下：", fealist)
            self.col = fealist.copy()
    
            #self.df.to_pickle("Ashares2train_tushare_day_halfday_all.pickle")
            return

    def gen_tensor(self, begindate, enddate, flag, batch_size, i):
        #day_halfday_data = open('Ashares2train_tushare_day_halfday_del8.pickle', 'rb')
        #self.df = pickle.load(day_halfday_data) 
        dff = self.df.copy(deep=True)
        dff = dff.loc[ (dff['TRADE_DT_day'] >= begindate) & (dff['TRADE_DT_day'] <= enddate) ]
        dff.replace([np.inf, -np.inf], np.nan, inplace=True)
        dff.dropna(axis=0, subset=['target_01'], inplace=True)
    
        print(dff['target_01'])
    
        dff = dff.dropna(thresh=2)
        dff.fillna(0, inplace=True)
        print(dff)
    
        self.f_len = len(dff)
        self.k = 0
        self.Batch_size = batch_size
    
        if int(self.f_len) % int(self.Batch_size) == 0:
            self.n = int(self.f_len / self.Batch_size)
        else:
            self.n = int(self.f_len / self.Batch_size) + 1
        batch_label = np.zeros((self.f_len, 1), dtype=np.float32)
        batch_data = dff
        featurearray = batch_data[self.col].to_numpy()
    
        self.date = batch_data['TRADE_DT_day'].to_numpy()
        self.stocks = batch_data['S_INFO_WINDCODE_day'].to_numpy()
        self.today_close = batch_data['today_close'].to_numpy()
        self.halfday_close = batch_data['S_FWDS_ADJCLOSE_halfday'].to_numpy()
    
        batch_data = featurearray[:, 1:]
        batch_data[batch_data > 10] = 0.0
        batch_data[batch_data < -10] = 0.0
        batch_label[:, 0] = np.log((featurearray[:, 0] + 1.0).astype('float'))
        self.x = torch.from_numpy(batch_data.astype('float')).to(torch.float32)
        self.y = torch.from_numpy(batch_label).to(torch.float32)
        tensor_data = {}
        tensor_data['x'] = self.x
        tensor_data['y'] = self.y
    
        tensor_data['date'] = self.date
        tensor_data['stocks'] = self.stocks
        tensor_data['today_close'] = self.today_close
        tensor_data['halfday_close'] = self.halfday_close
        fp = open("tensor_data_inday_" + flag + '_' + str(i) + ".pickle", 'wb')
        pickle.dump(tensor_data, fp,  protocol=4)
    
        self.f_len = len(self.y)
        self.idx = 0
       
        
    def __len__(self):
        return self.f_len

    def __getitem__(self, idx):
        return self.x[i], self.y[i]

    def get_batch_data(self, batch_size=-1):
        if self.idx == 0:
            indices = torch.randperm(self.f_len) 
            self.x = self.x[indices]
            self.y = self.y[indices]
        s = self.idx * batch_size
        e = s + batch_size
        #print(s,e)
        self.idx = (self.idx + 1) % ( self.f_len // batch_size ) 
        return self.x[s:e], self.y[s:e]


    def get_predict_batch_data(self, batch_size=-1):
        s = self.idx * batch_size
        e = s + batch_size
        #print(s,e)
        self.idx = (self.idx + 1) % ( self.f_len // batch_size ) 
        return self.x[s:e], self.y[s:e], self.date[s:e], self.stocks[s:e], self.today_close[s:e], self.halfday_close[s:e]
        

if __name__ == '__main__' :
    # 测试代码
    # import time
    # data_path = '/home/wanghexiang/dnn_rank/'
    #dataset = StockData_mini_batch_day_inday_tensor('Ashares2train_tushare.pickle', 'Ashares2train_tushare_half_new.pickle', '20110110', '20191231', 'train', 128)
    #testdataset = StockData_mini_batch_day_inday_tensor('Ashares2train_tushare.pickle', 'Ashares2train_tushare_half_new.pickle', '20200101', '20210830', 'test', 128)
    #dataset = StockData_mini_batch_day_rel_tensor('Ashares2train_tushare.pickle', 'Ashares2train_tushare_half_new.pickle', '20110101', '20210831', 'init', 128)
    #dataset = StockData_mini_batch_day_inday_tensor('Ashares2train_tushare.pickle', 'Ashares2train_tushare_half_new.pickle', '20110110', '20210830', 'init', 128)
    for i in range(7, 15):
        fn_day = '/da1/public/duyimin/train_del/Ashares2train_tushare_del8.pickle'
        fn_halfday = '/da1/public/duyimin/data/Ashares2train_tushare_inday_15.pickle_' + str(i)
        dataset = StockData_mini_batch_day_inday_tensor(fn_day, fn_halfday, '20110101', '20210831', 'init', 128)
        dataset.gen_tensor('20110110', '20201231', 'train', 128, i)
        dataset.gen_tensor('20210101', '20210830', 'test', 128, i)
    # t2 = 0.0

    # for i in range(5000):
    #     start = time.time()
    #     data = dataset.get_batch_data(128)
    #     end = time.time()
    #     t2 += end -start
    #     print(end - start)
    pass
