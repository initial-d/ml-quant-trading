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


class StockData_mini_batch_tensor(data.Dataset):
    def __init__(self, fn, begindate, enddate, flag, batch_size):
        cf = configparser.ConfigParser()
        cf.read('fea.ini')
        feaitems = cf.items('allfeatures')
        tmpfeature = dict((i[0],i[1]) for i in feaitems)
        fealist = tmpfeature['fealist']
        fealist = fealist.split(',')
        print("fealist训练特征如下：", fealist)
        self.col = fealist.copy()


        fp = open(fn, 'rb')
        df = pickle.load(fp)
        #fp = open(fn, 'rb')
        #df = pd.read_pickle(fn)
        #df.dropna(subset=['date', 'stocks'],inplace=True)
        self.df = df.loc[ (df['TRADE_DT'] >= begindate) & (df['TRADE_DT'] <= enddate) ]
        self.df.sort_values(by=['S_INFO_WINDCODE', 'TRADE_DT'], ascending=True, inplace=True)
        self.df['target_01'] = self.df['target_01'].shift(-1)
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
        
        if flag == 'train':
            self.df = shuffle(self.df)
            #self.df.sample(frac=1).reset_index(drop=True)
        #featurearray = train_df[self.col].to_numpy()
        self.f_len = len(self.df)
        self.k = 0
        self.Batch_size = batch_size

        if int(self.f_len) % int(self.Batch_size) == 0:
            self.n = int(self.f_len / self.Batch_size)
        else:
            self.n = int(self.f_len / self.Batch_size) + 1
        
    def __len__(self):
        return self.f_len

    def __getitem__(self, idx):
        return self.df[idx], self.df[idx]

    def get_batch_data(self, batch_size=-1):
        #indices = torch.randperm(len(self.x))[:batch_size] 
        #return self.x[indices], self.y[indices]
        batch_data = np.zeros((batch_size, 197), dtype=np.float32)
        batch_label = np.zeros((batch_size, 1), dtype=np.float32)
        #for i in range(n):
        i = self.k * self.Batch_size
        self.k = (self.k + 1) % self.n
        batch_data = self.df.iloc[i:i+batch_size, :]
        featurearray = batch_data[self.col].to_numpy()
        
        batch_data = featurearray[:, 1:]
        batch_data[batch_data > 10] = 0.0
        batch_data[batch_data < -10] = 0.0
        batch_label[:, 0] = np.log((featurearray[:, 0] + 1.0).astype('float'))
        #batch_label = np.where(batch_label > 0.08, 0.08, batch_label)
        #print(batch_data)
        #print(batch_label)

        t_batch_data = torch.from_numpy(batch_data.astype('float')).to(torch.float32)
        #print(self.label.shape)
        
        l_batch_data = torch.from_numpy(batch_label).to(torch.float32)
        return t_batch_data, l_batch_data

    def get_predict_batch_data(self, batch_size=-1):
        #indices = torch.randperm(len(self.x))[:batch_size] 
        #return self.x[indices], self.y[indices]
        batch_data = np.zeros((batch_size, 197), dtype=np.float32)
        batch_label = np.zeros((batch_size, 1), dtype=np.float32)
        #for i in range(n):
        i = self.k * self.Batch_size
        self.k = (self.k + 1) % self.n
        batch_data = self.df.iloc[i:i+batch_size, :]
        featurearray = batch_data[self.col].to_numpy()
        
        date = batch_data['TRADE_DT'].to_numpy()
        stocks = batch_data['S_INFO_WINDCODE'].to_numpy()
        lastday = batch_data['target_02'].to_numpy()

        batch_data = featurearray[:, 1:]
        batch_data[batch_data > 10] = 0.0
        batch_data[batch_data < -10] = 0.0
        batch_label[:, 0] = np.log((featurearray[:, 0] + 1.0).astype('float'))
        #batch_label = np.where(batch_label > 0.08, 0.08, batch_label)

        t_batch_data = torch.from_numpy(batch_data.astype('float')).to(torch.float32)
        #print(self.label.shape)
        
        l_batch_data = torch.from_numpy(batch_label).to(torch.float32)
        return t_batch_data, l_batch_data, date, stocks, lastday


class StockData_mini_batch_full_tensor(data.Dataset):
    def __init__(self, fn, begindate, enddate, flag, batch_size):
        cf = configparser.ConfigParser()
        cf.read('fea.ini')
        feaitems = cf.items('allfeatures')
        tmpfeature = dict((i[0],i[1]) for i in feaitems)
        fealist = tmpfeature['fealist']
        fealist = fealist.split(',')
        print("fealist训练特征如下：", fealist)
        self.col = fealist.copy()


        fp = open(fn, 'rb')
        df = pickle.load(fp)
        #fp = open(fn, 'rb')
        #df = pd.read_pickle(fn)
        #df.dropna(subset=['date', 'stocks'],inplace=True)
        self.df = df.loc[ (df['TRADE_DT'] >= begindate) & (df['TRADE_DT'] <= enddate) ]
        #self.df.sort_values(by=['S_INFO_WINDCODE', 'TRADE_DT'], ascending=True, inplace=True)
        #self.df['target_01'] = (self.df.S_FWDS_ADJOPEN.shift(-1) - self.df.S_FWDS_ADJCLOSE) / self.df.S_FWDS_ADJCLOSE
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
        
        if flag == 'train':
            self.df = shuffle(self.df)
            #self.df.sample(frac=1).reset_index(drop=True)
        #featurearray = train_df[self.col].to_numpy()
        self.f_len = len(self.df)
        self.k = 0
        self.Batch_size = batch_size

        if int(self.f_len) % int(self.Batch_size) == 0:
            self.n = int(self.f_len / self.Batch_size)
        else:
            self.n = int(self.f_len / self.Batch_size) + 1
        
    def __len__(self):
        return self.f_len

    def __getitem__(self, idx):
        return self.df[idx], self.df[idx]

    def get_batch_data(self, batch_size=-1):
        #indices = torch.randperm(len(self.x))[:batch_size] 
        #return self.x[indices], self.y[indices]
        batch_data = np.zeros((batch_size, 197), dtype=np.float32)
        batch_label = np.zeros((batch_size, 1), dtype=np.float32)
        #for i in range(n):
        i = self.k * self.Batch_size
        self.k = (self.k + 1) % self.n
        batch_data = self.df.iloc[i:i+batch_size, :]
        featurearray = batch_data[self.col].to_numpy()
        
        batch_data = featurearray[:, 1:]
        batch_data[batch_data > 10] = 0.0
        batch_data[batch_data < -10] = 0.0
        batch_label[:, 0] = np.log((featurearray[:, 0] + 1.0).astype('float'))
        batch_label = np.where(batch_label > 0.08, 0.08, batch_label)
        #print(batch_data)
        #print(batch_label)

        t_batch_data = torch.from_numpy(batch_data.astype('float')).to(torch.float32)
        #print(self.label.shape)
        
        l_batch_data = torch.from_numpy(batch_label).to(torch.float32)
        return t_batch_data, l_batch_data

    def get_predict_batch_data(self, batch_size=-1):
        #indices = torch.randperm(len(self.x))[:batch_size] 
        #return self.x[indices], self.y[indices]
        batch_data = np.zeros((batch_size, 197), dtype=np.float32)
        batch_label = np.zeros((batch_size, 1), dtype=np.float32)
        #for i in range(n):
        i = self.k * self.Batch_size
        self.k = (self.k + 1) % self.n
        batch_data = self.df.iloc[i:i+batch_size, :]
        featurearray = batch_data[self.col].to_numpy()
        
        date = batch_data['TRADE_DT'].to_numpy()
        stocks = batch_data['S_INFO_WINDCODE'].to_numpy()

        batch_data = featurearray[:, 1:]
        batch_data[batch_data > 10] = 0.0
        batch_data[batch_data < -10] = 0.0
        batch_label[:, 0] = np.log((featurearray[:, 0] + 1.0).astype('float'))
        #batch_label = np.where(batch_label > 0.08, 0.08, batch_label)

        t_batch_data = torch.from_numpy(batch_data.astype('float')).to(torch.float32)
        #print(self.label.shape)
        
        l_batch_data = torch.from_numpy(batch_label).to(torch.float32)
        return t_batch_data, l_batch_data, date, stocks






class StockData_mini_batch_limitup_tensor(data.Dataset):
    def __init__(self, fn, begindate, enddate, flag, batch_size):
        cf = configparser.ConfigParser()
        cf.read('fea.ini')
        feaitems = cf.items('allfeatures')
        tmpfeature = dict((i[0],i[1]) for i in feaitems)
        fealist = tmpfeature['fealist']
        fealist = fealist.split(',')
        print("fealist训练特征如下：", fealist)
        self.col = fealist.copy()


        fp = open(fn, 'rb')
        df = pickle.load(fp)
        #fp = open(fn, 'rb')
        #df = pd.read_pickle(fn)
        #df.dropna(subset=['date', 'stocks'],inplace=True)
        print(df)
        self.df = df
        self.df.sort_values(by=['S_INFO_WINDCODE', 'TRADE_DT'], ascending=True, inplace=True)
        self.df['target_01'] = (self.df['S_FWDS_ADJOPEN'].shift(-2) - self.df['S_FWDS_ADJOPEN'].shift(-1)) / self.df['S_FWDS_ADJOPEN'].shift(-1)
        
        #print(self.df['target_01'])
        #print(self.df['S_FWDS_ADJOPEN'])
        #print(self.df['S_FWDS_ADJCLOSE'])
        self.df.dropna(axis=0, subset=['target_01'], inplace=True)
        self.df = self.df.dropna(thresh=2)
        self.df.fillna(0, inplace=True)

        #self.df = self.df.loc[ ((self.df.LIMIT_UP / self.df.S_FWDS_ADJCLOSE - 1) > 0.01) & (self.df.S_DQ_VOLUME * self.df.S_DQ_AVGPRICE * 100 > 50000000) ]
        #self.df = self.df.loc[ ((self.df.LIMIT_UP / self.df.S_FWDS_ADJOPEN - 1) > 0.01) ]
        #self.df = self.df.loc[self.df.target_02 >= 0.097]
        #self.df = self.df.loc[ ((self.df.LIMIT_DOWN / self.df.S_FWDS_ADJCLOSE - 1) < -0.02)]	
        #self.df.target_01 = self.df.target_01.where(self.df.target_01 <= 0.08, 0.08)
        print(self.df.loc[self.df.S_INFO_WINDCODE == '000002'][['S_INFO_WINDCODE', 'TRADE_DT' ,'target_01']])
        self.df = self.df.loc[ ((self.df.LIMIT_UP.shift(-1) / self.df.S_FWDS_ADJOPEN.shift(-1) - 1) > 0.01) ]
        self.df = self.df.loc[((self.df.S_FWDS_ADJCLOSE / self.df.S_FWDS_ADJOPEN) > 1.097)]
        self.df = self.df.loc[ (df['TRADE_DT'] >= begindate) & (df['TRADE_DT'] <= enddate) ]

        print(self.df)
        
        if flag == 'train':
            self.df = shuffle(self.df)
            #self.df.sample(frac=1).reset_index(drop=True)
        #featurearray = train_df[self.col].to_numpy()
        self.f_len = len(self.df)
        self.k = 0
        self.Batch_size = batch_size

        if int(self.f_len) % int(self.Batch_size) == 0:
            self.n = int(self.f_len / self.Batch_size)
        else:
            self.n = int(self.f_len / self.Batch_size) + 1
        
    def __len__(self):
        return self.f_len

    def __getitem__(self, idx):
        return self.df[idx], self.df[idx]

    def get_batch_data(self, batch_size=-1):
        #indices = torch.randperm(len(self.x))[:batch_size] 
        #return self.x[indices], self.y[indices]
        batch_data = np.zeros((batch_size, 197), dtype=np.float32)
        batch_label = np.zeros((batch_size, 1), dtype=np.float32)
        #for i in range(n):
        i = self.k * self.Batch_size
        self.k = (self.k + 1) % self.n
        batch_data = self.df.iloc[i:i+batch_size, :]
        featurearray = batch_data[self.col].to_numpy()
        
        batch_data = featurearray[:, 1:]
        batch_data[batch_data > 10] = 0.0
        batch_data[batch_data < -10] = 0.0
        batch_label[:, 0] = np.log((featurearray[:, 0] + 1.0).astype('float'))
        batch_label = np.where(batch_label > 0.08, 0.08, batch_label)
        #print(batch_data)
        #print(batch_label)

        t_batch_data = torch.from_numpy(batch_data.astype('float')).to(torch.float32)
        #print(self.label.shape)
        
        l_batch_data = torch.from_numpy(batch_label).to(torch.float32)
        return t_batch_data, l_batch_data

    def get_predict_batch_data(self, batch_size=-1):
        #indices = torch.randperm(len(self.x))[:batch_size] 
        #return self.x[indices], self.y[indices]
        batch_data = np.zeros((batch_size, 197), dtype=np.float32)
        batch_label = np.zeros((batch_size, 1), dtype=np.float32)
        #for i in range(n):
        i = self.k * self.Batch_size
        self.k = (self.k + 1) % self.n
        batch_data = self.df.iloc[i:i+batch_size, :]
        featurearray = batch_data[self.col].to_numpy()
        
        date = batch_data['TRADE_DT'].to_numpy()
        stocks = batch_data['S_INFO_WINDCODE'].to_numpy()

        batch_data = featurearray[:, 1:]
        batch_data[batch_data > 10] = 0.0
        batch_data[batch_data < -10] = 0.0
        batch_label[:, 0] = np.log((featurearray[:, 0] + 1.0).astype('float'))
        #batch_label = np.where(batch_label > 0.08, 0.08, batch_label)

        t_batch_data = torch.from_numpy(batch_data.astype('float')).to(torch.float32)
        #print(self.label.shape)
        
        l_batch_data = torch.from_numpy(batch_label).to(torch.float32)
        return t_batch_data, l_batch_data, date, stocks


class StockData_mini_batch_day_inday_tensor(data.Dataset):
    def __init__(self, fn_day, fn_halfday, begindate, enddate, flag, batch_size):
        cf = configparser.ConfigParser()
        cf.read('fea_day_halfday.ini')
        feaitems = cf.items('allfeatures')
        tmpfeature = dict((i[0],i[1]) for i in feaitems)
        fealist = tmpfeature['fealist']
        fealist = fealist.split(',')
        print("fealist训练特征如下：", fealist)
        self.col = fealist.copy()

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


    
            fp_halfday = open(fn_halfday, 'rb')
            df_halfday = pd.read_pickle(fn_halfday)
            df_halfday = df_halfday.drop(columns=['date', 'stock'])
    
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
    
            self.df.to_pickle("Ashares2train_tushare_day_halfday_new.pickle")
            return
       
        day_halfday_data = open('Ashares2train_tushare_day_halfday_del8.pickle', 'rb')
        self.df = pickle.load(day_halfday_data) 
        self.df = self.df.loc[ (self.df['TRADE_DT_day'] >= begindate) & (self.df['TRADE_DT_day'] <= enddate) ]
        self.df.replace([np.inf, -np.inf], np.nan, inplace=True)
        self.df.dropna(axis=0, subset=['target_01'], inplace=True)

        print(self.df['target_01'])

        self.df = self.df.dropna(thresh=2)
        self.df.fillna(0, inplace=True)
        print(self.df)

        self.f_len = len(self.df)
        self.k = 0
        self.Batch_size = batch_size

        if int(self.f_len) % int(self.Batch_size) == 0:
            self.n = int(self.f_len / self.Batch_size)
        else:
            self.n = int(self.f_len / self.Batch_size) + 1
        batch_label = np.zeros((self.f_len, 1), dtype=np.float32)
        batch_data = self.df
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
        fp = open("tensor_data_halfday_del8_" + flag + ".pickle", 'wb')
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
        

class StockData_mini_batch_day_rel_tensor(data.Dataset):
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
            #df_day = df_day.drop(columns=['date', 'stock'])
    
            df_day.sort_values(by=['S_INFO_WINDCODE', 'TRADE_DT'], ascending=True, inplace=True)
            
    
            df_day['sd'] = df_day['S_INFO_WINDCODE'] + df_day['TRADE_DT']
            print(df_day)


            names = ['stock1', 'stock2', 'rel', 'cnt']
            dtype = {'stock1': np.str_, 'stock2': np.str_, 'rel': np.float32, 'cnt': np.float32}
            rel_df = pd.read_csv('result.txt.pair', header=None, sep=' ', names=names, index_col=False, dtype=dtype)
            #rel_df.drop(rel_df[rel_df.cnt < 100].index, inplace=True)
            rel_df.loc[rel_df.cnt < 100, 'rel'] = -1.0
            rel_df = rel_df.loc[ (rel_df['rel'] != 1.0) ] 
            matrix_df = rel_df.pivot(index='stock1', columns=['stock2'], values='rel')
            print(matrix_df)
            tmp_df = matrix_df[['000001']]
            tmp_df.reset_index(inplace=True)
            tmp_df['stock'] = tmp_df['stock1']
            print(tmp_df)
            df_day = pd.merge(df_day, tmp_df, how='right', on='stock')
            df_day.drop(columns=['stock1', '000001'], inplace=True)

            rel_df.sort_values(by=['stock1', 'rel'], ascending=[True, False], inplace=True)
            print(rel_df)
            matrix_list = []
            for stock1, group in rel_df.groupby('stock1'):
                matrix_list.append(group.head(1))

            matrix_df = pd.concat(matrix_list, axis=0)
            matrix_df['stock'] = matrix_df['stock1']
            matrix_df.drop(columns=['stock1', 'cnt'], inplace=True)

            df_day_tmp = df_day[['stock', 'date', 'sd']]
            df_day_rel = pd.merge(df_day_tmp, matrix_df, how='left', on='stock')
            print(df_day_rel)
            df_day_rel['sd'] = df_day_rel['stock2'] + df_day_rel['date']
            #res_df = df_day_rel.join(df_day.set_index('sd'), on='sd', lsuffix='_day', rsuffix='_relday')
            df_day.drop(columns=['stock', 'date'], inplace=True)
            res_df = df_day_rel.join(df_day.set_index('sd'), on='sd')
            print(res_df)
            res_df['sd'] = res_df['stock'] + res_df['date']
             
            #res_list = []
            #for stock, group in res_df.groupby('sd'):
            #    res_list.append(group[self.col].mean())
            #
            #mean_df = pd.concat(res_list, axis=0)
            #print(mean_df)
            
            #rel_fea_df = res_df.groupby('sd')[self.col].mean()
            rel_fea_df = res_df
            print(rel_fea_df)
            rel_fea_df.drop(columns=['target_01'], inplace=True)
            rel_fea_df.drop(columns=['inc_close', 'inc_open', 'inc_high', 'inc_avg', 'inc_low', 'cs_rank_close', 'cs_rank_open', 'cs_rank_high', 'cs_rank_avg', 'cs_rank_low', 'cs_rank_amount', 'OR_1', 'OR_2', 'OR_3', 'OR_4', 'OR_5', 'OR_6', 'OR_7', 'OR_8', 'OR_9', 'OR_10', 'OR_-1', 'OR_-2', 'OR_-3', 'OR_-4', 'OR_-5', 'OR_-6', 'OR_-7', 'OR_-8', 'OR_-9', 'OR_-10', 'OR_0', 'CR_1', 'CR_2', 'CR_3', 'CR_4', 'CR_5', 'CR_6', 'CR_7', 'CR_8', 'CR_9', 'CR_10', 'CR_-1', 'CR_-2', 'CR_-3', 'CR_-4', 'CR_-5', 'CR_-6', 'CR_-7', 'CR_-8', 'CR_-9', 'CR_-10', 'CR_0'], inplace=True)
            #rel_fea_df['sd'] = rel_fea_df.index 
                         


            self.df = df_day.join(rel_fea_df.set_index('sd'), on='sd', lsuffix='_day', rsuffix='_relday')
    
    
            self.df = self.df.loc[ (self.df['TRADE_DT_day'] >= begindate) & (self.df['TRADE_DT_day'] <= enddate) ]
            self.df.sort_values(by=['S_INFO_WINDCODE_day', 'TRADE_DT_day'], ascending=True, inplace=True)
            
            print(self.df[['S_INFO_WINDCODE_day', 'TRADE_DT_day', 'S_FWDS_ADJCLOSE_day', 'target_01']])
            print(self.df)
            #print(self.df['target_01'])
            #print(self.df['S_FWDS_ADJOPEN'])
            #print(self.df['S_FWDS_ADJCLOSE'])
            self.df.dropna(axis=0, subset=['target_01'], inplace=True)
            self.df = self.df.dropna(thresh=2)
            self.df.fillna(0, inplace=True)
    
            self.df = self.df.loc[ ((self.df.LIMIT_UP_day / self.df.S_FWDS_ADJCLOSE_day - 1) > 0.01) & (self.df.S_DQ_VOLUME_day * self.df.S_DQ_AVGPRICE_day * 100 > 50000000) ]
            #self.df = self.df.loc[ ((self.df.LIMIT_UP / self.df.S_FWDS_ADJCLOSE - 1) > 0.01) & (self.df.S_DQ_VOLUME * self.df.S_DQ_AVGPRICE * 100 > 50000000) ]	
            #self.df = self.df.loc[ ((self.df.LIMIT_DOWN / self.df.S_FWDS_ADJCLOSE - 1) < -0.02)]	
            #self.df.target_01 = self.df.target_01.where(self.df.target_01 <= 0.08, 0.08)
            print(self.df)
    
            self.df.to_pickle("Ashares2train_tushare_day_rel_new.pickle")
            return
       

        cf = configparser.ConfigParser()
        cf.read('fea_rel.ini')
        feaitems = cf.items('allfeatures')
        tmpfeature = dict((i[0],i[1]) for i in feaitems)
        fealist = tmpfeature['fealist']
        fealist = fealist.split(',')
        print("fealist训练特征如下：", fealist)
        self.col = fealist.copy()

        day_halfday_data = open('Ashares2train_tushare_day_rel_new.pickle', 'rb')
        self.df = pickle.load(day_halfday_data) 
        self.df = self.df.loc[ (self.df['TRADE_DT'] >= begindate) & (self.df['TRADE_DT'] <= enddate) ]
        self.df.replace([np.inf, -np.inf], np.nan, inplace=True)
        self.df.dropna(axis=0, subset=['target_01'], inplace=True)
        self.df = self.df.dropna(thresh=2)
        self.df.fillna(0, inplace=True)


        self.df = self.df.loc[ ((self.df.LIMIT_UP / self.df.S_FWDS_ADJCLOSE - 1) > 0.01) & (self.df.S_DQ_VOLUME * self.df.S_DQ_AVGPRICE * 100 > 50000000) ]


        print(self.df)
            #self.df.sample(frac=1).reset_index(drop=True)
        #featurearray = train_df[self.col].to_numpy()
        self.f_len = len(self.df)
        self.k = 0
        self.Batch_size = batch_size

        if int(self.f_len) % int(self.Batch_size) == 0:
            self.n = int(self.f_len / self.Batch_size)
        else:
            self.n = int(self.f_len / self.Batch_size) + 1

        batch_label = np.zeros((self.f_len, 1), dtype=np.float32)
        batch_data = self.df
        featurearray = batch_data[self.col].to_numpy()

        self.date = batch_data['TRADE_DT'].to_numpy()
        self.stocks = batch_data['S_INFO_WINDCODE'].to_numpy()

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
        fp = open("tensor_data_rel_" + flag + ".pickle", 'wb')
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





class StockData(data.Dataset):
    def __init__(self, file_name, file_len) :
        self.fp = open(os.path.join('/home/duyimin/trade_predict/', file_name), 'r')
        self.f_len = file_len
    
    def __len__(self):
        return self.f_len
    
    def __getitem__(self, idx):
        line = self.fp.readline().strip('\n\r')
        if not line :
            self.fp.seek(0, 0)
            line = self.fp.readline().strip('\n\r')
        line_data, label = line.split('\t')
        data = np.array([float(item) for item in line_data.split(' ')], dtype=np.float32)
        t_data = torch.from_numpy(data)
        l_data = torch.FloatTensor([label])
        return t_data, l_data

    def get_batch_data(self, batch_size):
        batch_data = np.zeros((batch_size, 159), dtype=np.float32)
        batch_label = np.zeros((batch_size, 1), dtype=np.float32)
        for i in range(batch_size):
            line = self.fp.readline().strip('\n\r')
            while ((line is None)):
                if not line :
                    self.fp.seek(0, 0)
                line = self.fp.readline().strip('\n\r')
            
            #line_data, label = line.split('\t')
            l = line.split(' ')
            fl = []
            for number in l:
                if number == 'nan' or number == 'inf':
                    fl.append(0.0)
                elif number == '':
                    continue
                elif float(number) > 1e+2 or float(number) < -1e+2:
                    fl.append(0.0)
                else:
                    fl.append(float(number))
            data = np.array(fl, dtype=np.float32)
            batch_data[i] = data[1:]
            batch_label[i][0] = np.log(float(data[0]) + 1.0)
            

        t_batch_data = torch.from_numpy(batch_data)
        l_batch_data = torch.from_numpy(batch_label)
        return t_batch_data, l_batch_data


class StockData_trend(data.Dataset):
    def __init__(self, file_name, file_len) :
        self.fp = open(os.path.join('/da2/search/wanghexiang/stock_data', file_name), 'r')
        self.f_len = file_len
    
    def __len__(self):
        return self.f_len
    
    def __getitem__(self, idx):
        line = self.fp.readline().strip('\n\r')
        if not line :
            self.fp.seek(0, 0)
            line = self.fp.readline().strip('\n\r')
        line_data, label = line.split('\t')
        data = np.array([float(item) for item in line_data.split(' ')], dtype=np.float32)
        t_data = torch.from_numpy(data)
        l_data = torch.FloatTensor([label])
        return t_data, l_data

    def get_batch_data(self, batch_size):
        batch_data = np.zeros((batch_size, 97), dtype=np.float32)
        batch_label = np.zeros((batch_size, 1), dtype=np.float32)
        for i in range(batch_size):
            line = self.fp.readline().strip('\n\r')
            while ((line is None) or len(line.split('\t')) != 2):
                if not line :
                    self.fp.seek(0, 0)
                line = self.fp.readline().strip('\n\r')
            
            line_data, label = line.split('\t')
            data = np.array([float(item) for item in line_data.split(' ')], dtype=np.float32)
            batch_data[i] = data
            batch_label[i][0] = label

        t_batch_data = torch.from_numpy(batch_data)
        l_batch_data = torch.from_numpy(batch_label)
        return t_batch_data, l_batch_data


class IdStockData(data.Dataset):
    def __init__(self, file_name, file_len) :
        self.fp = open(os.path.join('/da2/search/wanghexiang/stock_data', file_name), 'r')
        self.f_len = file_len

    def __len__(self):
        return self.f_len
    
    def __getitem__(self, idx):
        line = self.fp.readline().strip('\n\r')
        if not line :
            self.fp.seek(0, 0)
            line = self.fp.readline().strip('\n\r')
        id_line_data, line_data, label = line.split('\t')
        data = np.array([float(item) for item in line_data.split(' ')], dtype=np.float32)
        
        id_data = [int(item) for item in id_line_data.split(' ')]
        
        all_id_data = np.zeros([9000])
        for idx, item in enumerate(id_data):
            all_id_data[item + 100 * idx] = 1
        
        t_id_data = torch.from_numpy(id_data)
        t_data = torch.from_numpy(data)
        l_data = torch.FloatTensor([label])
        return t_id_data, t_data, l_data

    def get_batch_data(self, batch_size):
        batch_id_data = np.zeros((batch_size, 9090), dtype=np.float32)
        batch_data = np.zeros((batch_size, 96), dtype=np.float32)
        batch_label = np.zeros((batch_size, 1), dtype=np.float32)
        for i in range(batch_size):
            line = self.fp.readline().strip('\n\r')
            if not line :
                self.fp.seek(0, 0)
                line = self.fp.readline().strip('\n\r')
            id_line_data, line_data, label = line.split('\t')
            data = np.array([float(item) for item in line_data.split(' ')], dtype=np.float32)
            
            id_data = [int(float(item)) for item in id_line_data.split(' ')]
            
            for idx, item in enumerate(id_data):
                batch_id_data[i][item + idx * 100] = 1

            batch_data[i] = data
            batch_label[i][0] = label

        t_batch_data = torch.from_numpy(batch_data)
        t_id_batch_data = torch.from_numpy(batch_id_data)
        l_batch_data = torch.from_numpy(batch_label)
        return t_id_batch_data, t_batch_data, l_batch_data


# class Np_Idx_DataSetFile_v2_encode_ctr(data.Dataset):
#     def __init__(self, file_name, file_len) :
#         self.fp = open(os.path.join('/da2/search/wanghexiang/dnn_rank', file_name), 'r')
#         self.f_len = file_len
    
#     def __len__(self):
#         return self.f_len
    
#     def __getitem__(self, idx):

#         line = self.fp.readline().strip('\n\r')
#         if not line :
#             self.fp.seek(0, 0)
#             line = self.fp.readline().strip('\n\r')
        
#         q_str, k_str, h_str, l_str, num_str = line.split('\t')

#         max_q_len = 20
#         max_k_len = 50
#         seq_len = 10

#         q_list = np.zeros((seq_len, max_q_len), dtype=np.int64)
#         k_list = np.zeros((seq_len, max_k_len), dtype=np.int64)
#         h_list = np.zeros((seq_len,), dtype=np.int64)
#         l_list = np.zeros((max_q_len,), dtype=np.int64)
        
#         q_idx = 0
#         k_idx = 0
#         l_len = 0

#         q_len_list = np.ones((seq_len, 1), dtype=np.float32)
#         k_len_list = np.ones((seq_len, 1), dtype=np.float32)

#         if k_str != '<None>':
#             for item in k_str.split(','):
#                 t = np.array([int(i) for i in item.split('|')][0:max_k_len])
#                 len_t = t.size
#                 k_list[k_idx][0: len_t] = t
#                 k_len_list[k_idx][0] = len_t
#                 k_idx += 1

#         if q_str != '<None>':
#             for item in q_str.split(','):
#                 t = np.array([int(i) for i in item.split('|')][0:max_q_len])
#                 len_t = t.size
#                 q_list[q_idx][0: len_t] = t
#                 q_len_list[q_idx][0] = len_t
#                 q_idx += 1
        
#         if h_str != '<None>': 
#             t = np.array([int(item) for item in h_str.split('|')])
#             len_t = t.size
#             h_list[0:len_t] = t
        
#         l_np = np.array([int(item) for item in l_str.split('|')][0:max_q_len])
#         l_len = l_np.size
#         l_list[0:l_len] = l_np

#         num_list = np.array([float(t) for t in num_str.split('|')], dtype=np.float32)
#         label = int(num_list[-1])
#         ctr_list = num_list[:-1]

#         k_batch = torch.from_numpy(k_list)
#         q_batch = torch.from_numpy(q_list)
#         h_batch = torch.from_numpy(h_list)
#         l_batch = torch.from_numpy(l_list)
#         ctr_batch = torch.from_numpy(ctr_list)
#         label_batch = torch.FloatTensor([label])
#         q_len_batch = torch.from_numpy(q_len_list)
#         k_len_batch = torch.from_numpy(k_len_list)
#         l_len_batch = torch.FloatTensor([l_len])

#         return q_batch, k_batch, h_batch, l_batch, ctr_batch, label_batch, q_len_batch, k_len_batch, l_len_batch
    
#     def get_batch_data(self, batch_size):
#         max_q_len = 20
#         max_k_len = 50
#         seq_len = 10
#         q_list = np.zeros((batch_size, seq_len, max_q_len), dtype=np.int64)
#         k_list = np.zeros((batch_size, seq_len, max_k_len), dtype=np.int64)
#         h_list = np.zeros((batch_size, seq_len), dtype=np.int64)
#         l_list = np.zeros((batch_size, max_q_len), dtype=np.int64)

#         ctr_list = np.ones((batch_size, 1), dtype=np.int64)
#         q_len_list = np.ones((batch_size, seq_len, 1), dtype=np.float32)
#         k_len_list = np.ones((batch_size, seq_len, 1), dtype=np.float32)
#         l_len_list = np.ones((batch_size, 1), dtype=np.float32)
#         label_list = np.zeros((batch_size, 1), dtype=np.float32)

#         for i in range(batch_size):
#             q_idx = 0
#             k_idx = 0
#             l_len = 0
#             line = self.fp.readline().strip('\n\r')
#             if not line :
#                 self.fp.seek(0, 0)
#                 line = self.fp.readline().strip('\n\r')
            
#             q_str, k_str, h_str, l_str, num_str = line.split('\t')

#             if k_str != '<None>':
#                 for item in k_str.split(','):
#                     t = np.array([int(i) for i in item.split('|') if i != ''][0:max_k_len])
#                     len_t = t.size
#                     k_list[i][k_idx][0: len_t] = t
#                     k_len_list[i][k_idx][0] = len_t
#                     k_idx += 1

#             if q_str != '<None>':
#                 for item in q_str.split(','):
#                     t = np.array([int(i) for i in item.split('|') if i != ''][0:max_q_len])
#                     len_t = t.size
#                     q_list[i][q_idx][0: len_t] = t
#                     q_len_list[i][q_idx][0] = len_t
#                     q_idx += 1
            
#             if h_str != '<None>': 
#                 t = np.array([int(item) for item in h_str.split('|')])
#                 len_t = t.size
#                 h_list[i][0:len_t] = t
            
#             l_np = np.array([int(item) for item in l_str.split('|') if item != ''][0:max_q_len])
#             l_len = l_np.size
#             l_len_list[i][0] = l_len
#             l_list[i][0:l_len] = l_np

#             num_list = np.array([float(t) for t in num_str.split('|')], dtype=np.float32)
#             label_list[i][0] = int(num_list[-1])
#             ctr_list[i, 0] = int(round(num_list[0] * 100)) % 100 

#         k_batch = torch.from_numpy(k_list)
#         q_batch = torch.from_numpy(q_list)
#         h_batch = torch.from_numpy(h_list)
#         l_batch = torch.from_numpy(l_list)
#         ctr_batch = torch.from_numpy(ctr_list)
#         label_batch = torch.from_numpy(label_list)
#         q_len_batch = torch.from_numpy(q_len_list)
#         k_len_batch = torch.from_numpy(k_len_list)
#         l_len_batch = torch.from_numpy(l_len_list)

#         return q_batch, k_batch, h_batch, l_batch, ctr_batch, label_batch, q_len_batch, k_len_batch, l_len_batch


# class Np_Idx_DataSetFile_v3(data.Dataset):
#     def __init__(self, file_name, file_len) :
#         self.fp = open(os.path.join('/da2/search/wanghexiang/dnn_rank', file_name), 'r')
#         self.f_len = file_len
    
#     def __len__(self):
#         return self.f_len
    
#     def __getitem__(self, idx):
#         line = self.fp.readline().strip('\n\r')
#         if not line :
#             self.fp.seek(0, 0)
#             line = self.fp.readline().strip('\n\r')
#         q_str, k_str, h_str, l_str, c_str, num_str = line.split('\t')

#         max_q_len = 20
#         max_k_len = 50
#         seq_len = 1
#         q_list = np.zeros((seq_len, max_q_len), dtype=np.int64)
#         k_list = np.zeros((seq_len, max_k_len), dtype=np.int64)
#         h_list = np.zeros((seq_len,), dtype=np.int64)
#         l_list = np.zeros((max_q_len,), dtype=np.int64)
        
#         q_idx = 0
#         k_idx = 0
#         l_len = 0

#         q_len_list = np.ones((seq_len, 1), dtype=np.float32)
#         k_len_list = np.ones((seq_len, 1), dtype=np.float32)

#         if k_str != '<None>':
#             for item in k_str.split(','):
#                 t = np.array([int(i) for i in item.split('|')][0:max_k_len])
#                 len_t = t.size
#                 k_list[k_idx][0: len_t] = t
#                 k_len_list[k_idx][0] = len_t
#                 k_idx += 1

#         if q_str != '<None>':
#             for item in q_str.split(','):
#                 t = np.array([int(i) for i in item.split('|')][0:max_q_len])
#                 len_t = t.size
#                 q_list[q_idx][0: len_t] = t
#                 q_len_list[q_idx][0] = len_t
#                 q_idx += 1
        
#         if h_str != '<None>': 
#             t = np.array([int(item) for item in h_str.split('|')])
#             len_t = t.size
#             h_list[0:len_t] = t
        
#         l_np = np.array([int(item) for item in l_str.split('|')][0:max_q_len])
#         l_len = l_np.size
#         l_list[0:l_len] = l_np

#         num_list = np.array([float(t) for t in num_str.split('|')], dtype=np.float32)
#         label = int(num_list[-1])
#         ctr_list = num_list[:-1]

#         k_batch = torch.from_numpy(k_list)
#         q_batch = torch.from_numpy(q_list)
#         h_batch = torch.from_numpy(h_list)
#         l_batch = torch.from_numpy(l_list)
#         c_batch =  torch.from_numpy([int(c_str) + 1])
#         ctr_batch = torch.from_numpy(ctr_list)
#         label_batch = torch.FloatTensor([label])
#         q_len_batch = torch.from_numpy(q_len_list)
#         k_len_batch = torch.from_numpy(k_len_list)
#         l_len_batch = torch.FloatTensor([l_len])

#         return q_batch, k_batch, h_batch, l_batch, ctr_batch, label_batch, q_len_batch, k_len_batch, l_len_batch
    
#     def get_batch_data(self, batch_size):
#         max_q_len = 20
#         max_k_len = 50
#         seq_len = 10
#         q_list = np.zeros((batch_size, seq_len, max_q_len), dtype=np.int64)
#         k_list = np.zeros((batch_size, seq_len, max_k_len), dtype=np.int64)
#         h_list = np.zeros((batch_size, seq_len), dtype=np.int64)
#         l_list = np.zeros((batch_size, max_q_len), dtype=np.int64)
#         c_list = np.zeros((batch_size, 1), dtype=np.int64)

#         ctr_list = np.ones((batch_size, 2), dtype=np.float32)
#         q_len_list = np.ones((batch_size, seq_len, 1), dtype=np.float32)
#         k_len_list = np.ones((batch_size, seq_len, 1), dtype=np.float32)
#         l_len_list = np.ones((batch_size, 1), dtype=np.float32)
#         label_list = np.zeros((batch_size, 1), dtype=np.float32)

#         all_index = 0
#         for i in range(batch_size):
#             all_index += 1
#             q_idx = 0
#             k_idx = 0
#             l_len = 0
#             line = self.fp.readline().strip('\n\r')
#             if not line :
#                 self.fp.seek(0, 0)
#                 line = self.fp.readline().strip('\n\r')
#             try:
#                 q_str, k_str, h_str, l_str, c_str, num_str = line.split('\t')
            
#             except:
#                 continue
            
#             if k_str != '<None>':
#                 for item in k_str.split(','):
#                     t = np.array([int(i)  + 1 for i in item.split('|') if i != ''][0:max_k_len])
#                     len_t = t.size
#                     k_list[i, k_idx, 0: len_t] = t
#                     k_len_list[i, k_idx, 0] = len_t
#                     k_idx += 1

#             if q_str != '<None>':
#                 for item in q_str.split(','):
#                     t = np.array([int(i) + 1 for i in item.split('|') if i != ''][0:max_q_len])
#                     len_t = t.size
#                     q_list[i][q_idx][0: len_t] = t
#                     q_len_list[i][q_idx][0] = len_t
#                     q_idx += 1
            
#             if h_str != '<None>': 
#                 t = np.array([int(item) for item in h_str.split('|')])
#                 len_t = t.size
#                 h_list[i, 0:len_t] = t
            
#             l_np = np.array([int(item) for item in l_str.split('|') if item != ''][0:max_q_len])
#             l_len = l_np.size
#             l_len_list[i, 0] = l_len
#             l_list[i, 0:l_len] = l_np

#             num_list = np.array([float(t) for t in num_str.split('|')], dtype=np.float32)
#             label_list[i, 0] = int(num_list[-1])
#             ctr_list[i] = num_list[:-1]
#             c_list[i, 0] = int(c_str)

#         if all_index != batch_size:
#             k_list = k_list[0: all_index]
#             q_list = q_list[0: all_index]
#             h_list = h_list[0: all_index]
#             l_list = l_list[0: all_index]
#             c_list = c_list[0: all_index]
#             ctr_list = ctr_list[0: all_index]
#             label_list = label_list[0: all_index]
#             q_len_list = q_len_list[0: all_index]
#             k_len_list = k_len_list[0: all_index]
#             l_len_list = l_len_list[0: all_index]

#         k_batch = torch.from_numpy(k_list)
#         q_batch = torch.from_numpy(q_list)
#         h_batch = torch.from_numpy(h_list)
#         l_batch = torch.from_numpy(l_list)
#         c_batch = torch.from_numpy(c_list)
#         ctr_batch = torch.from_numpy(ctr_list)
#         label_batch = torch.from_numpy(label_list)
#         q_len_batch = torch.from_numpy(q_len_list)
#         k_len_batch = torch.from_numpy(k_len_list)
#         l_len_batch = torch.from_numpy(l_len_list)

#         return q_batch, k_batch, h_batch, l_batch, c_batch, ctr_batch, label_batch, q_len_batch, k_len_batch, l_len_batch


# def collate_fn_filter(batch):
#     batch = list(filter(lambda x: len(x) == 9, batch))
#     if len(batch) == 0:
#         return torch.Tensor()
#     return default_collate(batch)

if __name__ == '__main__' :
    # 测试代码
    # import time
    # data_path = '/home/wanghexiang/dnn_rank/'
    dataset = StockData_mini_batch_day_inday_tensor('Ashares2train_tushare.pickle', 'Ashares2train_tushare_half_new.pickle', '20110110', '20201231', 'train', 128)
    testdataset = StockData_mini_batch_day_inday_tensor('Ashares2train_tushare.pickle', 'Ashares2train_tushare_half_new.pickle', '20210101', '20210830', 'test', 128)
    #dataset = StockData_mini_batch_day_rel_tensor('Ashares2train_tushare.pickle', 'Ashares2train_tushare_half_new.pickle', '20110101', '20210831', 'init', 128)
    #dataset = StockData_mini_batch_day_inday_tensor('Ashares2train_tushare.pickle', 'Ashares2train_tushare_half_new.pickle', '20110110', '20210830', 'init', 128)
    # t2 = 0.0

    # for i in range(5000):
    #     start = time.time()
    #     data = dataset.get_batch_data(128)
    #     end = time.time()
    #     t2 += end -start
    #     print(end - start)
    pass
