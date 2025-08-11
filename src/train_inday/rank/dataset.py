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
import random
import configparser
from sklearn.utils import shuffle

def collate_fn_filter(batch):
    batch = list(filter(lambda x: len(x) == 9, batch))
    if len(batch) == 0:
        return torch.Tensor()
    return default_collate(batch)


class StockData_mini_batch_tensor_for_train_gpu_gbm(data.Dataset):
    def __init__(self, fn, file_len=0, device=torch.device('cpu')):
        self.device = device
        fp = open(fn, 'rb')
        d = pickle.load(fp)
        self.x = d['x'].to(device)
        self.y = d['y'].to(device)
        #self.gbm_param = d['gbm']
        self.f_len = len(self.y)
        self.idx = 0
        self.device = device
        #w = self.gbm_simulation(self.gbm_param)


    def __len__(self):
        return self.f_len

    def __getitem__(self, idx):
        return self.x[i], self.y[i]

    def get_batch_data(self, batch_size=-1):
        if self.idx == 0:
            indices = torch.randperm(self.f_len)
            self.x = self.x[indices]
            self.y = self.y[indices]

            #self.gbm_param = self.gbm_param[indices]
            #self.y = self.gbm_simulation(self.gbm_param)


            self.y = self.y.to(self.device)
            #self.y = torch.clip(self.y, -0.3, 0.08)
            self.y[torch.isnan(self.y)] = 0
            y = torch.flatten(self.y)

            #y = torch.clip(y, -0.3, 0.3)
            y = torch.round(y*100)
            y = torch.clip(y, -20, 20)
            y = (y + 20).to(torch.int64)
            y = F.one_hot(y)
            self.y = y.to(torch.float32)
            #print(self.y[torch.isnan(self.y)]
            #print(len(self.y[torch.isnan(self.y)]))
            #print(self.y)
        s = self.idx * batch_size
        e = s + batch_size
        #print(s,e)

        if self.f_len % batch_size == 0:
            self.idx = (self.idx + 1) % ( self.f_len // batch_size ) 
        else:
            self.idx = (self.idx + 1) % ( (self.f_len // batch_size) + 1 ) 

        #self.idx = (self.idx + 1) % ( self.f_len // batch_size )
        return self.x[s:e], self.y[s:e]

    def gbm_simulation(self, gbm_param):
        close_open = gbm_param[:, 0]
        mu = gbm_param[:, 1]
        sigma = gbm_param[:, 2]
        dmean = gbm_param[:, 4]
        m = len(sigma)

        n=120
        dt=1/120
        #dt = 1.0/gbm_param[:, 3]

        x0 = torch.exp(close_open)
        st = torch.normal(0, np.sqrt(dt), (1, n * m))
        #st = torch.normal(0, 1, (1, n * m))
        st = torch.reshape(st, (n, m))
        step = torch.exp((mu - sigma**2 / 2) * dt  + sigma * st)
        simulation = x0 * torch.cumprod(step, dim=0)
        ret = torch.log(simulation[-1] / (1+dmean))
        ret = torch.reshape(ret, (m, 1))
        ret = ret.to(torch.float32)
        return ret
 
class StockData_mini_batch_tensor_gbm(data.Dataset):
    def __init__(self, fn, file_len=0, device=torch.device('cpu')):
        self.device = device
        #w = self.gbm_simulation(self.gbm_param)
        fp = open(fn, 'rb')
        d = pickle.load(fp)
        self.x = d['x'].to(device)
        self.y = d['y'].to(device)
        print(self.x.shape, self.y.shape, flush=True)
        maskx = torch.where(torch.isnan(self.x).sum(dim=1, keepdim=False) <= 100, True, False)
        #maskx = torch.where((~torch.isnan(self.x)).sum(dim=1, keepdim=False) >= 2, True, False)
        masky = torch.where(torch.isnan(self.y), False, True)
        mask = (maskx & masky).numpy().tolist()
        #mask = torch.where(torch.isnan(self.x).sum(dim=1, keepdim=False) <= 2, True, False).numpy().tolist()
        print(len(mask))
        self.x = torch.tensor(self.x.numpy()[mask])

        self.x = torch.where(torch.isnan(self.x), torch.full_like(self.x, 0), self.x)
        self.x = torch.where(torch.isinf(self.x), torch.full_like(self.x, 0), self.x)
        self.x = torch.where(self.x > 10, torch.full_like(self.x, 0), self.x)
        self.x = torch.where(self.x < -10, torch.full_like(self.x, 0), self.x)
        pd.DataFrame(self.x.numpy()).head(1000).to_csv('train.csv')

        self.y = torch.tensor(self.y.numpy()[mask])


        self.y = torch.where(torch.isnan(self.y), torch.full_like(self.y, 0), self.y)
        self.y = torch.where(torch.isinf(self.y), torch.full_like(self.y, 0), self.y)

        self.date = np.array(d['date'])[mask].tolist()
        self.stocks = np.array(d['stocks'])[mask].tolist()

        tmp_df = pd.DataFrame({'date':self.date, 'return':self.y.numpy().tolist()})
        tmp_df['dmean'] = tmp_df.groupby(['date'])['return'].transform('mean')
        tmp_df['adj_return'] = tmp_df['return'] - tmp_df['dmean']
        print(tmp_df)
        self.y = torch.tensor(tmp_df.adj_return.to_numpy())


        self.y = torch.log(self.y + 1.0)
        #self.y = torch.where(self.y > 0.08, torch.full_like(self.y, 0.08), self.y)
        self.f_len = len(self.y)
        self.idx = 0

        #self.today_close = np.array(d['today_close'])[mask].tolist()
        #self.halfday_close = np.array(d['halfday_close'])[mask].tolist()
        print(self.x.shape, self.y.shape, self.y, self.date[:10], self.stocks[:10], mask[:10], flush=True)


    def __len__(self):
        return self.f_len

    def __getitem__(self, idx):
        return self.x[i], self.y[i]

    def get_batch_data(self, batch_size=-1):
        if self.idx == 0:
            indices = torch.randperm(self.f_len)
            self.x = self.x[indices]
            self.y = self.y[indices]
            
            #close gbm
            #self.gbm_param = self.gbm_param[indices]
            #self.y = self.gbm_simulation(self.gbm_param)


            self.y = self.y.to(self.device)
            #self.y = torch.clip(self.y, -0.3, 0.08)
            self.y[torch.isnan(self.y)] = 0
            y = torch.flatten(self.y)
            #print(y.numpy().tolist())

            #y = torch.clip(y, -0.3, 0.3)
            y = torch.round(y*100)
            y = torch.clip(y, -20, 20)
            y = (y + 20).to(torch.int64)
            y = F.one_hot(y)
            self.y = y.to(torch.float32)
            print(self.y.shape)
            #print(self.y[torch.isnan(self.y)]
            #print(len(self.y[torch.isnan(self.y)]))
        s = self.idx * batch_size
        e = s + batch_size
        #print(s,e)

        if self.f_len % batch_size == 0:
            self.idx = (self.idx + 1) % ( self.f_len // batch_size ) 
        else:
            self.idx = (self.idx + 1) % ( (self.f_len // batch_size) + 1 ) 

        #self.idx = (self.idx + 1) % ( self.f_len // batch_size )
        return self.x[s:e], self.y[s:e]

    def gbm_simulation(self, gbm_param):
        close_open = gbm_param[:, 0]
        mu = gbm_param[:, 1]
        sigma = gbm_param[:, 2]
        dmean = gbm_param[:, 4]
        m = len(sigma)

        n=119
        dt=1/119
        #dt = 1.0/gbm_param[:, 3]

        x0 = torch.exp(close_open)
        st = torch.normal(0, np.sqrt(dt), (1, n * m))
        #st = torch.normal(0, 1, (1, n * m))
        st = torch.reshape(st, (n, m))
        step = torch.exp((mu - sigma**2 / 2) * dt  + sigma * st)
        simulation = x0 * torch.cumprod(step, dim=0)
        ret = torch.log(simulation[-1] / (1+dmean))
        ret = torch.reshape(ret, (m, 1))
        ret = ret.to(torch.float32)
        return ret


class StockData_with_gbm(data.Dataset):
    def __init__(self, fn, file_len=0, device=torch.device('cpu')):
        self.device = device
        #w = self.gbm_simulation(self.gbm_param)


        fp = open(fn, 'rb')
        d = pickle.load(fp)
        self.x = d['x'].to(device)
        self.y = d['y'].to(device)
        self.gbm_param = d['gbm']
        maskx = torch.where(torch.isnan(self.x).sum(dim=1, keepdim=False) <= 0, True, False)
        #maskx = torch.where((~torch.isnan(self.x)).sum(dim=1, keepdim=False) >= 2, True, False)
        masky = torch.where(torch.isnan(self.y), False, True)
        maskz = torch.where(torch.isnan(self.gbm_param).sum(dim=1, keepdim=False) <= 0, True, False)
        mask = (maskx & masky & maskz).numpy().tolist()
        #mask = torch.where(torch.isnan(self.x).sum(dim=1, keepdim=False) <= 2, True, False).numpy().tolist()
        print(len(mask))
        self.x = torch.tensor(self.x.numpy()[mask])

        self.x = torch.where(torch.isnan(self.x), torch.full_like(self.x, 0), self.x)
        self.x = torch.where(torch.isinf(self.x), torch.full_like(self.x, 0), self.x)
        self.x = torch.where(self.x > 10, torch.full_like(self.x, 0), self.x)
        self.x = torch.where(self.x < -10, torch.full_like(self.x, 0), self.x)
        pd.DataFrame(self.x.numpy()).head(1000).to_csv('train.csv')

        self.y = torch.tensor(self.y.numpy()[mask])


        self.y = torch.where(torch.isnan(self.y), torch.full_like(self.y, 0), self.y)
        self.y = torch.where(torch.isinf(self.y), torch.full_like(self.y, 0), self.y)
        self.y = torch.log(self.y + 1.0)


        self.gbm_param = torch.tensor(self.gbm_param.numpy()[mask])

        #self.y = torch.where(self.y > 0.08, torch.full_like(self.y, 0.08), self.y)
        self.f_len = len(self.y)
        self.idx = 0

        self.date = np.array(d['date'])[mask].tolist()
        self.stocks = np.array(d['stocks'])[mask].tolist()
        #self.today_close = np.array(d['today_close'])[mask].tolist()
        #self.halfday_close = np.array(d['halfday_close'])[mask].tolist()
        print(self.x.shape, self.y.shape, self.y, self.date[:10], self.stocks[:10], mask[:10], flush=True)


    def __len__(self):
        return self.f_len

    def __getitem__(self, idx):
        return self.x[i], self.y[i]

    def get_batch_data(self, batch_size=-1):
        if self.idx == 0:
            indices = torch.randperm(self.f_len)
            self.x = self.x[indices]
            self.y = self.y[indices]
            
            #close gbm
            self.gbm_param = self.gbm_param[indices]
            
            self.y = self.gbm_simulation(self.gbm_param)


            self.y = self.y.to(self.device)
            #self.y = torch.clip(self.y, -0.3, 0.08)
            self.y[torch.isnan(self.y)] = 0
            y = torch.flatten(self.y)
            #print(y.numpy().tolist())

            #y = torch.clip(y, -0.3, 0.3)
            y = torch.round(y*100)
            y = torch.clip(y, -20, 20)
            y = (y + 20).to(torch.int64)
            y = F.one_hot(y)
            self.y = y.to(torch.float32)
            print(self.y.shape)
            #print(self.y[torch.isnan(self.y)]
            #print(len(self.y[torch.isnan(self.y)]))
        s = self.idx * batch_size
        e = s + batch_size
        #print(s,e)

        if self.f_len % batch_size == 0:
            self.idx = (self.idx + 1) % ( self.f_len // batch_size ) 
        else:
            self.idx = (self.idx + 1) % ( (self.f_len // batch_size) + 1 ) 

        #self.idx = (self.idx + 1) % ( self.f_len // batch_size )
        return self.x[s:e], self.y[s:e]

    def gbm_simulation(self, gbm_param):
        close_open = gbm_param[:, 0]
        mu = gbm_param[:, 1]
        sigma = gbm_param[:, 2]
        dmean = gbm_param[:, 4]
        m = len(sigma)

        n=1
        dt=1/1

        x0 = torch.exp(close_open)
        st = torch.normal(0, np.sqrt(dt), (1, n * m))
        #st = torch.normal(0, 1, (1, n * m))
        st = torch.reshape(st, (n, m))
        step = torch.exp((mu - sigma**2 / 2) * dt  + sigma * st)
        simulation = x0 * torch.cumprod(step, dim=0)
        print(torch.isnan(x0).sum(dim=0, keepdim=False))
        print(torch.isnan(sigma).sum(dim=0, keepdim=False))
        ret = torch.log(simulation[-1] / (1+dmean))
        ret = torch.reshape(ret, (m, 1))
        ret = ret.to(torch.float32)
        #return ret
        return torch.log(x0 / (1+dmean))


class StockData_gbm(data.Dataset):
    def __init__(self, fn, file_len=0, device=torch.device('cpu')):
        self.device = device
        #w = self.gbm_simulation(self.gbm_param)


        fp = open(fn, 'rb')
        d = pickle.load(fp)
        self.x = d['x'].to(device)
        self.y = d['y'].to(device)
        self.gbm_param = d['gbm']
        maskx = torch.where(torch.isnan(self.x).sum(dim=1, keepdim=False) <= 0, True, False)
        #maskx = torch.where((~torch.isnan(self.x)).sum(dim=1, keepdim=False) >= 2, True, False)
        masky = torch.where(torch.isnan(self.y), False, True)
        maskz = torch.where(torch.isnan(self.gbm_param).sum(dim=1, keepdim=False) <= 0, True, False)
        mask = (maskx & masky & maskz).numpy().tolist()
        #mask = torch.where(torch.isnan(self.x).sum(dim=1, keepdim=False) <= 2, True, False).numpy().tolist()
        print(len(mask))
        self.x = torch.tensor(self.x.numpy()[mask])

        self.x = torch.where(torch.isnan(self.x), torch.full_like(self.x, 0), self.x)
        self.x = torch.where(torch.isinf(self.x), torch.full_like(self.x, 0), self.x)
        self.x = torch.where(self.x > 10, torch.full_like(self.x, 0), self.x)
        self.x = torch.where(self.x < -10, torch.full_like(self.x, 0), self.x)
        pd.DataFrame(self.x.numpy()).head(1000).to_csv('train.csv')

        self.y = torch.tensor(self.y.numpy()[mask])


        self.y = torch.where(torch.isnan(self.y), torch.full_like(self.y, 0), self.y)
        self.y = torch.where(torch.isinf(self.y), torch.full_like(self.y, 0), self.y)
        self.y = torch.log(self.y + 1.0)


        self.gbm_param = torch.tensor(self.gbm_param.numpy()[mask])

        #self.y = torch.where(self.y > 0.08, torch.full_like(self.y, 0.08), self.y)
        self.f_len = len(self.y)
        self.idx = 0

        self.date = np.array(d['date'])[mask].tolist()
        self.stocks = np.array(d['stocks'])[mask].tolist()
        #self.today_close = np.array(d['today_close'])[mask].tolist()
        #self.halfday_close = np.array(d['halfday_close'])[mask].tolist()
        print(self.x.shape, self.y.shape, self.y, self.date[:10], self.stocks[:10], mask[:10], flush=True)


    def __len__(self):
        return self.f_len

    def __getitem__(self, idx):
        return self.x[i], self.y[i]

    def get_batch_data(self, batch_size=-1):
        if self.idx == 0:
            indices = torch.randperm(self.f_len)
            self.x = self.x[indices]
            self.y = self.y[indices]
            
            #close gbm
            self.gbm_param = self.gbm_param[indices]
            
            self.y = self.gbm_simulation(self.gbm_param)


            self.y = self.y.to(self.device)
            #self.y = torch.clip(self.y, -0.3, 0.08)
            self.y[torch.isnan(self.y)] = 0
            y = torch.flatten(self.y)
            #print(y.numpy().tolist())

            #y = torch.clip(y, -0.3, 0.3)
            y = torch.round(y*100)
            y = torch.clip(y, -20, 20)
            y = (y + 20).to(torch.int64)
            y = F.one_hot(y)
            self.y = y.to(torch.float32)
            print(self.y.shape)
            #print(self.y[torch.isnan(self.y)]
            #print(len(self.y[torch.isnan(self.y)]))
        s = self.idx * batch_size
        e = s + batch_size
        #print(s,e)

        if self.f_len % batch_size == 0:
            self.idx = (self.idx + 1) % ( self.f_len // batch_size ) 
        else:
            self.idx = (self.idx + 1) % ( (self.f_len // batch_size) + 1 ) 

        #self.idx = (self.idx + 1) % ( self.f_len // batch_size )
        return self.x[s:e], self.y[s:e]

    def gbm_simulation(self, gbm_param):
        close_open = gbm_param[:, 0]
        mu = gbm_param[:, 1]
        sigma = gbm_param[:, 2]
        dmean = gbm_param[:, 4]
        m = len(sigma)

        n=119
        dt=1/119

        x0 = torch.exp(close_open)
        st = torch.normal(0, np.sqrt(dt), (1, n * m))
        #st = torch.normal(0, 1, (1, n * m))
        st = torch.reshape(st, (n, m))
        step = torch.exp((mu - sigma**2 / 2) * dt  + sigma * st)
        simulation = x0 * torch.cumprod(step, dim=0)
        print(torch.isnan(x0).sum(dim=0, keepdim=False))
        print(torch.isnan(sigma).sum(dim=0, keepdim=False))
        ret = torch.log(simulation[-1] / (1+dmean))
        ret = torch.reshape(ret, (m, 1))
        ret = ret.to(torch.float32)
        return ret



class StockData_mini_batch_tensor_gbm_pre(data.Dataset):
    def __init__(self, fn, file_len=0, device=torch.device('cpu')):
        self.device = device
        #w = self.gbm_simulation(self.gbm_param)
        fp = open(fn, 'rb')
        d = pickle.load(fp)
        self.x = d['x'].to(device)
        self.y = d['y'].to(device)
        maskx = torch.where(torch.isnan(self.x).sum(dim=1, keepdim=False) <= 100, True, False)
        #maskx = torch.where((~torch.isnan(self.x)).sum(dim=1, keepdim=False) >= 2, True, False)
        masky = torch.where(torch.isnan(self.y), False, True)
        mask = (maskx & masky).numpy().tolist()
        #mask = torch.where(torch.isnan(self.x).sum(dim=1, keepdim=False) <= 2, True, False).numpy().tolist()
        print(len(mask))
        self.x = torch.tensor(self.x.numpy()[mask])

        self.x = torch.where(torch.isnan(self.x), torch.full_like(self.x, 0), self.x)
        self.x = torch.where(torch.isinf(self.x), torch.full_like(self.x, 0), self.x)
        self.x = torch.where(self.x > 10, torch.full_like(self.x, 0), self.x)
        self.x = torch.where(self.x < -10, torch.full_like(self.x, 0), self.x)
        pd.DataFrame(self.x.numpy()).head(1000).to_csv('train.csv')

        self.y = torch.tensor(self.y.numpy()[mask])


        self.y = torch.where(torch.isnan(self.y), torch.full_like(self.y, 0), self.y)
        self.y = torch.where(torch.isinf(self.y), torch.full_like(self.y, 0), self.y)
        self.y = torch.log(self.y + 1.0)
        #self.y = torch.where(self.y > 0.08, torch.full_like(self.y, 0.08), self.y)
        self.f_len = len(self.y)
        self.idx = 0

        l = map(lambda x:str(x)[:10].replace('-', ''), d['date'][mask])

        self.date = list(l)
        #self.date = np.array(d['date'])[mask].tolist()
        self.stocks = np.array(d['stocks'])[mask].tolist()
        self.today_close = np.array(d['today_close'])[mask].tolist()
        #self.after_open = np.array(d['after_open'])[mask].tolist()
        self.halfday_close = np.array(d['halfday_close'])[mask].tolist()
        print(self.x.shape, self.y.shape, self.y, self.date[:10], self.stocks[:10], mask[:10], flush=True)


    def __len__(self):
        return self.f_len

    def __getitem__(self, idx):
        return self.x[i], self.y[i]

    def get_batch_data(self, batch_size=-1):
        if self.idx == 0:
            indices = torch.randperm(self.f_len)
            self.x = self.x[indices]
            self.y = self.y[indices]

            self.gbm_param = self.gbm_param[indices]
            self.y = self.gbm_simulation(self.gbm_param)


            self.y = self.y.to(self.device)
            #self.y = torch.clip(self.y, -0.3, 0.08)
            self.y[torch.isnan(self.y)] = 0
            y = torch.flatten(self.y)
            #print(y.numpy().tolist())

            #y = torch.clip(y, -0.3, 0.3)
            y = torch.round(y*100)
            y = torch.clip(y, -20, 20)
            y = (y + 20).to(torch.int64)
            y = F.one_hot(y)
            self.y = y.to(torch.float32)
            #print(self.y[torch.isnan(self.y)]
            #print(len(self.y[torch.isnan(self.y)]))
        s = self.idx * batch_size
        e = s + batch_size
        #print(s,e)

        if self.f_len % batch_size == 0:
            self.idx = (self.idx + 1) % ( self.f_len // batch_size ) 
        else:
            self.idx = (self.idx + 1) % ( (self.f_len // batch_size) + 1 ) 

        #self.idx = (self.idx + 1) % ( self.f_len // batch_size )
        return self.x[s:e], self.y[s:e]

    def gbm_simulation(self, gbm_param):
        close_open = gbm_param[:, 0]
        mu = gbm_param[:, 1]
        sigma = gbm_param[:, 2]
        dmean = gbm_param[:, 4]
        m = len(sigma)

        n=119
        dt=1/119
        #dt = 1.0/gbm_param[:, 3]

        x0 = torch.exp(close_open)
        st = torch.normal(0, np.sqrt(dt), (1, n * m))
        #st = torch.normal(0, 1, (1, n * m))
        st = torch.reshape(st, (n, m))
        step = torch.exp((mu - sigma**2 / 2) * dt  + sigma * st)
        simulation = x0 * torch.cumprod(step, dim=0)
        ret = torch.log(simulation[-1] / (1+dmean))
        ret = torch.reshape(ret, (m, 1))
        ret = ret.to(torch.float32)
        return ret

class StockData_mini_batch_mse(data.Dataset):
    def __init__(self, fn, file_len=0, device=torch.device('cpu')):
        self.device = device
        #w = self.gbm_simulation(self.gbm_param)
        fp = open(fn, 'rb')
        d = pickle.load(fp)
        self.x = d['x'].to(device)
        self.y = d['y'].to(device)
        maskx = torch.where(torch.isnan(self.x).sum(dim=1, keepdim=False) <= 100, True, False)
        #maskx = torch.where((~torch.isnan(self.x)).sum(dim=1, keepdim=False) >= 2, True, False)
        masky1 = torch.where(torch.isnan(self.y), False, True)
        masky2 = torch.where(self.y > 0.2, False, True)
        masky3 = torch.where(self.y < -0.2, False, True)
        mask = (maskx & masky1 & masky2 & masky3).numpy().tolist()
        #mask = torch.where(torch.isnan(self.x).sum(dim=1, keepdim=False) <= 2, True, False).numpy().tolist()
        #print(len(mask))
        self.x = torch.tensor(self.x.numpy()[mask])

        self.x = torch.where(torch.isnan(self.x), torch.full_like(self.x, 0), self.x)
        self.x = torch.where(torch.isinf(self.x), torch.full_like(self.x, 0), self.x)
        self.x = torch.where(self.x > 10, torch.full_like(self.x, 0), self.x)
        self.x = torch.where(self.x < -10, torch.full_like(self.x, 0), self.x)
        pd.DataFrame(self.x.numpy()).head(1000).to_csv('train.csv')

        self.y = torch.tensor(self.y.numpy()[mask])


        self.y = torch.where(torch.isnan(self.y), torch.full_like(self.y, 0), self.y)
        self.y = torch.where(torch.isinf(self.y), torch.full_like(self.y, 0), self.y)



        #self.y = torch.where(self.y > 0.08, torch.full_like(self.y, 0.08), self.y)
        l = map(lambda x:str(x)[:10].replace('-', ''), d['date'][mask])
        #print(list(l))
        #self.date = np.array(d['date'])[mask].tolist()
        self.date = list(l)
        self.stocks = np.array(d['stocks'])[mask].tolist()
        self.today_close = np.array(d['today_close'])[mask].tolist()
        #self.after_open = np.array(d['after_open'])[mask].tolist()
        self.halfday_close = np.array(d['halfday_close'])[mask].tolist()
        
        self.real_return = self.y.clone()

        tmp_df = pd.DataFrame({'date':self.date, 'return':self.y.numpy().tolist()})
        tmp_df['dmean'] = tmp_df.groupby(['date'])['return'].transform('mean')
        tmp_df['adj_return'] = tmp_df['return'] - tmp_df['dmean']
        #print(tmp_df)
        self.y = torch.tensor(tmp_df.adj_return.to_numpy())

        #!!!!!
        self.y = torch.log(self.y + 1.0)

        self.f_len = len(self.y)
        self.idx = 0


        #print(self.x.shape, self.y.shape, self.y, self.date[:10], self.stocks[:10], mask[:10], flush=True)


    def __len__(self):
        return self.f_len

    def __getitem__(self, idx):
        return self.x[i], self.y[i]

    def get_batch_data(self, batch_size=-1):
        if self.idx == 0:
            indices = torch.randperm(self.f_len)
            self.x = self.x[indices]
            self.y = self.y[indices]

            #####
            #self.y = torch.clip(self.y, -0.2, 0.2)

        s = self.idx * batch_size
        e = s + batch_size
        #print(s,e)

        if self.f_len % batch_size == 0:
            self.idx = (self.idx + 1) % ( self.f_len // batch_size ) 
        else:
            self.idx = (self.idx + 1) % ( (self.f_len // batch_size) + 1 ) 

        #self.idx = (self.idx + 1) % ( self.f_len // batch_size )
        return self.x[s:e], self.y[s:e]



class StockData_day_neu_mse(data.Dataset):
    def __init__(self, fn, file_len=0, device=torch.device('cpu')):
        self.device = device
        #w = self.gbm_simulation(self.gbm_param)
        fp = open(fn, 'rb')
        d = pickle.load(fp)
        self.x = d['x'].to(device)
        self.y = d['y'].to(device)
        #maskx = torch.where(torch.isnan(self.x).sum(dim=1, keepdim=False) <= 20, True, False)
        maskx = torch.where(torch.isnan(self.x).sum(dim=1, keepdim=False) <= 20, True, False)
        #maskx = torch.where((~torch.isnan(self.x)).sum(dim=1, keepdim=False) >= 2, True, False)
        masky1 = torch.where(torch.isnan(self.y), False, True)
        masky2 = torch.where(self.y > 0.2, False, True)
        masky3 = torch.where(self.y < -0.2, False, True)
        mask = (maskx & masky1 & masky2 & masky3).numpy().tolist()
        #mask = torch.where(torch.isnan(self.x).sum(dim=1, keepdim=False) <= 2, True, False).numpy().tolist()
        #print(len(mask))
        self.x = torch.tensor(self.x.numpy()[mask])

        self.x = torch.where(torch.isnan(self.x), torch.full_like(self.x, 0), self.x)
        self.x = torch.where(torch.isinf(self.x), torch.full_like(self.x, 0), self.x)
        self.x = torch.where(self.x > 10, torch.full_like(self.x, 0), self.x)
        self.x = torch.where(self.x < -10, torch.full_like(self.x, 0), self.x)
        pd.DataFrame(self.x.numpy()).head(1000).to_csv('train.csv')

        self.y = torch.tensor(self.y.numpy()[mask])


        self.y = torch.where(torch.isnan(self.y), torch.full_like(self.y, 0), self.y)
        self.y = torch.where(torch.isinf(self.y), torch.full_like(self.y, 0), self.y)


        l = map(lambda x:str(x)[:10].replace('-', ''), d['date'][mask])
        #print(list(l))
        #self.date = np.array(d['date'])[mask].tolist()
        self.date = list(l)
        self.stocks = np.array(d['stocks'])[mask].tolist()
        self.today_close = np.array(d['today_close'])[mask].tolist()
        #self.after_open = np.array(d['after_open'])[mask].tolist()
        
        self.real_return = self.y.clone()


        tmp_df = pd.DataFrame({'date':self.date, 'return':self.y.numpy().tolist()})
        tmp_df['dmean'] = tmp_df.groupby(['date'])['return'].transform('mean')
        tmp_df['adj_return'] = tmp_df['return'] - tmp_df['dmean']
        #print(tmp_df)
        self.y = torch.tensor(tmp_df.adj_return.to_numpy())

        #!!!!!
        #self.y = torch.where(self.y > 0.08, torch.full_like(self.y, 0.08), self.y)
        self.y = torch.log(self.y + 1.0)

        self.f_len = len(self.y)
        self.idx = 0


        #print(self.x.shape, self.y.shape, self.y, self.date[:10], self.stocks[:10], mask[:10], flush=True)


    def __len__(self):
        return self.f_len

    def __getitem__(self, idx):
        return self.x[i], self.y[i]

    def get_batch_data(self, batch_size=-1):
        if self.idx == 0:
            indices = torch.randperm(self.f_len)
            self.x = self.x[indices]
            self.y = self.y[indices]

            #####
            #self.y = torch.clip(self.y, -0.2, 0.2)

        s = self.idx * batch_size
        e = s + batch_size
        #print(s,e)

        if self.f_len % batch_size == 0:
            self.idx = (self.idx + 1) % ( self.f_len // batch_size ) 
        else:
            self.idx = (self.idx + 1) % ( (self.f_len // batch_size) + 1 ) 

        #self.idx = (self.idx + 1) % ( self.f_len // batch_size )
        return self.x[s:e], self.y[s:e]



class StockData_day_neu_rank_mse(data.Dataset):
    def __init__(self, fn, file_len=0, device=torch.device('cpu')):
        self.device = device
        #w = self.gbm_simulation(self.gbm_param)
        fp = open(fn, 'rb')
        d = pickle.load(fp)
        self.x = d['x'].to(device)
        self.y = d['y'].to(device)
        self.z = d['z'].to(device)
        #maskx = torch.where(torch.isnan(self.x).sum(dim=1, keepdim=False) <= 20, True, False)
        maskx = torch.where(torch.isnan(self.x).sum(dim=1, keepdim=False) <= 20, True, False)
        #maskx = torch.where((~torch.isnan(self.x)).sum(dim=1, keepdim=False) >= 2, True, False)
        masky1 = torch.where(torch.isnan(self.z), False, True)
        masky2 = torch.where(self.z > 0.2, False, True)
        masky3 = torch.where(self.z < -0.2, False, True)
        mask = (maskx & masky1 & masky2 & masky3).numpy().tolist()
        #mask = torch.where(torch.isnan(self.x).sum(dim=1, keepdim=False) <= 2, True, False).numpy().tolist()
        #print(len(mask))
        self.x = torch.tensor(self.x.numpy()[mask])

        self.x = torch.where(torch.isnan(self.x), torch.full_like(self.x, 0), self.x)
        self.x = torch.where(torch.isinf(self.x), torch.full_like(self.x, 0), self.x)
        self.x = torch.where(self.x > 10, torch.full_like(self.x, 0), self.x)
        self.x = torch.where(self.x < -10, torch.full_like(self.x, 0), self.x)
        pd.DataFrame(self.x.numpy()).head(1000).to_csv('train.csv')

        self.y = torch.tensor(self.y.numpy()[mask])
        self.z = torch.tensor(self.z.numpy()[mask])
        self.y = torch.where(torch.isnan(self.y), torch.full_like(self.y, 0), self.y)
        self.y = torch.where(torch.isinf(self.y), torch.full_like(self.y, 0), self.y)

#test decile
        #self.y = self.y * 10
        #self.y = self.y.int().float() / 10.0


        l = map(lambda x:str(x)[:10].replace('-', ''), d['date'][mask])
        #print(list(l))
        #self.date = np.array(d['date'])[mask].tolist()
        self.date = list(l)
        self.stocks = np.array(d['stocks'])[mask].tolist()
        self.today_close = np.array(d['today_close'])[mask].tolist()
        #self.after_open = np.array(d['after_open'])[mask].tolist()
        
        self.real_return = self.z.clone()


        #tmp_df = pd.DataFrame({'date':self.date, 'return':self.y.numpy().tolist()})
        #tmp_df['dmean'] = tmp_df.groupby(['date'])['return'].transform('mean')
        #tmp_df['adj_return'] = tmp_df['return'] - tmp_df['dmean']
        #self.y = torch.tensor(tmp_df.adj_return.to_numpy())

        #!!!!!
        #self.y = torch.where(self.y > 0.08, torch.full_like(self.y, 0.08), self.y)
        #self.y = torch.log(self.y + 1.0)

        self.f_len = len(self.y)
        self.idx = 0


        #print(self.x.shape, self.y.shape, self.y, self.date[:10], self.stocks[:10], mask[:10], flush=True)


    def __len__(self):
        return self.f_len

    def __getitem__(self, idx):
        return self.x[i], self.y[i]

    def get_batch_data(self, batch_size=-1):
        if self.idx == 0:
            indices = torch.randperm(self.f_len)
            self.x = self.x[indices]
            self.y = self.y[indices]

            #####
            #self.y = torch.clip(self.y, -0.2, 0.2)

        s = self.idx * batch_size
        e = s + batch_size
        #print(s,e)

        if self.f_len % batch_size == 0:
            self.idx = (self.idx + 1) % ( self.f_len // batch_size ) 
        else:
            self.idx = (self.idx + 1) % ( (self.f_len // batch_size) + 1 ) 

        #self.idx = (self.idx + 1) % ( self.f_len // batch_size )
        return self.x[s:e], self.y[s:e]


class StockData_day_neu_rank_mse_shp(data.Dataset):
    def __init__(self, fn, file_len=0, device=torch.device('cpu')):
        self.device = device
        #w = self.gbm_simulation(self.gbm_param)
        fp = open(fn, 'rb')
        d = pickle.load(fp)
        self.x = d['x'].to(device)
        self.y = d['y'].to(device)
        self.z = d['z'].to(device)
        #maskx = torch.where(torch.isnan(self.x).sum(dim=1, keepdim=False) <= 20, True, False)
        maskx = torch.where(torch.isnan(self.x).sum(dim=1, keepdim=False) <= 20, True, False)
        #maskx = torch.where((~torch.isnan(self.x)).sum(dim=1, keepdim=False) >= 2, True, False)
        masky1 = torch.where(torch.isnan(self.z), False, True)
        #masky2 = torch.where(self.z > 0.2, False, True)
        #masky3 = torch.where(self.z < -0.2, False, True)
        mask = (maskx & masky1).numpy().tolist()
        #mask = torch.where(torch.isnan(self.x).sum(dim=1, keepdim=False) <= 2, True, False).numpy().tolist()
        #print(len(mask))
        self.x = torch.tensor(self.x.numpy()[mask])

        self.x = torch.where(torch.isnan(self.x), torch.full_like(self.x, 0), self.x)
        self.x = torch.where(torch.isinf(self.x), torch.full_like(self.x, 0), self.x)
        self.x = torch.where(self.x > 10, torch.full_like(self.x, 0), self.x)
        self.x = torch.where(self.x < -10, torch.full_like(self.x, 0), self.x)
        pd.DataFrame(self.x.numpy()).head(1000).to_csv('train.csv')

        self.y = torch.tensor(self.y.numpy()[mask])
        self.z = torch.tensor(self.z.numpy()[mask])
        self.y = torch.where(torch.isnan(self.y), torch.full_like(self.y, 0), self.y)
        self.y = torch.where(torch.isinf(self.y), torch.full_like(self.y, 0), self.y)

#test decile
        #self.y = self.y * 10
        #self.y = self.y.int().float() / 10.0


        l = map(lambda x:str(x)[:10].replace('-', ''), d['date'][mask])
        #print(list(l))
        #self.date = np.array(d['date'])[mask].tolist()
        self.date = list(l)
        self.stocks = np.array(d['stocks'])[mask].tolist()
        self.today_close = np.array(d['today_close'])[mask].tolist()
        #self.after_open = np.array(d['after_open'])[mask].tolist()
        
        self.real_return = self.z.clone()


        #tmp_df = pd.DataFrame({'date':self.date, 'return':self.y.numpy().tolist()})
        #tmp_df['dmean'] = tmp_df.groupby(['date'])['return'].transform('mean')
        #tmp_df['adj_return'] = tmp_df['return'] - tmp_df['dmean']
        #self.y = torch.tensor(tmp_df.adj_return.to_numpy())

        #!!!!!
        #self.y = torch.where(self.y > 0.08, torch.full_like(self.y, 0.08), self.y)
        #self.y = torch.log(self.y + 1.0)

        self.f_len = len(self.y)
        self.idx = 0


        #print(self.x.shape, self.y.shape, self.y, self.date[:10], self.stocks[:10], mask[:10], flush=True)


    def __len__(self):
        return self.f_len

    def __getitem__(self, idx):
        return self.x[i], self.y[i]

    def get_batch_data(self, batch_size=-1):
        if self.idx == 0:
            indices = torch.randperm(self.f_len)
            self.x = self.x[indices]
            self.y = self.y[indices]

            #####
            #self.y = torch.clip(self.y, -0.2, 0.2)

        s = self.idx * batch_size
        e = s + batch_size
        #print(s,e)

        if self.f_len % batch_size == 0:
            self.idx = (self.idx + 1) % ( self.f_len // batch_size ) 
        else:
            self.idx = (self.idx + 1) % ( (self.f_len // batch_size) + 1 ) 

        #self.idx = (self.idx + 1) % ( self.f_len // batch_size )
        return self.x[s:e], self.y[s:e]





class StockData_inday_neu_rank_mse(data.Dataset):
    def __init__(self, fn, file_len=0, device=torch.device('cpu')):
        self.device = device
        #w = self.gbm_simulation(self.gbm_param)
        fp = open(fn, 'rb')
        d = pickle.load(fp)
        self.x = d['x'].to(device)
        self.y = d['y'].to(device)
        self.z = d['z'].to(device)
        #maskx = torch.where(torch.isnan(self.x).sum(dim=1, keepdim=False) <= 20, True, False)
        maskx = torch.where(torch.isnan(self.x).sum(dim=1, keepdim=False) <= 100, True, False)
        #maskx = torch.where((~torch.isnan(self.x)).sum(dim=1, keepdim=False) >= 2, True, False)
        masky1 = torch.where(torch.isnan(self.z), False, True)
        masky2 = torch.where(self.z > 0.2, False, True)
        masky3 = torch.where(self.z < -0.2, False, True)
        mask = (maskx & masky1 & masky2 & masky3).numpy().tolist()
        #mask = torch.where(torch.isnan(self.x).sum(dim=1, keepdim=False) <= 2, True, False).numpy().tolist()
        #print(len(mask))
        self.x = torch.tensor(self.x.numpy()[mask])

        self.x = torch.where(torch.isnan(self.x), torch.full_like(self.x, 0), self.x)
        self.x = torch.where(torch.isinf(self.x), torch.full_like(self.x, 0), self.x)
        self.x = torch.where(self.x > 10, torch.full_like(self.x, 0), self.x)
        self.x = torch.where(self.x < -10, torch.full_like(self.x, 0), self.x)
        pd.DataFrame(self.x.numpy()).head(1000).to_csv('train.csv')

        self.y = torch.tensor(self.y.numpy()[mask])
        self.z = torch.tensor(self.z.numpy()[mask])
        self.y = torch.where(torch.isnan(self.y), torch.full_like(self.y, 0), self.y)
        self.y = torch.where(torch.isinf(self.y), torch.full_like(self.y, 0), self.y)


#decile 
        #print(self.y)
        self.y = self.y * 10
        self.y = self.y.int().float() / 10.0
        #print(self.y)



        #self.x = self.x * 10
        #self.x = self.x.int().float() / 10.0


        l = map(lambda x:str(x)[:10].replace('-', ''), d['date'][mask])
        #print(list(l))
        #self.date = np.array(d['date'])[mask].tolist()
        self.date = list(l)
        self.stocks = np.array(d['stocks'])[mask].tolist()
        self.today_close = np.array(d['today_close'])[mask].tolist()
        self.halfday_close = np.array(d['halfday_close'])[mask].tolist()
        #self.after_open = np.array(d['after_open'])[mask].tolist()
        
        self.real_return = self.z.clone()


        #tmp_df = pd.DataFrame({'date':self.date, 'return':self.y.numpy().tolist()})
        #tmp_df['dmean'] = tmp_df.groupby(['date'])['return'].transform('mean')
        #tmp_df['adj_return'] = tmp_df['return'] - tmp_df['dmean']
        #self.y = torch.tensor(tmp_df.adj_return.to_numpy())

        #!!!!!
        #self.y = torch.where(self.y > 0.08, torch.full_like(self.y, 0.08), self.y)
        #self.y = torch.log(self.y + 1.0)

        self.f_len = len(self.y)
        self.idx = 0

        self.tmp_df = pd.DataFrame({'date':self.date, 'stocks':self.stocks})
        self.tmp_df['index'] = range(len(self.tmp_df))
        #print(self.tmp_df)
        self.tmp_df.sort_values(by=['stocks', 'date'], ascending=True, inplace=True)
        #print(self.tmp_df)
        #tmp_df['index'] = random.randint(0,9)








        #print(self.x.shape, self.y.shape, self.y, self.date[:10], self.stocks[:10], mask[:10], flush=True)


    def __len__(self):
        return self.f_len

    def __getitem__(self, idx):
        return self.x[i], self.y[i]

    def get_batch_data(self, batch_size=-1):
        if self.idx == 0:
            indices = torch.randperm(self.f_len)
            self.x = self.x[indices]
            self.y = self.y[indices]

            #####
            #self.y = torch.clip(self.y, -0.2, 0.2)

        s = self.idx * batch_size
        e = s + batch_size
        #print(s,e)

        if self.f_len % batch_size == 0:
            self.idx = (self.idx + 1) % ( self.f_len // batch_size ) 
        else:
            self.idx = (self.idx + 1) % ( (self.f_len // batch_size) + 1 ) 

        #self.idx = (self.idx + 1) % ( self.f_len // batch_size )
        return self.x[s:e], self.y[s:e]

    def get_time_batch_data(self, batch_size=-1):
        if self.idx == 0:

            if self.f_len % batch_size == 0:
                self.idx_len = self.f_len // batch_size
            else:
                self.idx_len = (self.f_len // batch_size) + 1 

            self.indices = torch.tensor(self.tmp_df['index'].to_numpy())

            rand_shift = random.randint(0, self.f_len)
            self.indices = torch.roll(self.indices, rand_shift, 0)

            self.curs = torch.randperm(self.idx_len)

            self.x = self.x[self.indices]
            self.y = self.y[self.indices]

            #####
            #self.y = torch.clip(self.y, -0.2, 0.2)

        s = int(self.curs[self.idx]) * batch_size
        e = s + batch_size
        #print(s,e)

        if self.f_len % batch_size == 0:
            self.idx = (self.idx + 1) % ( self.f_len // batch_size ) 
        else:
            self.idx = (self.idx + 1) % ( (self.f_len // batch_size) + 1 ) 

        #self.idx = (self.idx + 1) % ( self.f_len // batch_size )
        return self.x[s:e], self.y[s:e]




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
    def __init__(self, fn, device = 'cpu'):

        fp = open(fn, 'rb')
        d = pickle.load(fp)
        self.x = d['x'].to(device)
        self.y = d['y'].to(device)
        self.f_len = len(self.y)
        self.idx = 0
 
        self.date = d['date']
        self.stocks = d['stocks']
        self.today_close = d['today_close']
        self.halfday_close = d['halfday_close']
        
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
        if self.f_len % batch_size == 0:
            self.idx = (self.idx + 1) % ( self.f_len // batch_size ) 
        else:
            self.idx = (self.idx + 1) % ( (self.f_len // batch_size) + 1 ) 
          
        return self.x[s:e], self.y[s:e]

    def get_predict_batch_data(self, batch_size=-1):
        s = self.idx * batch_size
        e = s + batch_size
        #print(s,e)
        if self.f_len % batch_size == 0:
            self.idx = (self.idx + 1) % ( self.f_len // batch_size ) 
        else:
            self.idx = (self.idx + 1) % ( (self.f_len // batch_size) + 1 ) 

        return self.x[s:e], self.y[s:e], self.date[s:e], self.stocks[s:e], self.today_close[s:e], self.halfday_close[s:e]
        



class StockData_mini_batch_rel_tensor(data.Dataset):
    def __init__(self, fn, device = 'cpu'):

        fp = open(fn, 'rb')
        d = pickle.load(fp)
        self.x = d['x'].to(device)
        self.y = d['y'].to(device)
        self.f_len = len(self.y)
        self.idx = 0
 
        self.date = d['date']
        self.stocks = d['stocks']
        
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
        return self.x[s:e], self.y[s:e], self.date[s:e], self.stocks[s:e]



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



class StockData_mini_batch_tensor_3part_2tar(data.Dataset):
    def __init__(self, fn, file_len=0, device=torch.device('cpu')):
        self.device = device
        #w = self.gbm_simulation(self.gbm_param)
        fp = open(fn, 'rb')
        d = pickle.load(fp)
        self.x = d['x'].to(device)
        self.y = d['y'].to(device)
        self.z = d['z'].to(device)
        print(self.x.shape, self.y.shape, flush=True)
        maskx = torch.where(torch.isnan(self.x).sum(dim=1, keepdim=False) <= 20, True, False)
        #maskx = torch.where((~torch.isnan(self.x)).sum(dim=1, keepdim=False) >= 2, True, False)
        masky = torch.where(torch.isnan(self.y), False, True)
        maskz = torch.where(torch.isnan(self.z), False, True)
        mask = (maskx & masky & maskz).numpy().tolist()
        #mask = torch.where(torch.isnan(self.x).sum(dim=1, keepdim=False) <= 2, True, False).numpy().tolist()
        print(len(mask))
        self.x = torch.tensor(self.x.numpy()[mask])

        self.x = torch.where(torch.isnan(self.x), torch.full_like(self.x, 0), self.x)
        self.x = torch.where(torch.isinf(self.x), torch.full_like(self.x, 0), self.x)
        self.x = torch.where(self.x > 10, torch.full_like(self.x, 0), self.x)
        self.x = torch.where(self.x < -10, torch.full_like(self.x, 0), self.x)
        pd.DataFrame(self.x.numpy()).head(1000).to_csv('train.csv')

        self.y = torch.tensor(self.y.numpy()[mask])
        self.y = torch.where(torch.isnan(self.y), torch.full_like(self.y, 0), self.y)
        self.y = torch.where(torch.isinf(self.y), torch.full_like(self.y, 0), self.y)

        self.z = torch.tensor(self.z.numpy()[mask])
        self.z = torch.where(torch.isnan(self.z), torch.full_like(self.z, 0), self.z)
        self.z = torch.where(torch.isinf(self.z), torch.full_like(self.z, 0), self.z)

        self.date = np.array(d['date'])[mask].tolist()
        self.stocks = np.array(d['stocks'])[mask].tolist()

        tmp_df = pd.DataFrame({'date':self.date, 'return':self.y.numpy().tolist()})
        tmp_df['dmean'] = tmp_df.groupby(['date'])['return'].transform('mean')
        tmp_df['adj_return'] = tmp_df['return'] - tmp_df['dmean']
        print(tmp_df)
        self.y = torch.tensor(tmp_df.adj_return.to_numpy())


        self.y = torch.log(self.y + 1.0)


        tmp_df = pd.DataFrame({'date':self.date, 'return':self.z.numpy().tolist()})
        tmp_df['dmean'] = tmp_df.groupby(['date'])['return'].transform('mean')
        tmp_df['adj_return'] = tmp_df['return'] - tmp_df['dmean']
        print(tmp_df)
        self.z = torch.tensor(tmp_df.adj_return.to_numpy())
        self.z = torch.log(self.z + 1.0)

        #self.y = torch.where(self.y > 0.08, torch.full_like(self.y, 0.08), self.y)
        self.f_len = len(self.y)
        self.idx = 0

        self.today_close = np.array(d['today_close'])[mask].tolist()
        self.after_open = np.array(d['after_open'])[mask].tolist()
        self.halfday_close = np.array(d['halfday_close'])[mask].tolist()
        print(self.x.shape, self.y.shape, self.y, self.date[:10], self.stocks[:10], mask[:10], flush=True)


    def __len__(self):
        return self.f_len

    def __getitem__(self, idx):
        return self.x[i], self.y[i]

    def get_batch_data(self, batch_size=-1):
        if self.idx == 0:
            indices = torch.randperm(self.f_len)
            self.x = self.x[indices]
            self.y = self.y[indices]
            self.y = self.y.to(self.device)
            self.y[torch.isnan(self.y)] = 0
            y = torch.flatten(self.y)
            y = torch.round(y*100)
            y = torch.clip(y, -20, 20)
            y = (y + 20).to(torch.int64)
            y = F.one_hot(y)
            self.y = y.to(torch.float32)
            print(self.y.shape)


            self.z = self.z[indices]
            self.z = self.z.to(self.device)
            self.z[torch.isnan(self.z)] = 0
            z = torch.flatten(self.z)
            z = torch.round(z*100)
            z = torch.clip(z, -20, 20)
            z = (z + 20).to(torch.int64)
            z = F.one_hot(z)
            self.z = z.to(torch.float32)
            print(self.z.shape)
            #print(self.y[torch.isnan(self.y)]
            #print(len(self.y[torch.isnan(self.y)]))
        s = self.idx * batch_size
        e = s + batch_size
        #print(s,e)

        if self.f_len % batch_size == 0:
            self.idx = (self.idx + 1) % ( self.f_len // batch_size ) 
        else:
            self.idx = (self.idx + 1) % ( (self.f_len // batch_size) + 1 ) 

        #self.idx = (self.idx + 1) % ( self.f_len // batch_size )
        return self.x[s:e], self.y[s:e], self.z[s:e], self.today_close[s:e], self.after_open[s:e], self.halfday_close[s:e]

class StockData_mini_batch_tensor_3part_2tar_pre(data.Dataset):
    def __init__(self, fn, file_len=0, device=torch.device('cpu')):
        self.device = device
        #w = self.gbm_simulation(self.gbm_param)
        fp = open(fn, 'rb')
        d = pickle.load(fp)
        self.x = d['x'].to(device)
        self.y = d['y'].to(device)
        self.z = d['z'].to(device)
        #print(self.x.shape, self.y.shape, flush=True)
        maskx = torch.where(torch.isnan(self.x).sum(dim=1, keepdim=False) <= 20, True, False)
        #maskx = torch.where((~torch.isnan(self.x)).sum(dim=1, keepdim=False) >= 2, True, False)
        masky = torch.where(torch.isnan(self.y), False, True)
        maskz = torch.where(torch.isnan(self.z), False, True)
        mask = (maskx & masky & maskz).numpy().tolist()
        #mask = torch.where(torch.isnan(self.x).sum(dim=1, keepdim=False) <= 2, True, False).numpy().tolist()
        #print(len(mask))
        self.x = torch.tensor(self.x.numpy()[mask])

        self.x = torch.where(torch.isnan(self.x), torch.full_like(self.x, 0), self.x)
        self.x = torch.where(torch.isinf(self.x), torch.full_like(self.x, 0), self.x)
        self.x = torch.where(self.x > 10, torch.full_like(self.x, 0), self.x)
        self.x = torch.where(self.x < -10, torch.full_like(self.x, 0), self.x)
        #pd.DataFrame(self.x.numpy()).head(1000).to_csv('train.csv')

        self.y = torch.tensor(self.y.numpy()[mask])
        self.y = torch.where(torch.isnan(self.y), torch.full_like(self.y, 0), self.y)
        self.y = torch.where(torch.isinf(self.y), torch.full_like(self.y, 0), self.y)

        self.z = torch.tensor(self.z.numpy()[mask])
        self.z = torch.where(torch.isnan(self.z), torch.full_like(self.z, 0), self.z)
        self.z = torch.where(torch.isinf(self.z), torch.full_like(self.z, 0), self.z)

        self.date = np.array(d['date'])[mask].tolist()
        self.stocks = np.array(d['stocks'])[mask].tolist()

        #tmp_df = pd.DataFrame({'date':self.date, 'return':self.y.numpy().tolist()})
        #tmp_df['dmean'] = tmp_df.groupby(['date'])['return'].transform('mean')
        #tmp_df['adj_return'] = tmp_df['return'] - tmp_df['dmean']
        #print(tmp_df)
        #self.y = torch.tensor(tmp_df.adj_return.to_numpy())


        self.y = torch.log(self.y + 1.0)


        #tmp_df = pd.DataFrame({'date':self.date, 'return':self.z.numpy().tolist()})
        #tmp_df['dmean'] = tmp_df.groupby(['date'])['return'].transform('mean')
        #tmp_df['adj_return'] = tmp_df['return'] - tmp_df['dmean']
        #print(tmp_df)
        #self.z = torch.tensor(tmp_df.adj_return.to_numpy())
        self.z = torch.log(self.z + 1.0)

        #self.y = torch.where(self.y > 0.08, torch.full_like(self.y, 0.08), self.y)
        self.f_len = len(self.y)
        self.idx = 0

        self.today_close = np.array(d['today_close'])[mask].tolist()
        self.after_open = np.array(d['after_open'])[mask].tolist()
        self.halfday_close = np.array(d['halfday_close'])[mask].tolist()
        #print(self.x.shape, self.y.shape, self.y, self.date[:10], self.stocks[:10], mask[:10], flush=True)


    def __len__(self):
        return self.f_len

    def __getitem__(self, idx):
        return self.x[i], self.y[i]

    def get_batch_data(self, batch_size=-1):
        if self.idx == 0:
            indices = torch.randperm(self.f_len)
            self.x = self.x[indices]
            self.y = self.y[indices]
            self.y = self.y.to(self.device)
            self.y[torch.isnan(self.y)] = 0
            y = torch.flatten(self.y)
            y = torch.round(y*100)
            y = torch.clip(y, -20, 20)
            y = (y + 20).to(torch.int64)
            y = F.one_hot(y)
            self.y = y.to(torch.float32)
            print(self.y.shape)


            self.z = self.z[indices]
            self.z = self.z.to(self.device)
            self.z[torch.isnan(self.z)] = 0
            z = torch.flatten(self.z)
            z = torch.round(z*100)
            z = torch.clip(z, -20, 20)
            z = (z + 20).to(torch.int64)
            z = F.one_hot(z)
            self.z = z.to(torch.float32)
            print(self.z.shape)
            #print(self.y[torch.isnan(self.y)]
            #print(len(self.y[torch.isnan(self.y)]))
        s = self.idx * batch_size
        e = s + batch_size
        #print(s,e)

        if self.f_len % batch_size == 0:
            self.idx = (self.idx + 1) % ( self.f_len // batch_size ) 
        else:
            self.idx = (self.idx + 1) % ( (self.f_len // batch_size) + 1 ) 

        #self.idx = (self.idx + 1) % ( self.f_len // batch_size )
        return self.x[s:e], self.y[s:e], self.z[s:e], self.today_close[s:e], self.after_open[s:e], self.halfday_close[s:e]








if __name__ == '__main__' :
    # 测试代码
    # import time
    # data_path = '/home/wanghexiang/dnn_rank/'
    dataset = StockData_mini_batch_day_inday_tensor('Ashares2train_tushare.pickle', 'Ashares2train_tushare_half.pickle', '20110110', '20210330', 'init', 128)
    # t2 = 0.0

    # for i in range(5000):
    #     start = time.time()
    #     data = dataset.get_batch_data(128)
    #     end = time.time()
    #     t2 += end -start
    #     print(end - start)
    pass
