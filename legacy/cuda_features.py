import os
import time
import glob
import torch as t
import torch.nn.functional as F
import sys
import cupy as cp
import numpy as np
from operator import itemgetter
import pickle5 as pickle
import gzip
import h5py
from scipy.stats import rankdata
import pandas as pd
from multiprocessing import Process, Queue, cpu_count, Array, Lock, Manager
import multiprocessing

def corrcoef(data1, data2):
    #return cp.where(((data1 - cp.mean(data1)).any() & (data2 - cp.mean(data2)).any()), cp.corrcoef(data1, data2)[0, 1], 0.0)
    return cp.where(((data1 == data1[0]).all() | (data2 == data2[0]).all()), 0.0, cp.corrcoef(data1, data2)[0, 1])

def safe_divide(data1, data2, default=0.0):
    return cp.where(data2 == 0.0, 0.0, data1 / data2)

def safe_log(data, fld):
    if fld == 8 or fld == 5:
        #return cp.log(data + 10)
        return data + 0.01

def safe_log2(data, fld):
    if fld == 8 or fld == 5:
        #return cp.log2(data + 10)
        return data + 0.01

def safe_log_opt(data, fld):
    if fld == 8 or fld == 5:
        #return cp.log(data + 10)
        return data + 0.01

dtype = cp.float32

class Factors():

    def __init__(self, pickle_file=None, ochl_file=None, device=None):
        if pickle_file:
            with open(pickle_file, 'rb') as f:
                self.all_stocks, self.all_dates, _, self.all_masks, self.all_raw_features, _, _ = pickle.load(f)
            print('stats:', len(self.all_stocks), len(self.all_dates), self.all_masks.shape, self.all_raw_features.shape, flush=True)
            self.all_masks = t.from_numpy(self.all_masks).to(device)
            self.all_raw_features = t.from_numpy(self.all_raw_features).to(device)
            print('Conversion from numpy to cupy finished.', flush=True)
        elif ochl_file:
            self.load_ochl(ochl_file, device)
            self.calc_basedata()
            #self.extract_features('inc_close_100', 'CR')
            #self.extract_features('inc_open_100', 'OR')
            self.df.set_index(keys=['S_INFO_WINDCODE', 'TRADE_DT'], drop=False, inplace=True)

    def load_ochl(self, ochl_fn, device):
        #names = ['TRADE_DT', 'S_INFO_WINDCODE', 'S_FWDS_ADJOPEN', 'S_FWDS_ADJCLOSE', 'S_FWDS_ADJHIGH', 'S_FWDS_ADJLOW', 'S_DQ_AVGPRICE', 'S_DQ_VOLUME',  'FHZS_FLAG', 'LIMIT_UP', 'LIMIT_DOWN']
        names = ['TRADE_DT', 'S_INFO_WINDCODE', 'S_FWDS_ADJOPEN', 'S_FWDS_ADJCLOSE', 'S_FWDS_ADJHIGH', 'S_FWDS_ADJLOW', 'S_DQ_AVGPRICE', 'S_DQ_VOLUME',  'FHZS_FLAG', 'LIMIT_UP', 'LIMIT_DOWN', 'LAST_CLOSE']
        # dtype = {'TRADE_DT': np.str_, 'S_INFO_WINDCODE': np.str_, 'S_FWDS_ADJOPEN': np.float32, 'S_FWDS_ADJCLOSE': np.float32, 'S_FWDS_ADJHIGH': np.float32,
        #         'S_FWDS_ADJLOW': np.float32, 'S_DQ_AVGPRICE': np.float32, 'S_DQ_VOLUME': np.float32, 'FHZS_FLAG': np.str_, 'LIMIT_UP': np.float32, 'LIMIT_DOWN': np.float32}
        dtype = {'TRADE_DT': np.str_, 'S_INFO_WINDCODE': np.str_, 'S_FWDS_ADJOPEN': np.float32, 'S_FWDS_ADJCLOSE': np.float32, 'S_FWDS_ADJHIGH': np.float32,
                 'S_FWDS_ADJLOW': np.float32, 'S_DQ_AVGPRICE': np.float32, 'S_DQ_VOLUME': np.float32, 'FHZS_FLAG': np.str_, 'LIMIT_UP': np.float32, 'LIMIT_DOWN': np.float32, 'LAST_CLOSE': np.float32}
        ochl_fn_pickle = ochl_fn + '.pickle'
        #if os.path.exists(ochl_fn_pickle):
        #    print('loading from cache:', ochl_fn_pickle, flush=True)
        #    with open(ochl_fn_pickle, 'rb') as f:
        #        self.__dict__.update(pickle.load(f))
        #else:
        if 1:
            df = pd.read_csv(ochl_fn, header=None, sep='\t', names=names, index_col=False, dtype=dtype)
            df.S_INFO_WINDCODE = df.S_INFO_WINDCODE.str[:6]
            df['S_DQ_AMOUNT'] = df.S_DQ_VOLUME * df.S_DQ_AVGPRICE
            df.drop_duplicates(subset=['TRADE_DT','S_INFO_WINDCODE'], keep='first', inplace=True)
            df.sort_values(by=['S_INFO_WINDCODE', 'TRADE_DT'], ascending=True, inplace=True)
            print(df)
            #df.LAST_CLOSE = df.S_FWDS_ADJCLOSE.shift(1)
            self.df = df

            self.S_FWDS_ADJOPEN = df.pivot(index='TRADE_DT', columns=['S_INFO_WINDCODE'], values='S_FWDS_ADJOPEN')
            self.all_stocks = self.S_FWDS_ADJOPEN.columns.values.tolist()
            self.all_stocks = { self.all_stocks[i]:i for i in range(len(self.all_stocks)) }
            #print('all_stocks:', len(self.all_stocks), self.all_stocks, flush=True, file=sys.stderr)
            self.all_dates = self.S_FWDS_ADJOPEN.index.values.tolist()
            self.all_dates = { self.all_dates[i]:i for i in range(len(self.all_dates)) }
            #print('all_dates:', len(self.all_dates), self.all_dates, flush=True, file=sys.stderr)


            self.S_FWDS_ADJCLOSE = df.pivot(index='TRADE_DT', columns=['S_INFO_WINDCODE'], values='S_FWDS_ADJCLOSE')
            self.S_FWDS_ADJHIGH = df.pivot(index='TRADE_DT', columns=['S_INFO_WINDCODE'], values='S_FWDS_ADJHIGH')
            self.S_FWDS_ADJLOW = df.pivot(index='TRADE_DT', columns=['S_INFO_WINDCODE'], values='S_FWDS_ADJLOW')
            self.S_DQ_AVGPRICE = df.pivot(index='TRADE_DT', columns=['S_INFO_WINDCODE'], values='S_DQ_AVGPRICE')
            self.S_DQ_VOLUME = df.pivot(index='TRADE_DT', columns=['S_INFO_WINDCODE'], values='S_DQ_VOLUME')
            self.S_DQ_AMOUNT = df.pivot(index='TRADE_DT', columns=['S_INFO_WINDCODE'], values='S_DQ_AMOUNT')
            self.TRADE_DT = df.pivot(index='TRADE_DT', columns=['S_INFO_WINDCODE'], values='TRADE_DT')
            self.LIMIT_DOWN = df.pivot(index='TRADE_DT', columns=['S_INFO_WINDCODE'], values='LIMIT_DOWN')
            self.LIMIT_UP = df.pivot(index='TRADE_DT', columns=['S_INFO_WINDCODE'], values='LIMIT_UP')
            self.LAST_CLOSE = df.pivot(index='TRADE_DT', columns=['S_INFO_WINDCODE'], values='LAST_CLOSE')
            print(self.S_FWDS_ADJCLOSE)
            print('check:', self.S_DQ_VOLUME.loc['20210104'])

            self.MASK = t.tensor(~np.isnan(self.S_FWDS_ADJCLOSE.to_numpy(dtype=np.float32))).to(device)
            self.S_FWDS_ADJOPEN = t.tensor(self.S_FWDS_ADJOPEN.to_numpy(dtype=np.float32)).to(device).masked_fill_(~self.MASK, t.tensor(0.0).to(device))  # [date, stock]
            self.S_FWDS_ADJCLOSE = t.tensor(self.S_FWDS_ADJCLOSE.to_numpy(dtype=np.float32)).to(device).masked_fill_(~self.MASK, t.tensor(0.0).to(device))
            self.S_FWDS_ADJHIGH = t.tensor(self.S_FWDS_ADJHIGH.to_numpy(dtype=np.float32)).to(device).masked_fill_(~self.MASK, t.tensor(0.0).to(device))
            self.S_FWDS_ADJLOW = t.tensor(self.S_FWDS_ADJLOW.to_numpy(dtype=np.float32)).to(device).masked_fill_(~self.MASK, t.tensor(0.0).to(device))
            self.S_DQ_AVGPRICE = t.tensor(self.S_DQ_AVGPRICE.to_numpy(dtype=np.float32)).to(device).masked_fill_(~self.MASK, t.tensor(0.0).to(device))
            self.S_DQ_VOLUME = t.tensor(self.S_DQ_VOLUME.to_numpy(dtype=np.float32)).to(device).masked_fill_(~self.MASK, t.tensor(0.0).to(device))
            self.S_DQ_AMOUNT = t.tensor(self.S_DQ_AMOUNT.to_numpy(dtype=np.float32)).to(device).masked_fill_(~self.MASK, t.tensor(0.0).to(device))
            #self.TRADE_DT = t.tensor(self.TRADE_DT.to_numpy(dtype=np.float32)).to(device).masked_fill_(~self.MASK, t.tensor(0.0).to(device))
            self.LIMIT_DOWN = t.tensor(self.LIMIT_DOWN.to_numpy(dtype=np.float32)).to(device).masked_fill_(~self.MASK, t.tensor(0.0).to(device))
            self.LIMIT_UP = t.tensor(self.LIMIT_UP.to_numpy(dtype=np.float32)).to(device).masked_fill_(~self.MASK, t.tensor(0.0).to(device))
            self.LAST_CLOSE = t.tensor(self.LAST_CLOSE.to_numpy(dtype=np.float32)).to(device)

            #df.to_pickle(ochl_fn_pickle)
            with open(ochl_fn_pickle, 'wb') as f:
                pickle.dump(self.__dict__, f, 2)

        print(self.df, flush=True, file=sys.stderr)
        self.load_test(self.df, device)

        #self.fealist = []
        #with open('allfea_del8.ini') as file:
        #    lines = file.readlines()
        #    self.fealist = [line.rstrip() for line in lines]

    def calc_basedata(self):
        self.df['inc_close'] = self.df.S_FWDS_ADJCLOSE / self.df.LAST_CLOSE - 1
        self.df['inc_open'] = self.df.S_FWDS_ADJOPEN / self.df.LAST_CLOSE - 1
        self.df['inc_high'] = self.df.S_FWDS_ADJHIGH / self.df.LAST_CLOSE - 1
        self.df['inc_avg'] = self.df.S_DQ_AVGPRICE / self.df.LAST_CLOSE - 1
        self.df['inc_low'] = self.df.S_FWDS_ADJLOW / self.df.LAST_CLOSE - 1

        self.df['inc_close'].where(self.df['inc_close'] < 0.1, 0.1, inplace=True)
        self.df['inc_close'].where(self.df['inc_close'] > -0.1, -0.1, inplace=True)
        self.df['inc_close_100'] = (self.df['inc_close'] * 100).astype(np.int32)

        self.df['inc_open'].where(self.df['inc_open'] < 0.1, 0.1, inplace=True)
        self.df['inc_open'].where(self.df['inc_open'] > -0.1, -0.1, inplace=True)
        self.df['inc_open_100'] = (self.df['inc_open'] * 100).astype(np.int32)

        # CS: cross section
        self.df['cs_rank_close'] = self.df.groupby([self.df.TRADE_DT])['inc_close'].rank(ascending=False, pct=True)
        self.df['cs_rank_open'] = self.df.groupby([self.df.TRADE_DT])['inc_open'].rank(ascending=False, pct=True)
        self.df['cs_rank_high'] = self.df.groupby([self.df.TRADE_DT])['inc_high'].rank(ascending=False, pct=True)
        self.df['cs_rank_avg'] = self.df.groupby([self.df.TRADE_DT])['inc_avg'].rank(ascending=False, pct=True)
        self.df['cs_rank_low'] = self.df.groupby([self.df.TRADE_DT])['inc_low'].rank(ascending=False, pct=True)
        self.df['cs_rank_amount'] = self.df.groupby([self.df.TRADE_DT])['S_DQ_AMOUNT'].rank(ascending=False, pct=True)

    def fill_masked_with_last(self, x, mask):
        #print('fill_masked_with_last x, mask:', x.shape, mask.shape, x[:, 3742], mask[:, 3742], flush=True, file=sys.stderr)
        x = t.gather(x, 0, (t.arange(x.shape[0], device=x.device).unsqueeze(-1) * mask).cummax(dim=0).values)
        m = t.gather(mask, 0, (t.arange(mask.shape[0], device=mask.device).unsqueeze(-1) * mask).cummax(dim=0).values)
        #print('fill_masked_with_last2 x, m:', x.shape, m.shape, x[:, 3742], m[:, 3742], flush=True, file=sys.stderr)
        return x, m

    def rank(self, x, mask, dim, copy=True, descending=False):
        x_shape = x.shape
        window = x_shape[dim]
        if copy:
            if descending:
                x = x.masked_fill(~mask, float('-inf'))
            else:
                x = x.masked_fill(~mask, float('inf'))
        else:
            if descending:
                x.masked_fill_(~mask, float('-inf'))
            else:
                x.masked_fill_(~mask, float('inf'))
        #print('rank mask:', mask.shape, mask[:, 129], flush=True, file=sys.stderr)
        x, i = t.sort(x, dim=dim, descending=descending)
        #print('rank x, i:', x.shape, i.shape, x[:, 129], i[:, 129], flush=True, file=sys.stderr)
        if dim == 2:
            assert len(x_shape) == 3
            vv = (F.pad(x[:, :, 1:] - x[:, :, :-1], (1, 0, 0, 0, 0, 0)) > 0).cumsum(dim=dim)
            ar = t.arange(1, window+1, device=x.device).expand(vv.shape)
        elif dim == 1:
            assert len(x_shape) == 2
            vv = (F.pad(x[:, 1:] - x[:, :-1], (1, 0, 0, 0)) > 0).cumsum(dim=dim)
            ar = t.arange(1, window+1, device=x.device).expand(vv.shape)
        elif dim == 0 and len(x_shape) == 2:
            vv = (F.pad(x[1:, :] - x[:-1, :], (0, 0, 1, 0)) > 0).cumsum(dim=dim)
            ar = t.arange(1, window+1, device=x.device).unsqueeze(1).expand(vv.shape)
        elif dim == 0 and len(x_shape) == 1:
            if descending:
                vv = (F.pad(x[1:] - x[:-1], (1, 0)) < 0).cumsum(dim=dim)
            else:
                vv = (F.pad(x[1:] - x[:-1], (1, 0)) > 0).cumsum(dim=dim)
            ar = t.arange(1, window+1, device=x.device)
        else:
            assert False
        #print('rank vv:', vv.shape, vv[:, 129], flush=True, file=sys.stderr)
        x = t.zeros_like(vv).scatter_add(dim, vv, ar) / t.zeros_like(vv).scatter_add(dim, vv, t.ones_like(vv))
        #print('rank x:', x.shape, x[:, 129], flush=True, file=sys.stderr)
        x = t.gather(x, dim, vv)
        #print('rank x:', x.shape, x[:, 129], flush=True, file=sys.stderr)
        x = t.zeros_like(x).scatter(dim, i,  x)
        #print('rank x:', x.shape, x[:, 129], flush=True, file=sys.stderr)
        x = (x / mask.sum(dim=dim, keepdim=True))
        #print('rank x:', x.shape, x[:, 129], flush=True, file=sys.stderr)
        if dim == 2:
            return x[:, :, -1], mask.all(dim=2)
        elif dim == 0 and len(x_shape) == 2:
            return x[-1, :], mask.all(dim=0)
        else:
            return x, mask

    def sum(self, x, mask, dim, copy=True):
        if copy:
            x = x.masked_fill(~mask, 0.0)
        else:
            x.masked_fill_(~mask, 0.0)
        return x.sum(dim=dim), mask.all(dim=dim)

    def mean(self, x, mask, dim, copy=True):
        if copy:
            x = x.masked_fill(~mask, 0.0)
        else:
            x.masked_fill_(~mask, 0.0)
        return x.sum(dim=dim) / mask.sum(dim=dim), mask.all(dim=dim)

    def min(self, x, mask, dim, copy=True):
        if copy:
            x = x.masked_fill(~mask, float('inf'))
        else:
            x.masked_fill_(~mask, float('inf'))
        return x.min(dim=dim).values, mask.all(dim=dim)

    def max(self, x, mask, dim, copy=True):
        if copy:
            x = x.masked_fill(~mask, float('-inf'))
        else:
            x.masked_fill_(~mask, float('-inf'))
        return x.max(dim=dim).values, mask.all(dim=dim)

    def std(self, x, mask, dim, copy=True):
        if copy:
            x = x.masked_fill(~mask, 0.0)
        else:
            x.masked_fill_(~mask, 0.0)
        m = mask.sum(dim=dim)
        return t.sqrt(((x ** 2).sum(dim=dim) - ((x.sum(dim=dim) ** 2) / m)) / (m - 1)), mask.all(dim=dim)

    def corr(self, x1, x2, mask, dim, copy=True):
        if copy:
            x1 = x1.masked_fill(~mask, 0.0)
            x2 = x2.masked_fill(~mask, 0.0)
        else:
            x1.masked_fill_(~mask, 0.0)
            x2.masked_fill_(~mask, 0.0)
        m = mask.sum(dim=dim, keepdim=True)
        x1 = x1 - (x1.sum(dim=dim, keepdim=True) / m)
        x2 = x2 - (x2.sum(dim=dim, keepdim=True) / m)
        m = mask.all(dim=dim)
        x1.masked_fill_(~m, 0)
        x2.masked_fill_(~m, 0)
        x = (x1 * x2).sum(dim=dim) / t.sqrt((x1 ** 2).sum(dim=dim) * (x2 ** 2).sum(dim=dim))
        return x, m

    def cov(self, x1, x2, mask, dim, copy=True):
        if copy:
            x1 = x1.masked_fill(~mask, 0.0)
            x2 = x2.masked_fill(~mask, 0.0)
        else:
            x1.masked_fill_(~mask, 0.0)
            x2.masked_fill_(~mask, 0.0)
        cnt = mask.sum(dim=dim, keepdim=True)
        x1 = x1 - (x1.sum(dim=dim, keepdim=True) / cnt)
        x2 = x2 - (x2.sum(dim=dim, keepdim=True) / cnt)
        m = mask.all(dim=dim)
        x1.masked_fill_(~m, 0)
        x2.masked_fill_(~m, 0)
        x = (x1 * x2).sum(dim=dim) / (cnt.squeeze(dim) - 1)
        return x, m

    def lowday(self, x, mask, copy=True):
        if copy:
            x = x.masked_fill(~mask, float('inf'))
        else:
            x.masked_fill_(~mask, float('inf'))
        x = len(x) - x.argmin(dim=0) # [stock]
        return x, mask.all(dim=0)

    #def ewma(self, feature, alpha):
    #    alpha_rev = 1-alpha
    #    n = feature.shape[0]
    #    pows = alpha_rev**(t.arange(n+1, dtype=feature.dtype, device=feature.device))
    #    scale_arr = 1/pows[:-1]
    #    offset = feature[0:1]*pows[1:].unsqueeze(-1)
    #    pw0 = alpha*(alpha_rev**(n-1))
    #    mult = feature*pw0*scale_arr.unsqueeze(-1)
    #    cumsums = mult.cumsum(dim=0)
    #    out = offset + cumsums*t.flip(scale_arr, dims=[0]).unsqueeze(-1)
    #    return out

    def ewma(self, feature, mask, alpha, dtype=t.float64):
        alpha_rev = t.tensor(1 - alpha, dtype=dtype, device=feature.device)
        n = feature.shape[0]  # [date, stock]
        pows = alpha_rev ** t.arange(n, dtype=dtype, device=feature.device).unsqueeze(-1)  # [date, 1]
        #print('ewma, pows:', pows.shape, pows.dtype, pows, flush=True, file=sys.stderr)
        pw0 = alpha_rev ** (n - 1) # []
        #print('ewma, pw0:', pw0.dtype, pw0, flush=True, file=sys.stderr)
        scale_arr = pw0 / pows  # [date, 1]
        #print('ewma, scale_arr:', scale_arr.shape, scale_arr.dtype, scale_arr, flush=True, file=sys.stderr)
        scale_arr_rev = t.flip(1 / pows, dims=[0])
        #print('ewma, scale_arr_rev:', scale_arr_rev.shape, scale_arr_rev.dtype, scale_arr_rev, flush=True, file=sys.stderr)
        #print('ewma 0:', feature.shape, mask.shape, feature[:, 8], mask[:, 8], file=sys.stderr, flush=True)
        mult = feature * scale_arr  # [date, stock]
        #print('ewma, mult:', mult.shape, mult.dtype, mult, flush=True, file=sys.stderr)
        div_mult = mask.float() * scale_arr  # [date, stock]
        #print('ewma, div_mult:', div_mult.shape, div_mult.dtype, div_mult, flush=True, file=sys.stderr)
        #print('mult 0:', mult.shape, div_mult.shape, mult[:, 8], div_mult[:, 8], file=sys.stderr, flush=True)
        mult.masked_fill_(~mask, 0.0)
        div_mult.masked_fill_(~mask, 0.0)
        #print('mult 1:', mult.shape, div_mult.shape, mult[:, 8], div_mult[:, 8], file=sys.stderr, flush=True)
        cumsums = mult.cumsum(dim=0) * scale_arr_rev  # [date, stock]
        #print('ewma, cumsums:', cumsums.shape, cumsums.dtype, cumsums, flush=True, file=sys.stderr)
        div_cumsums = div_mult.cumsum(dim=0) * scale_arr_rev  # [date, stock]
        #print('ewma, div_cumsums:', div_cumsums.shape, div_cumsums.dtype, div_cumsums, flush=True, file=sys.stderr)
        #print('cumsums 0:', cumsums.shape, div_cumsums.shape, cumsums[:, 8], div_cumsums[:, 8], file=sys.stderr, flush=True)
        out = cumsums / div_cumsums  # [date, stock]
        out = out.type(feature.dtype)
        #print('ewma, out:', out.shape, out.dtype, out, flush=True, file=sys.stderr)
        #print('out 0:', out.shape, out[:, 8], file=sys.stderr, flush=True)
        return out

    def pct_change(self, feature, mask):
        x, m = self.fill_masked_with_last(feature, mask)
        #x, m = feature, mask
        x = x[1:] / x[:-1] - 1
        m = m[1:] & m[:-1]
        return x, m

    def rolling_sum(self, feature, mask, window):
        x = feature.unfold(0, window, 1)
        m = mask.unfold(0, window, 1)
        x, m = self.sum(x, m, 2, copy=False)
        return x, m

    def rolling_mean(self, feature, mask, window):
        x = feature.unfold(0, window, 1)
        m = mask.unfold(0, window, 1)
        x, m = self.mean(x, m, 2, copy=False)
        return x, m

    def rolling_min(self, feature, mask, window):
        x = feature.unfold(0, window, 1)
        m = mask.unfold(0, window, 1)
        x, m = self.min(x, m, 2, copy=False)
        return x, m

    def rolling_max(self, feature, mask, window):
        x = feature.unfold(0, window, 1)
        m = mask.unfold(0, window, 1)
        x, m = self.max(x, m, dim=2, copy=False)
        return x, m

    def rolling_std(self, feature, mask, window):
        x = feature.unfold(0, window, 1)
        m = mask.unfold(0, window, 1)
        x, m = self.std(x, m, dim=2, copy=False)
        return x, m

    def rolling_tsrank(self, feature, mask, window):
        x = feature.unfold(0, window, 1) # [date, stock, window]
        m = mask.unfold(0, window, 1) # [date, stock, window]
        x, m = self.rank(x, m, dim=2, copy=False)
        return x, m

    def rg(self, feature, pos, window, length):
        pos = pos + 1
        start = pos - (window + length - 1)
        start = 0 if start < 0 else start
        return feature[start:pos, :] # [date, stock]

    def bn(self, feature, pos, length):
        pos = pos + 1
        start = pos - length
        start = 0 if start < 0 else start
        return feature[start:pos, :] # [length, stock]

    def eg(self, feature, pos, window):
        #length = 200
        length = 66
        #length = 65 + 20
        pos = pos + 1
        #start = pos - (window + length - 1)
        #start = 0 if start < 0 else start
        #return feature[start:pos, :] # [length, stock]
        start = pos - length
        start = 0 if start < 0 else start
        return F.pad(feature[start:pos, :], (0, 0, window-1, 0), value=False) # [length, stock]

    def en(self, feature, pos):
        #length = 200
        length = 66
        #length = 65 + 20
        pos = pos + 1
        start = pos - length
        start = 0 if start < 0 else start
        return feature[start:pos, :] # [length, stock]

    def ro(self, mask, pos, length):
        pos = pos + 1
        start = pos - length
        start = 0 if start < 0 else start
        mask = mask[start:pos, :]
        #print('ro mask0:', mask.shape, mask[:, 129], flush=True, file=sys.stderr)
        mask = mask.all(dim=0, keepdim=True)
        #print('ro mask1:', mask.shape, mask[:, 129], flush=True, file=sys.stderr)
        mask = mask.expand(length, -1)
        #print('ro mask2:', mask.shape, mask[:, 129], flush=True, file=sys.stderr)
        return  mask # [length, stock]

    def process_nan_infinite_and_mask(self, x, mask):
        return mask & t.isfinite(x)


    def p(self, feature, pos):
        return feature[pos] # [stock]

    def best_014(self, pos):
        ts_volume = self.rg(self.S_DQ_VOLUME, pos, 7, 5)
        #print('best_014 0:', ts_volume.shape, ts_volume[:, 1107], flush=True)
        ts_volume, ts_volume_mask = self.rolling_tsrank(ts_volume, self.rg(self.MASK, pos, 7, 5), 7) # [5, stocks]
        #print('best_014 1:', ts_volume.shape, ts_volume_mask.shape, ts_volume[:, 1107], ts_volume_mask[:, 1107], flush=True)
        ts_high = ((self.rg(self.S_FWDS_ADJHIGH, pos, 7, 5) - self.rg(self.S_FWDS_ADJLOW, pos, 7, 5)) / self.rg(self.S_FWDS_ADJCLOSE, pos, 7, 5))
        #print('best_014 2:', ts_high.shape, ts_high[:, 1107], flush=True)
        ts_high, ts_high_mask = self.rolling_tsrank(ts_high, self.rg(self.MASK, pos, 7, 5), 7) #[5, stocks]
        #print('best_014 3:', ts_high.shape, ts_high_mask.shape, ts_high[:, 1107], ts_high_mask[:, 1107], flush=True)
        corr, corr_mask = self.corr(ts_volume, ts_high, (ts_volume_mask & ts_high_mask), 0) 
        #print('best_014 4:', corr.shape, corr_mask.shape, corr[1107], corr_mask[1107], flush=True)
        corr.masked_fill_(~corr_mask, float('nan'))
        #print('corr:', corr.shape, corr[0:10].cpu().numpy(), flush=True, file=sys.stderr)
        return corr, corr_mask

    def extra_005(self, pos):
        data1, data1_mask = self.rolling_mean(self.rg(self.S_FWDS_ADJCLOSE, pos, 20, 22), self.rg(self.MASK, pos, 20, 22), 20) # [22, stock]
        data1 = (self.bn(self.S_FWDS_ADJHIGH, pos, 22) - self.bn(self.S_FWDS_ADJLOW, pos, 22)) / data1  # [22, stock]
        rank1, rank1_mask = self.rank(data1[:-2], data1_mask[:-2], dim=0)  # [stock]
        rank2, rank2_mask = self.rank(self.bn(self.S_DQ_VOLUME, pos, 20), self.bn(self.MASK, pos, 20), dim=0) # [stock]
        data2 = (data1[-1] / (self.p(self.S_DQ_AVGPRICE, pos) - self.p(self.S_FWDS_ADJCLOSE, pos))) # [stock]
        alpha = (rank1 * rank2) / data2 # [stock]
        #mask = (data1_mask[-3] & self.p(self.MASK, pos))
        mask = rank1_mask & rank2_mask & data1_mask[-1] & self.p(self.MASK, pos)
        alpha.masked_fill_(~mask, float('nan'))
        return alpha, mask

    def best_001(self, pos):
        part = -((2 * self.bn(self.S_FWDS_ADJCLOSE, pos, 5) - self.bn(self.S_FWDS_ADJLOW, pos, 5) - self.bn(self.S_FWDS_ADJHIGH, pos, 5)) / (self.bn(self.S_FWDS_ADJHIGH, pos, 5) - self.bn(self.S_FWDS_ADJLOW, pos, 5))) # [5, stock]
        part_mask = self.process_nan_infinite_and_mask(part, self.bn(self.MASK, pos, 5))
        alpha, alpha_mask = self.rank(part, part_mask, dim=0) # [stock]
        mask = alpha_mask & self.p(self.MASK, pos)
        alpha.masked_fill_(~mask, float('nan'))
        return alpha, mask

    def best_002(self, pos):
        part = -((2 * self.bn(self.S_FWDS_ADJCLOSE, pos, 10) - self.bn(self.S_FWDS_ADJLOW, pos, 10) - self.bn(self.S_FWDS_ADJHIGH, pos, 10)) / (self.bn(self.S_FWDS_ADJHIGH, pos, 10) - self.bn(self.S_FWDS_ADJLOW, pos, 10)))
        part_mask = self.process_nan_infinite_and_mask(part, self.bn(self.MASK, pos, 10))
        alpha, alpha_mask = self.rank(part, part_mask, dim=0)
        mask = alpha_mask & self.p(self.MASK, pos)
        alpha.masked_fill_(~mask, float('nan'))
        return alpha, mask

    def best_003(self, pos):
        part = -((2 * self.bn(self.S_FWDS_ADJCLOSE, pos, 20) - self.bn(self.S_FWDS_ADJLOW, pos, 20) - self.bn(self.S_FWDS_ADJHIGH, pos, 20)) / (self.bn(self.S_FWDS_ADJHIGH, pos, 20) - self.bn(self.S_FWDS_ADJLOW, pos, 20)))
        part_mask = self.process_nan_infinite_and_mask(part, self.bn(self.MASK, pos, 20))
        alpha, alpha_mask = self.rank(part, part_mask, dim=0)
        mask = alpha_mask & self.p(self.MASK, pos)
        alpha.masked_fill_(~mask, float('nan'))
        return alpha, mask
    
    def stock018(self, pos):
        data1, data1_mask = self.rank(1 / self.p(self.S_FWDS_ADJCLOSE, pos), self.p(self.MASK, pos), dim=0)  # [stock]
        data2, data2_mask = self.mean(self.bn(self.S_DQ_VOLUME, pos, 20), self.bn(self.MASK, pos, 20), dim=0) # [stock]
        part1 = (data1 * self.p(self.S_DQ_VOLUME, pos)) / data2
        data3, data3_mask = self.rank(self.p(self.S_FWDS_ADJHIGH, pos) - self.p(self.S_FWDS_ADJCLOSE, pos), self.p(self.MASK, pos), dim=0) # [stock]
        data4, data4_mask = self.mean(self.bn(self.S_FWDS_ADJHIGH, pos, 5), self.bn(self.MASK, pos, 5), dim=0) # [stock]
        part2 = (data3 * self.p(self.S_FWDS_ADJHIGH, pos)) / data4
        part3, part3_mask = self.rank(self.bn(self.S_DQ_AVGPRICE, pos, 20) - self.bn(self.S_DQ_AVGPRICE, pos-5, 20), (self.bn(self.MASK, pos, 20) & self.bn(self.MASK, pos-5, 20)), dim=0) # [stock]
        alpha = part1 * part2 - part3
        mask = data1_mask & data2_mask & data3_mask & data4_mask & part3_mask & self.p(self.MASK, pos)
        alpha.masked_fill_(~mask, float('nan'))
        return alpha, mask

    def extra_006(self, pos):
        data1, data1_mask = self.rolling_mean(self.rg(self.S_DQ_VOLUME, pos, 20, 20), self.rg(self.MASK, pos, 20, 20), 20) # [20, stock]
        data1 = self.bn(self.S_DQ_VOLUME, pos, 20) / data1 # [20, stock]
        data1_mask &= self.bn(self.MASK, pos, 20)
        alpha, alpha_mask = self.rank(data1, data1_mask, dim=0)
        mask = alpha_mask & self.p(self.MASK, pos)
        alpha.masked_fill_(~mask, float('nan'))
        return alpha, mask

    def old_035(self, pos):
        data1, data1_mask = self.lowday(self.bn(self.S_FWDS_ADJLOW, pos, 20), self.bn(self.MASK, pos, 20))
        alpha = 1 - data1 / 20
        mask = data1_mask
        alpha.masked_fill_(~mask, float('nan'))
        return alpha, mask

    def best_006(self, pos):
        alpha = -((2 * self.en(self.S_FWDS_ADJCLOSE, pos) - self.en(self.S_FWDS_ADJLOW, pos) - self.en(self.S_FWDS_ADJHIGH, pos)) / (self.en(self.S_FWDS_ADJHIGH, pos) - self.en(self.S_FWDS_ADJLOW, pos)))  # [ewma, stock]
        alpha_mask = self.process_nan_infinite_and_mask(alpha, self.en(self.MASK, pos))
        #print('best_006, 0:', alpha.shape, alpha[:, 8], flush=True, file=sys.stderr)
        alpha = self.ewma(alpha, alpha_mask, alpha=1/10.0)[-1]
        #print('best_006, 1:', alpha.shape, alpha[8], flush=True, file=sys.stderr)
        mask = self.p(self.MASK, pos)
        alpha.masked_fill_(~mask, float('nan'))
        return alpha, mask

    def original_005(self, pos):
        filled, filled_mask = self.fill_masked_with_last(self.bn(self.S_FWDS_ADJCLOSE, pos, 65), self.bn(self.MASK, pos, 65))
        ret, ret_mask = self.pct_change(self.rg(filled, 64, 20, 6), self.rg(filled_mask, 64, 20, 6)) # [20+5, stock]
        part1, part1_mask = self.rolling_std(ret, ret_mask, 20) # [5, stock]
        part2, part2_mask = self.bn(self.S_FWDS_ADJCLOSE, pos, 5), self.bn(self.MASK, pos, 5)
        ret.masked_fill_(~ret_mask, float('nan'))
        alpha = t.where(ret[-5:] >= 0, t.zeros_like(part1), part1) + t.where(ret[-5:] < 0, t.zeros_like(part2), part2) # [5, stock]
        alpha_mask1 = t.where(ret[-5:] >= 0, t.ones_like(part1_mask).bool(), part1_mask) 
        alpha_mask2 = t.where(ret[-5:] < 0, t.ones_like(part2_mask).bool(), part2_mask) # [5, stock]
        alpha_mask = alpha_mask1 & alpha_mask2
        alpha, alpha_mask = self.max(alpha**2, alpha_mask, dim=0) # [stock]
        alpha.masked_fill_(~alpha_mask, float('nan'))
        return alpha, alpha_mask
    
    def stock022(self, pos):
        volume_avg, volume_avg_mask = self.rolling_mean(self.rg(self.S_DQ_VOLUME, pos, 20, 5), self.rg(self.MASK, pos, 20, 5), 20)  # [5, stock]
        corr, corr_mask = self.corr(volume_avg, self.bn(self.S_FWDS_ADJLOW, pos, 5), (self.bn(self.MASK, pos, 5) & volume_avg_mask), dim=0)
        alpha = corr + (self.p(self.S_FWDS_ADJHIGH, pos) + self.p(self.S_FWDS_ADJLOW, pos)) / 2 - self.p(self.S_FWDS_ADJCLOSE, pos)
        mask = corr_mask & self.p(self.MASK, pos)
        alpha.masked_fill_(~mask, float('nan'))
        return alpha, mask

    def original_006(self, pos):
        temp = (2 * self.bn(self.S_FWDS_ADJCLOSE, pos, 6) - self.bn(self.S_FWDS_ADJLOW, pos, 6) - self.bn(self.S_FWDS_ADJHIGH, pos, 6)) / (self.bn(self.S_FWDS_ADJHIGH, pos, 6) - self.bn(self.S_FWDS_ADJLOW, pos, 6))
        alpha, alpha_mask = self.sum(temp * self.bn(self.S_DQ_VOLUME, pos, 6), self.bn(self.MASK, pos, 6), dim=0)
        alpha.masked_fill_(~alpha_mask, float('nan'))
        return alpha, alpha_mask

    def change_004(self, pos):
        test0, test0_mask = self.en(self.S_FWDS_ADJCLOSE, pos), self.en(self.MASK, pos)
        #print('change_004, test0, test0_mask:', test0.shape, test0_mask.shape, test0, test0_mask, flush=True, file=sys.stderr)
        test1 = self.ewma(test0, test0_mask, alpha=1/20.0)
        #print('change_004, test0:', test0.shape, test0, flush=True, file=sys.stderr)
        part0 = self.bn(self.S_FWDS_ADJCLOSE, pos, 10) - self.ewma(self.en(self.S_FWDS_ADJCLOSE, pos), self.en(self.MASK, pos), alpha=1/20.0)[-10:]  # [10, stock]
        #print('change_004, close:', self.en(self.S_FWDS_ADJCLOSE, pos)[:, 1135], self.en(self.MASK, pos)[:, 1135], flush=True, file=sys.stderr)
        #print('change_004, part0:', part0.shape, part0[:, 1135], flush=True, file=sys.stderr)
        part1, part1_mask = self.std(self.bn(self.S_FWDS_ADJCLOSE, pos, 10), self.bn(self.MASK, pos, 10), dim=0)  # [stock]
        #print('change_004, part1:', part1.shape, part1_mask.shape, part1[1135], part1_mask[1135], flush=True, file=sys.stderr)
        alpha = part0[-1] / part1
        #print('change_004, alpha:', alpha.shape, part1_mask.shape, alpha[1135], part1_mask[1135], flush=True, file=sys.stderr)
        alpha.masked_fill_(~part1_mask, float('nan'))
        return alpha, part1_mask

    def change_003(self, pos):
        part0 = self.bn(self.S_FWDS_ADJCLOSE, pos, 10) - self.ewma(self.en(self.S_FWDS_ADJCLOSE, pos), self.en(self.MASK, pos), alpha=1/10.0)[-10:]  # [10, stock]
        part1, part1_mask = self.std(self.bn(self.S_FWDS_ADJCLOSE, pos, 10), self.bn(self.MASK, pos, 10), dim=0)  # [stock]
        alpha = part0[-1] / part1
        alpha.masked_fill_(~part1_mask, float('nan'))
        return alpha, part1_mask

    def change_002(self, pos):
        part0 = self.bn(self.S_FWDS_ADJCLOSE, pos, 10) - self.ewma(self.en(self.S_FWDS_ADJCLOSE, pos), self.en(self.MASK, pos), alpha=1/5.0)[-10:]  # [10, stock]
        part1, part1_mask = self.std(self.bn(self.S_FWDS_ADJCLOSE, pos, 10), self.bn(self.MASK, pos, 10), dim=0)  # [stock]
        alpha = part0[-1] / part1
        alpha.masked_fill_(~part1_mask, float('nan'))
        return alpha, part1_mask

    def original_001(self, pos):
        delay1, delay1_mask = self.bn(self.S_FWDS_ADJCLOSE, pos-1, 6), self.bn(self.MASK, pos-1, 6)  # [6, stock]
        nodelay, nodelay_mask = self.bn(self.S_FWDS_ADJCLOSE, pos, 6), self.bn(self.MASK, pos, 6)  # [6, stock]
        condition1 = (nodelay <= delay1)  # [6, stock]
        condition2 = (nodelay >= delay1)  # [6, stock]
        part1 = (nodelay - t.minimum(self.bn(self.S_FWDS_ADJLOW, pos, 6), delay1))  # [6, stock]
        part1[condition1] = 0  # [6, stock]
        part2 = (nodelay - t.maximum(self.bn(self.S_FWDS_ADJHIGH, pos, 6), delay1))  # [6, stock]
        part2[condition2] = 0  # [6, stock]
        alpha, alpha_mask = self.sum(part1+part2, (delay1_mask & nodelay_mask), dim=0, copy=False)  # [stock]
        alpha.masked_fill_(~alpha_mask, float('nan'))
        return alpha, alpha_mask

    def target_01(self, pos):
        alpha = (self.p(self.S_FWDS_ADJCLOSE, pos+1) - self.p(self.S_FWDS_ADJCLOSE, pos)) / self.p(self.S_FWDS_ADJCLOSE, pos)
        mask = self.p(self.MASK, pos+1) & self.p(self.MASK, pos)
        alpha.masked_fill_(~mask, float('nan'))
        return alpha, mask

    def change_005(self, pos):
        part0 = self.ewma(self.en(self.S_FWDS_ADJCLOSE, pos), self.en(self.MASK, pos), alpha=1/5.0)[-1]  # [stock]
        #print('change_005, part0:', part0.shape, part0[3271], flush=True, file=sys.stderr)
        part1 = self.ewma(self.en(self.S_FWDS_ADJCLOSE, pos), self.en(self.MASK, pos), alpha=1/10.0)[-1]  # [stock]
        #print('change_005, part1:', part1.shape, part1[3271], flush=True, file=sys.stderr)
        part2 = self.ewma(self.en(self.S_FWDS_ADJCLOSE, pos), self.en(self.MASK, pos), alpha=1/20.0)[-1]  # [stock]
        #print('change_005, part2:', part2.shape, part2[3271], flush=True, file=sys.stderr)
        alpha = t.zeros_like(self.p(self.S_FWDS_ADJCLOSE, pos))  # [stock]
        cond1 = (part0 > part1) & (part1 > part2)
        cond2 = (part0 < part1) & (part1 < part2)
        alpha[cond1] = 1
        alpha[cond2] = -1
        #print('change_005, alpha:', alpha.shape, alpha[3271], flush=True, file=sys.stderr)
        mask = self.p(self.MASK, pos)
        alpha.masked_fill_(~mask, float('nan'))
        return alpha, mask

    def original_004(self, pos):
        temp = ((self.en(self.S_FWDS_ADJHIGH, pos) + self.en(self.S_FWDS_ADJLOW, pos)) * 0.5 - (self.en(self.S_FWDS_ADJHIGH, pos-1) + self.en(self.S_FWDS_ADJLOW, pos-1)) * 0.5) * (self.en(self.S_FWDS_ADJHIGH, pos) - self.en(self.S_FWDS_ADJLOW, pos)) / self.en(self.S_DQ_VOLUME, pos)  # [ewma, stock]
        temp_mask = self.en(self.MASK, pos) & self.en(self.MASK, pos-1)
        alpha = self.ewma(temp, temp_mask, alpha=2/7.0)[-1]
        mask = self.p(self.MASK, pos)
        alpha.masked_fill_(~mask, float('nan'))
        return alpha, mask

    def change_001(self, pos):
        result = (2 * self.bn(self.S_FWDS_ADJCLOSE, pos, 5) - self.bn(self.S_FWDS_ADJLOW, pos, 5) - self.bn(self.S_FWDS_ADJHIGH, pos, 5)) / (self.bn(self.S_FWDS_ADJHIGH, pos, 5) - self.bn(self.S_FWDS_ADJLOW, pos, 5)) # [5, stock]
        result_mask = self.bn(self.MASK, pos, 5)
        alpha, alpha_mask = self.mean(result, result_mask, dim=0)
        alpha.masked_fill_(~alpha_mask, float('nan'))
        return alpha, alpha_mask

    def add_002(self, pos):
        data1, data1_mask = t.log(self.bn(self.S_DQ_VOLUME, pos, 21) + 1), self.bn(self.MASK, pos, 21)  # [21, stock]
        data1 = data1[1:] - data1[:-1]  # [20, stock]
        data1_mask = data1_mask[1:] & data1_mask[:-1]  # [20, stock]
        data2 = ((self.bn(self.S_FWDS_ADJCLOSE, pos, 20) - self.bn(self.S_FWDS_ADJOPEN, pos, 20)) / self.bn(self.S_FWDS_ADJOPEN, pos, 20))
        alpha, alpha_mask = self.corr(data1, data2, data1_mask, dim=0)
        alpha.masked_fill_(~alpha_mask, float('nan'))
        return alpha, alpha_mask

    def original_002(self, pos):
        condition = (self.p(self.S_FWDS_ADJOPEN, pos) * 0.85 + self.p(self.S_FWDS_ADJHIGH, pos) * 0.15) - (self.p(self.S_FWDS_ADJOPEN, pos-4) * 0.85 + self.p(self.S_FWDS_ADJHIGH, pos-4) * 0.15)
        condition_mask = self.p(self.MASK, pos) & self.p(self.MASK, pos-4)
        #print('original_002, condition, condition_mask:', condition.shape, condition_mask.shape, condition[1493], condition_mask[1493], flush=True, file=sys.stderr)
        condition.masked_fill_(~condition_mask, float('nan'))
        #print('original_002, condition:', condition.shape, condition[1493], flush=True, file=sys.stderr)
        condition1 = (condition <= 0)
        condition2 = (condition >= 0)
        indicator1 = t.ones_like(condition)
        indicator2 = -t.ones_like(condition)
        indicator1[condition1] = 0
        indicator2[condition2] = 0
        #print('original_002, indicator:', indicator1.shape, indicator2.shape, indicator1[1493], indicator2[1493], flush=True, file=sys.stderr)
        alpha = indicator1 + indicator2
        mask = self.p(self.MASK, pos)
        alpha.masked_fill_(~mask, float('nan'))
        return alpha, mask

    def original_003(self, pos):
        temp = (self.p(self.S_FWDS_ADJHIGH, pos) + self.p(self.S_FWDS_ADJLOW, pos)) * 0.2 / 2 + self.p(self.S_DQ_AVGPRICE, pos) * 0.8
        temp2 = (self.p(self.S_FWDS_ADJHIGH, pos-4) + self.p(self.S_FWDS_ADJLOW, pos-4)) * 0.2 / 2 + self.p(self.S_DQ_AVGPRICE, pos-4) * 0.8
        alpha = temp2 - temp
        mask = self.p(self.MASK, pos) & self.p(self.MASK, pos-4)
        alpha.masked_fill_(~mask, float('nan'))
        return alpha, mask

    def add_021(self, pos):
        data1, data1_mask = self.rolling_std(self.eg(self.S_FWDS_ADJCLOSE, pos, 10), self.eg(self.MASK, pos, 10), 10)  # [ewma, stock]
        cond1, cond1_mask = self.en(self.S_FWDS_ADJCLOSE, pos), self.en(self.MASK, pos)
        cond2, cond2_mask = self.en(self.S_FWDS_ADJCLOSE, pos-1), self.en(self.MASK, pos-1)
        cond1.masked_fill_(~cond1_mask, float('nan'))
        cond2.masked_fill_(~cond2_mask, float('nan'))
        cond = cond1 <= cond2
        data1[~cond] = 0
        data1_mask[~cond] = True
        alpha = self.ewma(data1, data1_mask, alpha=1/5)[-1] - self.ewma(data1, data1_mask, alpha=1/20)[-1]
        mask = self.p(self.MASK, pos)
        alpha.masked_fill_(~mask, float('nan'))
        return alpha, mask

    def add_020(self, pos):
        data1 = self.bn(self.S_FWDS_ADJHIGH, pos, 6) - self.bn(self.S_FWDS_ADJCLOSE, pos-1, 6)
        data1 = t.maximum(data1, t.tensor(0, dtype=data1.dtype, device=data1.device)) # [6, stock]
        data2 = self.bn(self.S_FWDS_ADJCLOSE, pos-1, 20) - self.bn(self.S_FWDS_ADJLOW, pos, 20)
        data2 = t.maximum(data2, t.tensor(0, dtype=data2.dtype, device=data2.device)) # [20, stock]
        data1_mask = self.bn(self.MASK, pos, 6) & self.bn(self.MASK, pos-1, 6) # [6, stock]
        data2_mask = self.bn(self.MASK, pos, 20) & self.bn(self.MASK, pos-1, 20) # [20, stock]
        sum1, sum1_mask = self.sum(data1, data1_mask, dim=0) # [stock]
        sum2, sum2_mask = self.sum(data2, data2_mask, dim=0) # [stock]
        alpha = sum1 / sum2 * 100
        mask = sum1_mask & sum2_mask
        alpha.masked_fill_(~mask, float('nan'))
        return alpha, mask

    def add_030(self, pos):
        alpha = self.p(self.S_FWDS_ADJCLOSE, pos) / self.ewma(self.en(self.S_FWDS_ADJCLOSE, pos), self.en(self.MASK, pos), alpha=1/20.0)[-1]
        mask = self.p(self.MASK, pos)
        alpha.masked_fill_(~mask, float('nan'))
        return alpha, mask

    def add_029(self, pos):
        alpha = self.p(self.S_FWDS_ADJCLOSE, pos) / self.ewma(self.en(self.S_FWDS_ADJCLOSE, pos), self.en(self.MASK, pos), alpha=1/10.0)[-1]
        mask = self.p(self.MASK, pos)
        alpha.masked_fill_(~mask, float('nan'))
        return alpha, mask

    def add_028(self, pos):
        alpha = self.p(self.S_FWDS_ADJCLOSE, pos) / self.ewma(self.en(self.S_FWDS_ADJCLOSE, pos), self.en(self.MASK, pos), alpha=1/5.0)[-1]
        mask = self.p(self.MASK, pos)
        alpha.masked_fill_(~mask, float('nan'))
        return alpha, mask

    def stock016(self, pos):
        data1 = t.minimum(self.bn(self.S_FWDS_ADJLOW, pos, 24), self.bn(self.S_FWDS_ADJCLOSE, pos-1, 24))  # [24, stock]
        data2 = t.maximum(self.bn(self.S_FWDS_ADJHIGH, pos, 24), self.bn(self.S_FWDS_ADJCLOSE, pos-1, 24))  # [24, stock]
        data_mask = self.bn(self.MASK, pos, 24) & self.bn(self.MASK, pos-1, 24)  # [24, stock]
        part1_1, part1_1_mask = self.sum(data1[ -6:], data_mask[ -6:], dim=0)  # [stock]
        part2_1, part2_1_mask = self.sum(data1[-12:], data_mask[-12:], dim=0)  # [stock]
        part3_1, part3_1_mask = self.sum(data1[-24:], data_mask[-24:], dim=0)  # [stock]
        part1_2, part1_2_mask = self.sum(data2[ -6:] - data1[ -6:], data_mask[ -6:], dim=0)  # [stock]
        part2_2, part2_2_mask = self.sum(data2[-12:] - data1[-12:], data_mask[-12:], dim=0)  # [stock]
        part3_2, part3_2_mask = self.sum(data2[-24:] - data1[-24:], data_mask[-24:], dim=0)  # [stock]
        part1 = (self.p(self.S_FWDS_ADJCLOSE, pos) - part1_1) / part1_2 * 12 * 24  # [stock]
        part2 = (self.p(self.S_FWDS_ADJCLOSE, pos) - part2_1) / part2_2 * 6 * 24  # [stock]
        part3 = (self.p(self.S_FWDS_ADJCLOSE, pos) - part3_1) / part3_2 * 6 * 24  # [stock]
        mask = self.p(self.MASK, pos) & part1_1_mask & part2_1_mask & part3_1_mask
        alpha = (part1 + part2 + part3) * 100 / (6 * 12 + 6 * 24 + 12 * 24)
        alpha.masked_fill_(~mask, float('nan'))
        return alpha, mask

    def add_011(self, pos):
        data1 = (1 / self.p(self.S_FWDS_ADJCLOSE, pos))  # [stock]
        data2, data2_mask = self.mean(self.bn(self.S_DQ_VOLUME, pos, 20), self.bn(self.MASK, pos, 20), dim=0)  # [stock]
        part1 = (data1 * self.p(self.S_DQ_VOLUME, pos)) / data2  # [stock]
        part1_mask = self.p(self.MASK, pos) & data2_mask # [stock]
        data3 = (self.p(self.S_FWDS_ADJHIGH, pos) - self.p(self.S_FWDS_ADJCLOSE, pos)) # [stock]
        data4, data4_mask = self.mean(self.bn(self.S_FWDS_ADJHIGH, pos, 5), self.bn(self.MASK, pos, 5), dim=0)  # [stock]
        part2 = (data3 * self.p(self.S_FWDS_ADJHIGH, pos)) / data4  # [stock]
        part2_mask = self.p(self.MASK, pos) & data4_mask # [stock]
        part3 = (self.p(self.S_DQ_AVGPRICE, pos) - self.p(self.S_DQ_AVGPRICE, pos-5))  # [stock]
        part3_mask = self.p(self.MASK, pos) & self.p(self.MASK, pos-5)
        alpha = part1 * part2 - part3
        mask = part1_mask & part2_mask & part3_mask
        alpha.masked_fill_(~mask, float('nan'))
        return alpha, mask

    def add_019(self, pos):
        data, data_mask = self.mean(self.bn(self.S_DQ_VOLUME, pos, 5), self.bn(self.MASK, pos, 5), dim=0)
        alpha = self.p(self.S_DQ_VOLUME, pos) / data
        mask = self.p(self.MASK, pos) & data_mask
        alpha.masked_fill_(~mask, float('nan'))
        return alpha, mask

    def add_018(self, pos):
        data, data_mask = self.mean(self.bn(self.S_DQ_VOLUME, pos, 10), self.bn(self.MASK, pos, 10), dim=0)
        alpha = self.p(self.S_DQ_VOLUME, pos) / data
        mask = self.p(self.MASK, pos) & data_mask
        alpha.masked_fill_(~mask, float('nan'))
        return alpha, mask

    def add_013(self, pos):
        part1, part1_mask = self.max(self.bn(self.S_DQ_AVGPRICE, pos, 3) - self.bn(self.S_FWDS_ADJCLOSE, pos, 3), self.bn(self.MASK, pos, 3), dim=0)  # [stock]
        part2, part2_mask = self.min(self.bn(self.S_DQ_AVGPRICE, pos, 3) - self.bn(self.S_FWDS_ADJCLOSE, pos, 3), self.bn(self.MASK, pos, 3), dim=0)  # [stock]
        part3 = self.p(self.S_DQ_VOLUME, pos) - self.p(self.S_DQ_VOLUME, pos-3)
        part3_mask = self.p(self.MASK, pos) & self.p(self.MASK, pos-3) 
        alpha = part1 + part2 * part3
        mask = part1_mask & part2_mask & part3_mask
        alpha.masked_fill_(~mask, float('nan'))
        return alpha, mask

    def add_010(self, pos):
        df1, df1_mask = self.bn(self.S_FWDS_ADJCLOSE, pos, 11), self.bn(self.MASK, pos, 11) # [11, stock]
        df1.masked_fill_(~df1_mask, float('nan'))
        df1 = df1[1:] < df1[:-1]
        df1_mask = t.ones_like(df1).bool()  # [10, stock]
        sumif = t.abs(self.bn(self.S_FWDS_ADJCLOSE, pos, 10) / self.bn(self.S_FWDS_ADJCLOSE, pos-1, 10) - 1) / t.log(self.bn(self.S_DQ_AMOUNT, pos, 10) + 1)  # [10, stock]
        sumif_mask = self.bn(self.MASK, pos, 10) & self.bn(self.MASK, pos-1, 10)  # [10, stock]
        sumif.masked_fill_(~df1, 0)  # [10, stock]
        sumif_mask = self.process_nan_infinite_and_mask(sumif, t.ones_like(sumif).bool())
        sumif, sumif_mask = self.sum(sumif, sumif_mask, dim=0)  # [stock]
        count, count_mask = self.sum(df1, df1_mask, dim=0) # [stock]
        alpha = (sumif / count) # [stock]
        mask = sumif_mask & count_mask & self.p(self.MASK, pos) # [stock]
        alpha.masked_fill_(~mask, float('nan'))
        return alpha, mask

    def add_017(self, pos):
        data1, data1_mask = self.rolling_mean(self.rg(self.S_FWDS_ADJCLOSE, pos, 10, 4), self.rg(self.MASK, pos, 10, 4), 10)  # [3, stock]
        data1 = (self.bn(self.S_FWDS_ADJHIGH, pos, 4) - self.bn(self.S_FWDS_ADJLOW, pos, 4)) / data1 # [3, stock]
        data1_mask = self.bn(self.MASK, pos, 4) & data1_mask
        rank1, rank1_mask = data1[0], data1_mask[0]
        rank2, rank2_mask = self.p(self.S_DQ_VOLUME, pos), self.p(self.MASK, pos)
        data2 = (data1[-1] / (self.p(self.S_DQ_AVGPRICE, pos) - self.p(self.S_FWDS_ADJCLOSE, pos)))
        data2_mask = data1_mask[-1] & self.p(self.MASK, pos)
        alpha = (rank1 * rank2) / data2
        mask = rank1_mask & rank2_mask & data2_mask
        alpha.masked_fill_(~mask, float('nan'))
        return alpha, mask

    def add_024(self, pos):
        delay5 = self.p(self.S_FWDS_ADJCLOSE, pos-20)
        #print('add_024, delay5:', delay5.shape, delay5[3527], flush=True, file=sys.stderr)
        alpha = self.p(self.S_FWDS_ADJCLOSE, pos) / delay5
        #print('add_024, alpha:', alpha.shape, alpha[3527], flush=True, file=sys.stderr)
        mask = self.p(self.MASK, pos) & self.p(self.MASK, pos-20)
        alpha.masked_fill_(~mask, float('nan'))
        return alpha, mask

    def add_016(self, pos):
        data1, data1_mask = self.rolling_mean(self.rg(self.S_FWDS_ADJCLOSE, pos, 20, 3), self.rg(self.MASK, pos, 20, 3), 20)  # [3, stock]
        data1 = (self.bn(self.S_FWDS_ADJHIGH, pos, 3) - self.bn(self.S_FWDS_ADJLOW, pos, 3)) / data1  # [3, stock]
        data1_mask = self.bn(self.MASK, pos, 3) & data1_mask # [3, stock]
        rank1, rank1_mask = data1[0], data1_mask[0]  #[stock]
        rank2 = self.p(self.S_DQ_VOLUME, pos) # [stock]
        data2 = (data1[2] / (self.p(self.S_DQ_AVGPRICE, pos) - self.p(self.S_FWDS_ADJCLOSE, pos)))  # [stock]
        data2_mask = data1_mask[2] & self.p(self.MASK, pos)
        alpha = (rank1 * rank2) / data2
        mask = rank1_mask & data2_mask
        alpha.masked_fill_(~mask, float('nan'))
        return alpha, mask

    def add_023(self, pos):
        delay5 = self.p(self.S_FWDS_ADJCLOSE, pos-10)
        alpha = self.p(self.S_FWDS_ADJCLOSE, pos) / delay5
        mask = self.p(self.MASK, pos) & self.p(self.MASK, pos-10)
        alpha.masked_fill_(~mask, float('nan'))
        return alpha, mask

    def add_006(self, pos):
        filled, filled_mask = self.fill_masked_with_last(self.bn(self.S_FWDS_ADJCLOSE, pos, 65), self.bn(self.MASK, pos, 65))
        ret, ret_mask = self.pct_change(self.rg(filled, 64, 5, 12), self.rg(filled_mask, 64, 5, 12)) # [5+10, stock]
        temp1, temp1_mask = self.rolling_sum(self.rg(self.S_FWDS_ADJOPEN, pos, 5, 11), self.rg(self.MASK, pos, 5, 11), 5) # [11, stock]
        temp2, temp2_mask = self.rolling_sum(ret, ret_mask, 5) # [11, stock]
        temp = temp1 * temp2
        temp_mask = temp1_mask & temp2_mask
        alpha = temp[10] - temp[0]
        mask = temp_mask[10] & temp_mask[0]
        alpha.masked_fill_(~mask, float('nan'))
        return alpha, mask

    def stock002(self, pos):
        data1, data1_mask = t.log(self.bn(self.S_DQ_VOLUME, pos, 7) + 1), self.bn(self.MASK, pos, 7)  # [7, stock]
        data1, data1_mask = data1[1:] - data1[:-1], data1_mask[1:] & data1_mask[:-1]  # [6, stock]
        data2 = ((self.bn(self.S_FWDS_ADJCLOSE, pos, 6) - self.bn(self.S_FWDS_ADJOPEN, pos, 6)) / self.bn(self.S_FWDS_ADJOPEN, pos, 6))  # [6, stock]
        data2_mask = self.bn(self.MASK, pos, 6)  # [6, stock]
        alpha, alpha_mask = self.corr(data1, data2, (data1_mask & data2_mask), dim=0)  # [stock]
        alpha.masked_fill_(~alpha_mask, float('nan'))
        return alpha, alpha_mask

    def stock001(self, pos):
        data1, data1_mask = t.log(self.bn(self.S_DQ_VOLUME, pos, 5) + 1), self.bn(self.MASK, pos, 5)  # [5, stock]
        data1, data1_mask = data1[1:] - data1[:-1], data1_mask[1:] & data1_mask[:-1]  # [4, stock]
        data1, data1_mask = self.rank(data1, data1_mask, dim=1)  # [4, stock]
        data2 = ((self.bn(self.S_FWDS_ADJCLOSE, pos, 4) - self.bn(self.S_FWDS_ADJOPEN, pos, 4)) / self.bn(self.S_FWDS_ADJOPEN, pos, 4))  # [4, stock]
        data2_mask = self.bn(self.MASK, pos, 4)  # [4, stock]
        data2, data2_mask = self.rank(data2, data2_mask, dim=1)  # [4, stock]
        alpha, alpha_mask = self.corr(data1, data2, (data1_mask & data2_mask), dim=0)  # [stock]
        alpha.masked_fill_(~alpha_mask, float('nan'))
        return alpha, alpha_mask

    def add_008(self, pos):
        alpha, alpha_mask = self.std(t.log(self.bn(self.S_DQ_AMOUNT, pos, 20) + 1), self.bn(self.MASK, pos, 20), dim=0) 
        alpha.masked_fill_(~alpha_mask, float('nan'))
        return alpha, alpha_mask

    def add_015(self, pos):
        data1 = self.en(self.S_DQ_VOLUME, pos) * (2 * self.en(self.S_FWDS_ADJCLOSE, pos) - self.en(self.S_FWDS_ADJLOW, pos) - self.en(self.S_FWDS_ADJHIGH, pos)) / (self.en(self.S_FWDS_ADJHIGH, pos) - self.en(self.S_FWDS_ADJLOW, pos))
        data1_mask = self.process_nan_infinite_and_mask(data1, self.en(self.MASK, pos))
        #print('add_015, data1:', data1.shape, data1_mask.shape, data1[:, 8],  data1_mask[:, 8], flush=True, file=sys.stderr)
        x = self.ewma(data1, data1_mask, alpha=1.0/20)[-1]
        y = self.ewma(data1, data1_mask, alpha=1.0/10)[-1]
        #print('add_015, x,y:', x.shape, y.shape, x[8],  y[8], flush=True, file=sys.stderr)
        alpha = (x - y)
        mask = self.p(self.MASK, pos)
        #print('add_015, alpha:', alpha.shape, mask.shape, alpha[8],  mask[8], flush=True, file=sys.stderr)
        alpha.masked_fill_(~mask, float('nan'))
        return alpha, mask

    def add_014(self, pos):
        data1 = self.en(self.S_DQ_VOLUME, pos) * (2 * self.en(self.S_FWDS_ADJCLOSE, pos) - self.en(self.S_FWDS_ADJLOW, pos) - self.en(self.S_FWDS_ADJHIGH, pos)) / (self.en(self.S_FWDS_ADJHIGH, pos) - self.en(self.S_FWDS_ADJLOW, pos))
        data1_mask = self.process_nan_infinite_and_mask(data1, self.en(self.MASK, pos))
        x = self.ewma(data1, data1_mask, alpha=1.0/10)[-1]
        y = self.ewma(data1, data1_mask, alpha=1.0/5)[-1]
        alpha = (x - y)
        mask = self.p(self.MASK, pos)
        alpha.masked_fill_(~mask, float('nan'))
        return alpha, mask

    def add_007(self, pos):
        alpha, alpha_mask = self.std(t.log(self.bn(self.S_DQ_AMOUNT, pos, 10) + 1), self.bn(self.MASK, pos, 10), dim=0) 
        alpha.masked_fill_(~alpha_mask, float('nan'))
        return alpha, alpha_mask

    def add_012(self, pos):
        sma = self.ewma(self.en(self.S_FWDS_ADJHIGH, pos) - self.en(self.S_FWDS_ADJLOW, pos), self.en(self.MASK, pos), alpha=1/11)[-1]
        alpha = ((self.p(self.S_FWDS_ADJHIGH, pos) - self.p(self.S_FWDS_ADJLOW, pos) - sma) / sma * 100)
        mask = self.p(self.MASK, pos)
        alpha.masked_fill_(~mask, float('nan'))
        return alpha, mask

    def add_003(self, pos):
        part0, part0_mask = self.rolling_mean(self.eg(self.S_FWDS_ADJCLOSE, pos, 10), self.eg(self.MASK, pos, 10), 10)  # [ewma, stock]
        part1 = (self.en(self.S_FWDS_ADJCLOSE, pos) - part0) / part0  # [ewma, stock]
        part1_mask = self.en(self.MASK, pos) & part0_mask  # [ewma, stock]
        alpha = part1[5:] - part1[:-5] # [ewma-6, stock]
        alpha_mask = part1_mask[5:] & part1_mask[:-5] # [ewma-6, stock]
        alpha = self.ewma(alpha, alpha_mask, alpha=1.0/20)[-1] # [stock]
        mask = self.p(self.MASK, pos)
        alpha.masked_fill_(~mask, float('nan'))
        return alpha, mask

    def add_009(self, pos):
        temp1 = self.en(self.S_DQ_VOLUME, pos) - self.en(self.S_DQ_VOLUME, pos-1)  # [ewma, stock]
        temp1_mask = self.en(self.MASK, pos) & self.en(self.MASK, pos-1)  # [ewma, stock]
        temp1.masked_fill_(~temp1_mask, float('nan'))
        part1 = t.maximum(temp1, t.zeros_like(temp1))
        part1 = self.ewma(part1, temp1_mask, alpha=1.0/12)[-1]
        temp2 = temp1.abs()
        part2 = self.ewma(temp2, temp1_mask, alpha=1.0/12)[-1]
        alpha = part1 * 100 / part2
        mask = self.p(self.MASK, pos)
        alpha.masked_fill_(~mask, float('nan'))
        return alpha, mask

    def extra_008(self, pos):
        data1 = self.bn(self.S_FWDS_ADJHIGH, pos, 5) - self.bn(self.S_FWDS_ADJCLOSE, pos-1, 5)
        data1 = t.maximum(data1, t.tensor(0, dtype=data1.dtype, device=data1.device)) # [5, stock]
        data2 = self.bn(self.S_FWDS_ADJCLOSE, pos-1, 10) - self.bn(self.S_FWDS_ADJLOW, pos, 10)
        data2 = t.maximum(data2, t.tensor(0, dtype=data2.dtype, device=data2.device)) # [10, stock]
        data1_mask = self.bn(self.MASK, pos, 5) & self.bn(self.MASK, pos-1, 5) # [5, stock]
        data2_mask = self.bn(self.MASK, pos, 10) & self.bn(self.MASK, pos-1, 10) # [10, stock]
        sum1, sum1_mask = self.sum(data1, data1_mask, dim=0) # [stock]
        sum2, sum2_mask = self.sum(data2, data2_mask, dim=0) # [stock]
        alpha = sum1 / sum2 * 100
        mask = sum1_mask & sum2_mask
        alpha.masked_fill_(~mask, float('nan'))
        return alpha, mask


    def extra_009(self, pos):
        data1, data1_mask = self.rolling_std(self.eg(self.S_FWDS_ADJCLOSE, pos, 20), self.eg(self.MASK, pos, 20), 20)  # [ewma, stock]
        cond1, cond1_mask = self.en(self.S_FWDS_ADJCLOSE, pos).clone(), self.en(self.MASK, pos)
        cond2, cond2_mask = self.en(self.S_FWDS_ADJCLOSE, pos-1).clone(), self.en(self.MASK, pos-1)
        cond1.masked_fill_(~cond1_mask, float('nan'))
        cond2.masked_fill_(~cond2_mask, float('nan'))
        cond = cond1 <= cond2
        data1[~cond] = 0
        data1_mask[~cond] = True
        alpha = self.ewma(data1, data1_mask, alpha=1/5)[-1] - self.ewma(data1, data1_mask, alpha=1/20)[-1]
        mask = self.p(self.MASK, pos)
        alpha.masked_fill_(~mask, float('nan'))
        return alpha, mask

    def add_005(self, pos):
        delay6, delay6_mask = self.p(self.S_FWDS_ADJCLOSE, pos-20), self.p(self.MASK, pos-20)
        alpha = (self.p(self.S_FWDS_ADJCLOSE, pos) - delay6) * (self.p(self.S_DQ_VOLUME, pos) + 1) / delay6
        mask = self.p(self.MASK, pos) & delay6_mask
        alpha.masked_fill_(~mask, float('nan'))
        return alpha, mask

    def add_004(self, pos):
        delay6, delay6_mask = self.p(self.S_FWDS_ADJCLOSE, pos-10), self.p(self.MASK, pos-10)
        alpha = (self.p(self.S_FWDS_ADJCLOSE, pos) - delay6) * (self.p(self.S_DQ_VOLUME, pos) + 1) / delay6
        mask = self.p(self.MASK, pos) & delay6_mask
        alpha.masked_fill_(~mask, float('nan'))
        return alpha, mask

    def stock009(self, pos):
        part1, part1_mask = self.mean(self.bn(self.S_FWDS_ADJCLOSE, pos, 3), self.bn(self.MASK, pos, 3), dim=0) # [stock]
        part2, part2_mask = self.mean(self.bn(self.S_FWDS_ADJCLOSE, pos, 6), self.bn(self.MASK, pos, 6), dim=0) # [stock]
        part3, part3_mask = self.mean(self.bn(self.S_FWDS_ADJCLOSE, pos, 12), self.bn(self.MASK, pos, 12), dim=0) # [stock]
        part4, part4_mask = self.mean(self.bn(self.S_FWDS_ADJCLOSE, pos, 24), self.bn(self.MASK, pos, 24), dim=0) # [stock]
        alpha = (part1 + part2 + part3 + part4) * 0.25 / self.p(self.S_FWDS_ADJCLOSE, pos) # [stock]
        mask = part1_mask & part2_mask & part3_mask & part4_mask & self.p(self.MASK, pos)
        alpha.masked_fill_(~mask, float('nan'))
        return alpha, mask

    def extra_002(self, pos):
        part1, part1_mask = self.max(self.bn(self.S_DQ_AVGPRICE, pos, 3) - self.bn(self.S_FWDS_ADJCLOSE, pos, 3), self.bn(self.MASK, pos, 3), dim=0)  # [stock]
        part1, part1_mask = self.rank(part1, part1_mask, dim=0) # [stock]
        part2, part2_mask = self.min(self.bn(self.S_DQ_AVGPRICE, pos, 3) - self.bn(self.S_FWDS_ADJCLOSE, pos, 3), self.bn(self.MASK, pos, 3), dim=0)  # [stock]
        part2, part2_mask = self.rank(part2, part2_mask, dim=0) # [stock]
        part3, part3_mask = self.p(self.S_DQ_VOLUME, pos) - self.p(self.S_DQ_VOLUME, pos-3), self.p(self.MASK, pos) & self.p(self.MASK, pos-3)  # [stock]
        part3, part3_mask = self.rank(part3, part3_mask, dim=0) # [stock]
        alpha = part1 + part2 * part3
        mask = part1_mask & part2_mask & part3_mask
        alpha.masked_fill_(~mask, float('nan'))
        return alpha, mask

    def extra_014(self, pos):
        alpha = (self.p(self.S_FWDS_ADJHIGH, pos) - self.p(self.S_FWDS_ADJLOW, pos)) / self.p(self.S_FWDS_ADJCLOSE, pos)
        mask = self.p(self.MASK, pos)
        alpha.masked_fill_(~mask, float('nan'))
        return alpha, mask

    def extra_013(self, pos):
        alpha = self.p(self.S_DQ_AVGPRICE, pos) / self.p(self.S_FWDS_ADJCLOSE, pos)
        mask = self.p(self.MASK, pos)
        alpha.masked_fill_(~mask, float('nan'))
        return alpha, mask

    def old_036(self, pos):
        corr1, corr1_mask = self.corr(self.bn(self.S_FWDS_ADJHIGH, pos, 5), self.bn(self.S_DQ_VOLUME, pos, 5), self.bn(self.MASK, pos, 5), dim=0) # [stock]
        corr2, corr2_mask = self.corr(self.bn(self.S_FWDS_ADJHIGH, pos-5, 5), self.bn(self.S_DQ_VOLUME, pos-5, 5), self.bn(self.MASK, pos-5, 5), dim=0) # [stock]
        corr3, corr3_mask = self.std(self.bn(self.S_FWDS_ADJCLOSE, pos, 20), self.bn(self.MASK, pos, 20), dim=0) # [stock]
        alpha = (-(corr1 - corr2) * corr3) 
        mask = corr1_mask & corr2_mask & corr3_mask
        alpha.masked_fill_(~mask, float('nan'))
        return alpha, mask

    def extra_012(self, pos):
        alpha = self.p(self.S_FWDS_ADJHIGH, pos) / self.p(self.S_FWDS_ADJOPEN, pos)
        mask = self.p(self.MASK, pos)
        alpha.masked_fill_(~mask, float('nan'))
        return alpha, mask

    def stock011(self, pos):
        data1, data1_mask = self.mean((self.bn(self.S_FWDS_ADJHIGH, pos, 12) + self.bn(self.S_FWDS_ADJLOW, pos, 12) + self.bn(self.S_FWDS_ADJCLOSE, pos, 12)) / 3, self.bn(self.MASK, pos, 12), dim=0) # [stock]
        data1 = (self.p(self.S_FWDS_ADJHIGH, pos) + self.p(self.S_FWDS_ADJLOW, pos) + self.p(self.S_FWDS_ADJCLOSE, pos)) / 3 - data1
        data1_mask = data1_mask & self.p(self.MASK, pos)
        data2, data2_mask = self.rolling_mean((self.rg(self.S_FWDS_ADJHIGH, pos, 12, 12) + self.rg(self.S_FWDS_ADJLOW, pos, 12, 12) + self.rg(self.S_FWDS_ADJCLOSE, pos, 12, 12)) / 3, self.rg(self.MASK, pos, 12, 12), 12) # [12, stock]
        data2 = t.abs(self.bn(self.S_FWDS_ADJCLOSE, pos, 12) - data2) # [12, stock]
        data2_mask = data2_mask & self.bn(self.MASK, pos, 12) # [12, stock]
        data3, data3_mask = self.mean(data2, data2_mask, dim=0)
        data3 = data3 * 0.015  # [stock]
        alpha = (data1 / data3)  # [stock]
        mask = data1_mask & data3_mask
        alpha.masked_fill_(~mask, float('nan'))
        return alpha, mask

    def extra_007(self, pos):
        data = self.en(self.S_FWDS_ADJHIGH, pos) - self.en(self.S_FWDS_ADJLOW, pos) # [ewma, stock]
        data_mask = self.en(self.MASK, pos) # [ewma, stock]
        sma1 = self.ewma(data, data_mask, alpha=2.0/5) # [ewma, stock]
        sma1_mask = self.process_nan_infinite_and_mask(sma1, t.ones_like(sma1).bool()) # [ewma, stock]
        sma2 = self.ewma(sma1, sma1_mask, alpha=2.0/20) # [ewma, stock]
        alpha = (sma1[-1] / sma2[-1])
        mask = self.p(self.MASK, pos)
        alpha.masked_fill_(~mask, float('nan'))
        return alpha, mask

    def extra_011(self, pos):
        alpha = self.p(self.S_DQ_AMOUNT, pos).clone()
        mask = self.p(self.MASK, pos)
        alpha.masked_fill_(~mask, float('nan'))
        return alpha, mask

    def extra_010(self, pos):
        delay5 = self.p(self.S_FWDS_ADJCLOSE, pos-5)
        alpha = self.p(self.S_FWDS_ADJCLOSE, pos) / delay5
        mask = self.p(self.MASK, pos-5) & self.p(self.MASK, pos)
        alpha.masked_fill_(~mask, float('nan'))
        return alpha, mask

    def extra_004(self, pos):
        data1 = self.en(self.S_DQ_VOLUME, pos) * (2 * self.en(self.S_FWDS_ADJCLOSE, pos) - self.en(self.S_FWDS_ADJLOW, pos) - self.en(self.S_FWDS_ADJHIGH, pos)) / (self.en(self.S_FWDS_ADJHIGH, pos) - self.en(self.S_FWDS_ADJLOW, pos))
        data1_mask = self.process_nan_infinite_and_mask(data1, self.en(self.MASK, pos))
        x = self.ewma(data1, data1_mask, alpha=2.0/9)[-1]
        y = self.ewma(data1, data1_mask, alpha=2.0/4)[-1]
        alpha = (x - y)
        mask = self.p(self.MASK, pos)
        alpha.masked_fill_(~mask, float('nan'))
        return alpha, mask

    def stock015(self, pos):
        df1, df1_mask = self.bn(self.S_FWDS_ADJCLOSE, pos, 21).clone(), self.bn(self.MASK, pos, 21) # [21, stock]
        df1.masked_fill_(~df1_mask, float('nan'))
        df1 = df1[1:] < df1[:-1] # [20, stock]
        df1_mask = t.ones_like(df1).bool()  # [20, stock]
        sumif = t.abs(self.bn(self.S_FWDS_ADJCLOSE, pos, 20) / self.bn(self.S_FWDS_ADJCLOSE, pos-1, 20) - 1) / t.log(self.bn(self.S_DQ_AMOUNT, pos, 20) + 1)  # [20, stock]
        sumif_mask = self.bn(self.MASK, pos, 20) & self.bn(self.MASK, pos-1, 20)  # [20, stock]
        sumif.masked_fill_(~df1, 0)  # [20, stock]
        sumif_mask = self.process_nan_infinite_and_mask(sumif, t.ones_like(sumif).bool())
        sumif, sumif_mask = self.sum(sumif, sumif_mask, dim=0)  # [stock]
        count, count_mask = self.sum(df1, df1_mask, dim=0) # [stock]
        alpha = (sumif / count) # [stock]
        mask = sumif_mask & count_mask & self.p(self.MASK, pos) # [stock]
        alpha.masked_fill_(~mask, float('nan'))
        return alpha, mask

    def stock007(self, pos):
        filled, filled_mask = self.fill_masked_with_last(self.bn(self.S_FWDS_ADJCLOSE, pos, 65), self.bn(self.MASK, pos, 65))
        ret, ret_mask = self.pct_change(self.bn(filled, 64, 61), self.bn(filled_mask, 64, 61)) # [60, stock]
        temp1, temp1_mask = self.min(self.bn(self.S_FWDS_ADJLOW, pos, 5), self.bn(self.MASK, pos, 5), dim=0) # [stock]
        temp1_d5, temp1_d5_mask = self.min(self.bn(self.S_FWDS_ADJLOW, pos-5, 5), self.bn(self.MASK, pos-5, 5), dim=0) # [stock]
        part1, part1_mask = temp1_d5 - temp1, temp1_mask & temp1_d5_mask  # [stock]
        temp2_1, temp2_1_mask = self.sum(ret, ret_mask, dim=0) # [stock]
        temp2_2, temp2_2_mask = self.sum(ret[-20:], ret_mask[-20:], dim=0) # [stock]
        temp2 = (temp2_1 - temp2_2) / 40 # [stock]
        temp2_mask = temp2_1_mask & temp2_2_mask # [stock]
        part2, part2_mask = self.rank(temp2, temp2_mask, dim=0)  # [stock]
        part3, part3_mask = self.rank(self.p(self.S_DQ_VOLUME, pos), self.p(self.MASK, pos), dim=0) # [stock]
        alpha = part1 * part2 * part3
        mask = part1_mask & part2_mask & part3_mask
        alpha.masked_fill_(~mask, float('nan'))
        return alpha, mask

    def old_037(self, pos):
        part1, part1_mask = self.rank(self.bn(self.S_FWDS_ADJOPEN, pos, 10), self.bn(self.MASK, pos, 10), dim=1) # [10, stock]
        part2, part2_mask = self.rank(self.bn(self.S_DQ_VOLUME, pos, 10), self.bn(self.MASK, pos, 10), dim=1) # [10, stock]
        alpha, alpha_mask = self.corr(part1, part2, (part1_mask & part2_mask), dim=0)  # [stock]
        alpha = -alpha
        alpha.masked_fill_(~alpha_mask, float('nan'))
        return alpha, alpha_mask

    def stock017(self, pos):
        data, data_mask = self.mean(self.bn(self.S_DQ_VOLUME, pos, 20), self.bn(self.MASK, pos, 20), dim=0)  # [stock]
        alpha = (-1) * (self.p(self.S_FWDS_ADJCLOSE, pos) / self.p(self.S_FWDS_ADJCLOSE, pos-1) - 1) * data * self.p(self.S_DQ_AVGPRICE, pos) * (self.p(self.S_FWDS_ADJHIGH, pos) - self.p(self.S_FWDS_ADJCLOSE, pos))
        mask = self.p(self.MASK, pos) & data_mask
        alpha.masked_fill_(~mask, float('nan'))
        return alpha, mask

    def extra_003(self, pos):
        alpha = self.p(self.S_FWDS_ADJOPEN, pos) / self.p(self.S_FWDS_ADJCLOSE, pos-1) - 1
        mask = self.p(self.MASK, pos) & self.p(self.MASK, pos-1)
        alpha.masked_fill_(~mask, float('nan'))
        return alpha, mask

    def stock021(self, pos):
        sma = self.ewma(self.en(self.S_FWDS_ADJHIGH, pos) - self.en(self.S_FWDS_ADJLOW, pos), self.en(self.MASK, pos), alpha=2/11)[-1]
        alpha = ((self.p(self.S_FWDS_ADJHIGH, pos) - self.p(self.S_FWDS_ADJLOW, pos) - sma) / sma * 100)
        mask = self.p(self.MASK, pos)
        alpha.masked_fill_(~mask, float('nan'))
        return alpha, mask

    def extra_001(self, pos):
        part1 = (2 * self.p(self.S_FWDS_ADJCLOSE, pos) - self.p(self.S_FWDS_ADJLOW, pos) - self.p(self.S_FWDS_ADJHIGH, pos)) / (self.p(self.S_FWDS_ADJHIGH, pos) - self.p(self.S_FWDS_ADJLOW, pos))
        part2 = (2 * self.p(self.S_FWDS_ADJCLOSE, pos-1) - self.p(self.S_FWDS_ADJLOW, pos-1) - self.p(self.S_FWDS_ADJHIGH, pos-1)) / (self.p(self.S_FWDS_ADJHIGH, pos-1) - self.p(self.S_FWDS_ADJLOW, pos-1))
        alpha = -(part1 - part2)
        mask = self.p(self.MASK, pos) & self.p(self.MASK, pos-1)
        alpha.masked_fill_(~mask, float('nan'))
        return alpha, mask

    def stock008(self, pos):
        filled, filled_mask = self.fill_masked_with_last(self.bn(self.S_FWDS_ADJCLOSE, pos, 65), self.bn(self.MASK, pos, 65))
        ret, ret_mask = self.pct_change(self.rg(filled, 64, 5, 12), self.rg(filled_mask, 64, 5, 12)) # [5+10, stock]
        temp1, temp1_mask = self.rolling_sum(self.rg(self.S_FWDS_ADJOPEN, pos, 5, 11), self.rg(self.MASK, pos, 5, 11), 5) # [11, stock]
        temp2, temp2_mask = self.rolling_sum(ret, ret_mask, 5) # [11, stock]
        temp = temp1 * temp2
        temp_mask = temp1_mask & temp2_mask
        part, part_mask  = temp[10] - temp[0], temp_mask[10] & temp_mask[0] # [stock]
        alpha, alpha_mask = self.rank(part, part_mask, dim=0)
        alpha = -alpha
        alpha.masked_fill_(~alpha_mask, float('nan'))
        return alpha, alpha_mask

    def stock019(self, pos):
        data1 = -1 * (self.p(self.S_FWDS_ADJLOW, pos) - self.p(self.S_FWDS_ADJCLOSE, pos)) * (self.p(self.S_FWDS_ADJOPEN, pos) ** 5)  # [stock]
        data1_mask = self.p(self.MASK, pos)
        data2 = (self.p(self.S_FWDS_ADJCLOSE, pos) - self.p(self.S_FWDS_ADJHIGH, pos)) * (self.p(self.S_FWDS_ADJCLOSE, pos) ** 5)  # [stock]
        data2_mask = self.p(self.MASK, pos)
        alpha = (data1 / data2)
        mask = data1_mask & data2_mask
        alpha.masked_fill_(~mask, float('nan'))
        return alpha, mask

    def stock020(self, pos):
        alpha = ((self.p(self.S_FWDS_ADJCLOSE, pos) - self.p(self.S_FWDS_ADJCLOSE, pos-1)) / self.p(self.S_FWDS_ADJCLOSE, pos-1) * self.p(self.S_DQ_VOLUME, pos))  # [stock]
        mask = self.p(self.MASK, pos) & self.p(self.MASK, pos-1)
        alpha.masked_fill_(~mask, float('nan'))
        return alpha, mask

    def stock010(self, pos):
        alpha, alpha_mask = self.std(t.log(self.bn(self.S_DQ_AMOUNT, pos, 6) + 1), self.bn(self.MASK, pos, 6), dim=0)  # [stock]
        alpha.masked_fill_(~alpha_mask, float('nan'))
        return alpha, alpha_mask

    def old_032(self, pos):
        # cmp:    old_032 20210104    0   300059  46222952783509.75   1564.0  -9064995.0  -2266248.5873508453
        alpha, alpha_mask = self.cov(self.bn(self.S_FWDS_ADJCLOSE, pos, 5), self.bn(self.S_DQ_VOLUME, pos, 5), self.bn(self.MASK, pos, 5), dim=0)  # [stock]
        alpha = -alpha
        alpha.masked_fill_(~alpha_mask, float('nan'))
        return alpha, alpha_mask

    def stock012(self, pos):
        temp1 = self.en(self.S_DQ_VOLUME, pos) - self.en(self.S_DQ_VOLUME, pos-1)  # [ewma, stock]
        temp1_mask = self.en(self.MASK, pos) & self.en(self.MASK, pos-1)  # [ewma, stock]
        temp1.masked_fill_(~temp1_mask, float('nan'))
        part1 = t.maximum(temp1, t.zeros_like(temp1))
        part1 = self.ewma(part1, temp1_mask, alpha=1.0/6)[-1]
        temp2 = temp1.abs()
        part2 = self.ewma(temp2, temp1_mask, alpha=1.0/6)[-1]
        alpha = part1 * 100 / part2
        mask = self.p(self.MASK, pos)
        alpha.masked_fill_(~mask, float('nan'))
        return alpha, mask

    def stock014(self, pos):
        alpha = (self.p(self.S_FWDS_ADJCLOSE, pos) / self.p(self.S_FWDS_ADJCLOSE, pos-12) - 1) * self.p(self.S_DQ_VOLUME, pos)
        mask = self.p(self.MASK, pos) & self.p(self.MASK, pos-12)
        alpha.masked_fill_(~mask, float('nan'))
        return alpha, mask

    def stock004(self, pos):
        part0, part0_mask = self.rolling_mean(self.eg(self.S_FWDS_ADJCLOSE, pos, 6), self.eg(self.MASK, pos, 6), 6)  # [ewma, stock]
        part1 = (self.en(self.S_FWDS_ADJCLOSE, pos) - part0) / part0  # [ewma, stock]
        part1_mask = self.en(self.MASK, pos) & part0_mask  # [ewma, stock]
        alpha = part1[3:] - part1[:-3] # [ewma-3, stock]
        alpha_mask = part1_mask[3:] & part1_mask[:-3] # [ewma-3, stock]
        alpha = self.ewma(alpha, alpha_mask, alpha=1.0/12)[-1] # [stock]
        mask = self.p(self.MASK, pos)
        alpha.masked_fill_(~mask, float('nan'))
        return alpha, mask

    def best_021(self, pos):
        part1, part1_mask = self.max((self.bn(self.S_DQ_AVGPRICE, pos, 3) - self.bn(self.S_FWDS_ADJCLOSE, pos, 3)) * self.bn(self.S_DQ_VOLUME, pos, 3), self.bn(self.MASK, pos, 3), dim=0)  # [stock]
        part2, part2_mask = self.min((self.bn(self.S_DQ_AVGPRICE, pos, 3) - self.bn(self.S_FWDS_ADJCLOSE, pos, 3)) * self.bn(self.S_DQ_VOLUME, pos, 3), self.bn(self.MASK, pos, 3), dim=0)  # [stock]
        part3 = self.p(self.S_DQ_VOLUME, pos) - self.p(self.S_DQ_VOLUME, pos-3)
        part3_mask = self.p(self.MASK, pos) & self.p(self.MASK, pos-3) 
        alpha = part1 + part2 * part3
        mask = part1_mask & part2_mask & part3_mask
        alpha.masked_fill_(~mask, float('nan'))
        return alpha, mask

    def stock006(self, pos):
        part, part_mask = self.mean(self.bn(self.S_FWDS_ADJCLOSE, pos, 12), self.bn(self.MASK, pos, 12), dim=0)  # [stock]
        alpha = (self.p(self.S_FWDS_ADJCLOSE, pos) - part) / part
        mask = part_mask & self.p(self.MASK, pos)
        alpha.masked_fill_(~mask, float('nan'))
        return alpha, mask

    def stock013(self, pos):
        data1 = (self.p(self.S_DQ_AVGPRICE, pos) - self.p(self.S_FWDS_ADJCLOSE, pos)) - (self.p(self.S_DQ_AVGPRICE, pos-1) - self.p(self.S_FWDS_ADJCLOSE, pos-1))
        data1_mask = self.p(self.MASK, pos) & self.p(self.MASK, pos-1)
        data2 = (self.p(self.S_DQ_AVGPRICE, pos) + self.p(self.S_FWDS_ADJCLOSE, pos)) - (self.p(self.S_DQ_AVGPRICE, pos-1) + self.p(self.S_FWDS_ADJCLOSE, pos-1))
        data2_mask = self.p(self.MASK, pos) & self.p(self.MASK, pos-1)
        alpha = (data1 / data2)
        mask = data1_mask & data2_mask
        alpha.masked_fill_(~mask, float('nan'))
        return alpha, mask

    def stock003(self, pos):
        # cmp:    stock003    20210104    0   605009  inf 4130.0  inf 2.6166185146384057e+88
        temp1, temp1_mask = self.max(self.bn(self.S_DQ_AVGPRICE, pos, 15), self.bn(self.MASK, pos, 15), dim=0)   # [stock]
        #print('stock003 temp1:', temp1.shape, temp1_mask.shape, temp1[0]
        temp2 = (self.p(self.S_FWDS_ADJCLOSE, pos) - temp1) # [stock]
        temp2_mask = temp1_mask & self.p(self.MASK, pos) # [stock]
        part1, part1_mask = self.rank(temp2, temp2_mask, dim=0) # [stock]
        part2 = self.p(self.S_FWDS_ADJCLOSE, pos) - self.p(self.S_FWDS_ADJCLOSE, pos-5)  # [stock]
        part2_mask = self.p(self.MASK, pos) & self.p(self.MASK, pos-5)  # [stock]
        alpha = part1 ** part2
        mask = part1_mask & part2_mask
        alpha.masked_fill_(~mask, float('nan'))
        return alpha, mask

    def stock005(self, pos):
        delay6, delay6_mask = self.p(self.S_FWDS_ADJCLOSE, pos-6), self.p(self.MASK, pos-6)
        alpha = (self.p(self.S_FWDS_ADJCLOSE, pos) - delay6) * (self.p(self.S_DQ_VOLUME, pos) + 1) / delay6
        mask = delay6_mask & self.p(self.MASK, pos)
        alpha.masked_fill_(~mask, float('nan'))
        return alpha, mask

    def best_020(self, pos):
        alpha = self.ewma((self.en(self.S_FWDS_ADJCLOSE, pos) - self.en(self.S_DQ_AVGPRICE, pos)) * self.en(self.S_DQ_VOLUME, pos), self.en(self.MASK, pos), alpha=1/5.0)[-1]
        mask = self.p(self.MASK, pos)
        alpha.masked_fill_(~mask, float('nan'))
        return alpha, mask

    def best_019(self, pos):
        part0 = self.ewma(self.en(self.S_FWDS_ADJCLOSE, pos), self.en(self.MASK, pos), alpha=1/20.0)  # [ewma, stock]
        alpha = part0[-1] - part0[-2]
        mask = self.p(self.MASK, pos)
        alpha.masked_fill_(~mask, float('nan'))
        return alpha, mask

    def best_018(self, pos):
        part0 = self.ewma(self.en(self.S_FWDS_ADJCLOSE, pos), self.en(self.MASK, pos), alpha=1/10.0)  # [ewma, stock]
        alpha = part0[-1] - part0[-2]
        mask = self.p(self.MASK, pos)
        alpha.masked_fill_(~mask, float('nan'))
        return alpha, mask

    def best_012(self, pos):
        alpha, alpha_mask = self.mean(self.bn(self.S_FWDS_ADJOPEN, pos, 5) / self.bn(self.S_FWDS_ADJCLOSE, pos-1, 5) - 1, (self.bn(self.MASK, pos, 5) & self.bn(self.MASK, pos-1, 5)), dim=0)  # [stock]
        alpha.masked_fill_(~alpha_mask, float('nan'))
        return alpha, alpha_mask

    def best_017(self, pos):
        part0 = self.ewma(self.en(self.S_FWDS_ADJCLOSE, pos), self.en(self.MASK, pos), alpha=1/5.0)  # [ewma, stock]
        alpha = part0[-1] - part0[-2]
        mask = self.p(self.MASK, pos)
        alpha.masked_fill_(~mask, float('nan'))
        return alpha, mask

    def best_015(self, pos):
        data = (self.p(self.S_FWDS_ADJHIGH, pos) - self.p(self.S_FWDS_ADJLOW, pos)) / self.p(self.S_FWDS_ADJCLOSE, pos)  # [stock]
        data_mask = self.p(self.MASK, pos)  # [stock]
        alpha, alpha_mask = self.rank(data, data_mask, dim=0)  # [stock]
        alpha.masked_fill_(~alpha_mask, float('nan'))
        return alpha, alpha_mask

    def best_016(self, pos):
        alpha = ((self.p(self.S_FWDS_ADJHIGH, pos) - self.p(self.S_FWDS_ADJLOW, pos)) / self.p(self.S_FWDS_ADJCLOSE, pos)) / (1 + t.sqrt(self.p(self.S_DQ_VOLUME, pos)))
        mask = self.p(self.MASK, pos)
        alpha.masked_fill_(~mask, float('nan'))
        return alpha, mask

    def best_008(self, pos):
        alpha, alpha_mask = self.mean(self.bn(self.S_DQ_AVGPRICE, pos, 3) / self.bn(self.S_FWDS_ADJCLOSE, pos, 3), self.bn(self.MASK, pos, 3), dim=0)  # [stock]
        alpha.masked_fill_(~alpha_mask, float('nan'))
        return alpha, alpha_mask

    def best_013(self, pos):
        alpha = self.ewma(self.en(self.S_FWDS_ADJOPEN, pos) / self.en(self.S_FWDS_ADJCLOSE, pos-1) - 1, (self.en(self.MASK, pos) & self.en(self.MASK, pos-1)), alpha=1/5.0)[-1]
        mask = self.p(self.MASK, pos)
        alpha.masked_fill_(~mask, float('nan'))
        return alpha, mask

    def best_009(self, pos):
        part1 = self.ewma(self.en(self.S_DQ_AVGPRICE, pos), self.en(self.MASK, pos), alpha=1/5.0)[-1]
        part2 = self.ewma(self.en(self.S_FWDS_ADJCLOSE, pos), self.en(self.MASK, pos), alpha=1/5.0)[-1]
        alpha = part1 / part2
        mask = self.p(self.MASK, pos)
        alpha.masked_fill_(~mask, float('nan'))
        return alpha, mask

    def best_011(self, pos):
        data = self.p(self.S_FWDS_ADJOPEN, pos) / self.p(self.S_FWDS_ADJCLOSE, pos-1) - 1 # [stock]
        data_mask = self.p(self.MASK, pos) & self.p(self.MASK, pos-1) # [stock]
        alpha, alpha_mask = self.rank(data, data_mask, dim=0) # [stock]
        alpha.masked_fill_(~alpha_mask, float('nan'))
        return alpha, alpha_mask

    def old_042(self, pos):
        data1, data1_mask = self.bn(self.S_FWDS_ADJHIGH, pos, 20) - self.bn(self.S_FWDS_ADJOPEN, pos, 20), self.bn(self.MASK, pos, 20)  # [20, stock]
        data2, data2_mask = self.bn(self.S_FWDS_ADJOPEN, pos, 20) - self.bn(self.S_FWDS_ADJLOW, pos, 20), self.bn(self.MASK, pos, 20)  # [20, stock]
        data3, data3_mask = self.sum(data1, data1_mask, dim=0)  # [stock]
        data4, data4_mask = self.sum(data2, data2_mask, dim=0)  # [stock]
        alpha = data3 / data4
        mask = data3_mask & data4_mask
        alpha.masked_fill_(~mask, float('nan'))
        return alpha, mask

    def better_012(self, pos):
        data1 = t.minimum(self.bn(self.S_FWDS_ADJLOW, pos, 24), self.bn(self.S_DQ_AVGPRICE, pos-1, 24))  # [24, stock]
        data2 = t.maximum(self.bn(self.S_FWDS_ADJHIGH, pos, 24), self.bn(self.S_DQ_AVGPRICE, pos-1, 24))  # [24, stock]
        data_mask = self.bn(self.MASK, pos, 24) & self.bn(self.MASK, pos-1, 24)  # [24, stock]
        part1_1, part1_1_mask = self.sum(data1[ -6:], data_mask[ -6:], dim=0)  # [stock]
        part2_1, part2_1_mask = self.sum(data1[-12:], data_mask[-12:], dim=0)  # [stock]
        part3_1, part3_1_mask = self.sum(data1[-24:], data_mask[-24:], dim=0)  # [stock]
        part1_2, part1_2_mask = self.sum(data2[ -6:] - data1[ -6:], data_mask[ -6:], dim=0)  # [stock]
        part2_2, part2_2_mask = self.sum(data2[-12:] - data1[-12:], data_mask[-12:], dim=0)  # [stock]
        part3_2, part3_2_mask = self.sum(data2[-24:] - data1[-24:], data_mask[-24:], dim=0)  # [stock]
        part1 = (self.p(self.S_DQ_AVGPRICE, pos) - part1_1) / part1_2 * 12 * 24  # [stock]
        part2 = (self.p(self.S_DQ_AVGPRICE, pos) - part2_1) / part2_2 * 6 * 24  # [stock]
        part3 = (self.p(self.S_DQ_AVGPRICE, pos) - part3_1) / part3_2 * 6 * 24  # [stock]
        mask = self.p(self.MASK, pos) & part1_1_mask & part2_1_mask & part3_1_mask
        alpha = (part1 + part2 + part3) * 100 / (6 * 12 + 6 * 24 + 12 * 24)
        alpha.masked_fill_(~mask, float('nan'))
        return alpha, mask

    def old_040(self, pos):
        part1, part1_mask = self.bn(self.S_FWDS_ADJCLOSE, pos, 12).clone(), self.bn(self.MASK, pos, 12)   # [12, stock]
        part2, part2_mask = self.bn(self.S_FWDS_ADJCLOSE, pos-1, 12).clone(), self.bn(self.MASK, pos-1, 12)   # [12, stock]
        part1.masked_fill_(~part1_mask, float('nan'))
        part2.masked_fill_(~part2_mask, float('nan'))
        cond1 = part1 > part2  # [12, stock]
        cond2 = part1 < part2  # [12, stock]
        data1 = part1 - part2  # [12, stock]
        data1_mask = part1_mask & part2_mask  # [12, stock]
        data2 = data1.clone()  # [12, stock]
        data2_mask = data1_mask.clone()  # [12, stock]
        data1[~cond1] = 0  # [12, stock]
        data1_mask[~cond1] = True  # [12, stock]
        data2[~cond2] = 0  # [12, stock]
        data2_mask[~cond2] = True  # [12, stock]
        data2 = data2.abs()  # [12, stock]
        sum1, sum1_mask = self.sum(data1, data1_mask, dim=0)  # [stock]
        sum2, sum2_mask = self.sum(data2, data2_mask, dim=0)  # [stock]
        alpha = ((sum1 - sum2) / (sum1 + sum2) * 100)
        mask = sum1_mask & sum2_mask & self.p(self.MASK, pos)
        alpha.masked_fill_(~mask, float('nan'))
        return alpha, mask

    def best_010(self, pos):
        alpha = (self.p(self.S_DQ_AVGPRICE, pos) / self.p(self.S_FWDS_ADJCLOSE, pos) - 1) * self.p(self.S_DQ_VOLUME, pos)
        mask = self.p(self.MASK, pos)
        alpha.masked_fill_(~mask, float('nan'))
        return alpha, mask

    def better_011(self, pos):
        data1 = t.minimum(self.bn(self.S_FWDS_ADJLOW, pos, 24), self.bn(self.S_FWDS_ADJLOW, pos-1, 24))  # [24, stock]
        data2 = t.maximum(self.bn(self.S_FWDS_ADJHIGH, pos, 24), self.bn(self.S_FWDS_ADJHIGH, pos-1, 24))  # [24, stock]
        data_mask = self.bn(self.MASK, pos, 24) & self.bn(self.MASK, pos-1, 24)  # [24, stock]
        part1_1, part1_1_mask = self.sum(data1[ -6:], data_mask[ -6:], dim=0)  # [stock]
        part2_1, part2_1_mask = self.sum(data1[-12:], data_mask[-12:], dim=0)  # [stock]
        part3_1, part3_1_mask = self.sum(data1[-24:], data_mask[-24:], dim=0)  # [stock]
        part1_2, part1_2_mask = self.sum(data2[ -6:] - data1[ -6:], data_mask[ -6:], dim=0)  # [stock]
        part2_2, part2_2_mask = self.sum(data2[-12:] - data1[-12:], data_mask[-12:], dim=0)  # [stock]
        part3_2, part3_2_mask = self.sum(data2[-24:] - data1[-24:], data_mask[-24:], dim=0)  # [stock]
        part1 = (self.p(self.S_FWDS_ADJCLOSE, pos) - part1_1) / part1_2 * 12 * 24  # [stock]
        part2 = (self.p(self.S_FWDS_ADJCLOSE, pos) - part2_1) / part2_2 * 6 * 24  # [stock]
        part3 = (self.p(self.S_FWDS_ADJCLOSE, pos) - part3_1) / part3_2 * 6 * 24  # [stock]
        mask = self.p(self.MASK, pos) & part1_1_mask & part2_1_mask & part3_1_mask
        alpha = (part1 + part2 + part3) * 100 / (6 * 12 + 6 * 24 + 12 * 24)
        alpha.masked_fill_(~mask, float('nan'))
        return alpha, mask

    def old_030(self, pos):
        part1, part1_mask = self.rolling_min(self.eg(self.S_FWDS_ADJLOW, pos, 9), self.eg(self.MASK, pos, 9), 9)  # [ewma, stock]
        part2, part2_mask = self.rolling_max(self.eg(self.S_FWDS_ADJHIGH, pos, 9), self.eg(self.MASK, pos, 9), 9)  # [ewma, stock]
        part3, part3_mask = self.rolling_min(self.eg(self.S_FWDS_ADJLOW, pos, 9), self.eg(self.MASK, pos, 9), 9)  # [ewma, stock]
        part_mask = part1_mask & part2_mask & part3_mask
        sma1 = self.ewma(100 * (self.en(self.S_FWDS_ADJCLOSE, pos) - part1) / (part2 - part3), part_mask, alpha=1.0/3) # [ewma, stock]
        sma1_mask = self.process_nan_infinite_and_mask(sma1, t.ones_like(sma1).bool())
        alpha = self.ewma(sma1, sma1_mask, alpha=1.0/3)[-1]   # [stock]
        mask = self.p(self.MASK, pos)
        alpha.masked_fill_(~mask, float('nan'))
        return alpha, mask

    def best_007(self, pos):
        data = -((2 * self.p(self.S_FWDS_ADJCLOSE, pos) - self.p(self.S_FWDS_ADJLOW, pos) - self.p(self.S_FWDS_ADJHIGH, pos)) / (self.p(self.S_FWDS_ADJHIGH, pos) - self.p(self.S_FWDS_ADJLOW, pos)))  #[stock]
        data_mask = self.process_nan_infinite_and_mask(data, self.p(self.MASK, pos))  #[stock]
        alpha, alpha_mask = self.rank(data, data_mask, dim=0)  #[stock]
        alpha.masked_fill_(~alpha_mask, float('nan'))
        return alpha, alpha_mask

    def old_044(self, pos):
        log_close = t.log(self.en(self.S_FWDS_ADJCLOSE, pos) + 1)
        log_close_mask = self.en(self.MASK, pos)
        data = self.ewma(log_close, log_close_mask, alpha=2/13)
        data_mask = self.process_nan_infinite_and_mask(data, t.ones_like(data).bool())
        data = self.ewma(data, data_mask, alpha=2/13)
        data_mask = self.process_nan_infinite_and_mask(data, t.ones_like(data).bool())
        data = self.ewma(data, data_mask, alpha=2/13)
        data_mask = self.process_nan_infinite_and_mask(data, t.ones_like(data).bool())
        alpha = data[-1] / data[-2] - 1
        mask = data_mask[-1] & data_mask[-2] & self.p(self.MASK, pos)
        alpha.masked_fill_(~mask, float('nan'))
        return alpha, mask

    def best_005(self, pos):
        part1 = (2 * self.p(self.S_FWDS_ADJCLOSE, pos) - self.p(self.S_FWDS_ADJLOW, pos) - self.p(self.S_FWDS_ADJHIGH, pos)) / (self.p(self.S_FWDS_ADJHIGH, pos) - self.p(self.S_FWDS_ADJLOW, pos))
        part2 = (2 * self.p(self.S_FWDS_ADJCLOSE, pos-5) - self.p(self.S_FWDS_ADJLOW, pos-5) - self.p(self.S_FWDS_ADJHIGH, pos-5)) / (self.p(self.S_FWDS_ADJHIGH, pos-5) - self.p(self.S_FWDS_ADJLOW, pos-5))
        alpha = -(part1 - part2)
        mask = self.p(self.MASK, pos) & self.p(self.MASK, pos-5)
        alpha.masked_fill_(~mask, float('nan'))
        return alpha, mask

    def best_004(self, pos):
        part1 = (2 * self.p(self.S_FWDS_ADJCLOSE, pos) - self.p(self.S_FWDS_ADJLOW, pos) - self.p(self.S_FWDS_ADJHIGH, pos)) / (self.p(self.S_FWDS_ADJHIGH, pos) - self.p(self.S_FWDS_ADJLOW, pos))
        part2 = (2 * self.p(self.S_FWDS_ADJCLOSE, pos-3) - self.p(self.S_FWDS_ADJLOW, pos-3) - self.p(self.S_FWDS_ADJHIGH, pos-3)) / (self.p(self.S_FWDS_ADJHIGH, pos-3) - self.p(self.S_FWDS_ADJLOW, pos-3))
        alpha = -(part1 - part2)
        mask = self.p(self.MASK, pos) & self.p(self.MASK, pos-3)
        alpha.masked_fill_(~mask, float('nan'))
        return alpha, mask

    def old_043(self, pos):
        data1, data1_mask = (self.p(self.S_DQ_AVGPRICE, pos) - self.p(self.S_FWDS_ADJCLOSE, pos)), self.p(self.MASK, pos) # [stock]
        data1, data1_mask = self.rank(data1, data1_mask, dim=0) # [stock]
        data2, data2_mask = (self.p(self.S_DQ_AVGPRICE, pos) + self.p(self.S_FWDS_ADJCLOSE, pos)), self.p(self.MASK, pos) # [stock]
        data2, data2_mask = self.rank(data2, data2_mask, dim=0) # [stock]
        alpha = (data1 / data2)
        mask = data1_mask & data2_mask
        alpha.masked_fill_(~mask, float('nan'))
        return alpha, mask

    def better_025(self, pos):
        part1, part1_mask = self.max(self.bn(self.S_DQ_AVGPRICE, pos, 5) - self.bn(self.S_FWDS_ADJCLOSE, pos, 5), self.bn(self.MASK, pos, 5), dim=0)  # [stock]
        part2, part2_mask = self.min(self.bn(self.S_DQ_AVGPRICE, pos, 5) - self.bn(self.S_FWDS_ADJCLOSE, pos, 5), self.bn(self.MASK, pos, 5), dim=0)  # [stock]
        part3, part3_mask = self.mean(self.bn(self.S_DQ_VOLUME, pos, 5), self.bn(self.MASK, pos, 5), dim=0)  # [stock]
        part3 = self.p(self.S_DQ_VOLUME, pos) - part3  # [stock]
        part3_mask = self.p(self.MASK, pos) & part3_mask
        alpha = (part1 + part2) * part3
        mask = part1_mask & part2_mask & part3_mask
        alpha.masked_fill_(~mask, float('nan'))
        return alpha, mask

    def old_039(self, pos):
        data = self.en(self.S_FWDS_ADJHIGH, pos) - self.en(self.S_FWDS_ADJLOW, pos)  # [ewma, stock]
        data_mask = self.en(self.MASK, pos)  # [ewma, stock]
        sma1 = self.ewma(data, data_mask, alpha=1.0/10)  # [ewma, stock]
        sma1_mask = self.process_nan_infinite_and_mask(sma1, t.ones_like(sma1).bool()) # [ewma, stock]
        sma2 = self.ewma(sma1, sma1_mask, alpha=1.0/10)  # [ewma, stock]
        alpha = sma1[-1] / sma2[-1]
        mask = self.p(self.MASK, pos)
        alpha.masked_fill_(~mask, float('nan'))
        return alpha, mask

    def old_033(self, pos):
        alpha, alpha_mask = self.std(self.bn(self.S_DQ_VOLUME, pos, 20), self.bn(self.MASK, pos, 20), dim=0)
        alpha.masked_fill_(~alpha_mask, float('nan'))
        return alpha, alpha_mask

    def better_028(self, pos):
        df1, df1_mask = self.bn(self.S_DQ_AVGPRICE, pos, 20).clone(), self.bn(self.MASK, pos, 20) # [20, stock]
        df1.masked_fill_(~df1_mask, float('nan'))
        df2, df2_mask = self.bn(self.S_FWDS_ADJCLOSE, pos, 20).clone(), self.bn(self.MASK, pos, 20) # [20, stock]
        df2.masked_fill_(~df2_mask, float('nan'))
        df1 = df1 < df2 # [20, stock]
        df1_mask = t.ones_like(df1).bool()  # [20, stock]
        sumif = t.abs(self.bn(self.S_DQ_AVGPRICE, pos, 20) / self.bn(self.S_FWDS_ADJCLOSE, pos, 20) - 1) / t.log(self.bn(self.S_DQ_AMOUNT, pos, 20) + 1)  # [20, stock]
        sumif_mask = self.bn(self.MASK, pos, 20) & self.bn(self.MASK, pos-1, 20)  # [20, stock]
        sumif.masked_fill_(~df1, 0)  # [20, stock]
        sumif_mask = self.process_nan_infinite_and_mask(sumif, t.ones_like(sumif).bool())
        sumif, sumif_mask = self.sum(sumif, sumif_mask, dim=0)  # [stock]
        count, count_mask = self.sum(df1, df1_mask, dim=0) # [stock]
        alpha = (sumif / count) # [stock]
        mask = sumif_mask & count_mask & self.p(self.MASK, pos) # [stock]
        alpha.masked_fill_(~mask, float('nan'))
        return alpha, mask

    def old_031(self, pos):
        alpha, alpha_mask = self.std(self.bn(self.S_DQ_VOLUME, pos, 10), self.bn(self.MASK, pos, 10), dim=0) 
        alpha.masked_fill_(~alpha_mask, float('nan'))
        return alpha, alpha_mask

    def better_027(self, pos):
        df1, df1_mask = self.bn(self.S_DQ_AVGPRICE, pos, 20).clone(), self.bn(self.MASK, pos, 20) # [20, stock]
        df1.masked_fill_(~df1_mask, float('nan'))
        df2, df2_mask = self.bn(self.S_FWDS_ADJCLOSE, pos-1, 20).clone(), self.bn(self.MASK, pos-1, 20) # [20, stock]
        df2.masked_fill_(~df2_mask, float('nan'))
        df1 = df1 < df2
        df1_mask = t.ones_like(df1).bool()  # [20, stock]
        sumif = t.abs(self.bn(self.S_DQ_AVGPRICE, pos, 20) / self.bn(self.S_FWDS_ADJCLOSE, pos-1, 20) - 1) / t.log(self.bn(self.S_DQ_AMOUNT, pos, 20) + 1)  # [20, stock]
        sumif_mask = self.bn(self.MASK, pos, 20) & self.bn(self.MASK, pos-1, 20)  # [20, stock]
        sumif.masked_fill_(~df1, 0)  # [20, stock]
        sumif_mask = self.process_nan_infinite_and_mask(sumif, t.ones_like(sumif).bool())
        sumif, sumif_mask = self.sum(sumif, sumif_mask, dim=0)  # [stock]
        count, count_mask = self.sum(df1, df1_mask, dim=0) # [stock]
        alpha = (sumif / count) # [stock]
        mask = sumif_mask & count_mask & self.p(self.MASK, pos) # [stock]
        alpha.masked_fill_(~mask, float('nan'))
        return alpha, mask

    def old_038(self, pos):
        rank1, rank1_mask = (self.p(self.S_FWDS_ADJOPEN, pos) - self.p(self.S_FWDS_ADJHIGH, pos-1)), (self.p(self.MASK, pos) & self.p(self.MASK, pos-1))
        rank2, rank2_mask = (self.p(self.S_FWDS_ADJOPEN, pos) - self.p(self.S_FWDS_ADJCLOSE, pos-1)), (self.p(self.MASK, pos) & self.p(self.MASK, pos-1))
        rank3, rank3_mask = (self.p(self.S_FWDS_ADJOPEN, pos) - self.p(self.S_FWDS_ADJLOW, pos-1)), (self.p(self.MASK, pos) & self.p(self.MASK, pos-1))
        rank1, rank1_mask = self.rank(rank1, rank1_mask, dim=0)
        rank1 = -rank1
        rank2, rank2_mask = self.rank(rank2, rank2_mask, dim=0)
        rank3, rank3_mask = self.rank(rank3, rank3_mask, dim=0)
        alpha = (rank1 * rank2 * rank3)
        mask = rank1_mask & rank2_mask & rank3_mask
        alpha.masked_fill_(~mask, float('nan'))
        return alpha, mask

    def old_029(self, pos):
        alpha, alpha_mask = self.std(self.bn(self.S_DQ_AMOUNT, pos, 20), self.bn(self.MASK, pos, 20), dim=0)
        alpha.masked_fill_(~alpha_mask, float('nan'))
        return alpha, alpha_mask

    def old_028(self, pos):
        data1, data1_mask = self.bn(self.S_FWDS_ADJCLOSE, pos, 20).clone(), self.bn(self.MASK, pos, 20)
        data1.masked_fill_(~data1_mask, float('nan'))
        delay1, delay1_mask = self.bn(self.S_FWDS_ADJCLOSE, pos-1, 20).clone(), self.bn(self.MASK, pos-1, 20)
        delay1.masked_fill_(~delay1_mask, float('nan'))
        part1, part1_mask = self.bn(self.S_DQ_VOLUME, pos, 20).clone(), self.bn(self.MASK, pos, 20).clone()
        cond1 = data1 <= delay1
        cond2 = data1 >= delay1
        part1[cond1] = 0
        part1_mask[cond1] = True
        part2, part2_mask = -self.bn(self.S_DQ_VOLUME, pos, 20).clone(), self.bn(self.MASK, pos, 20).clone()
        part2[cond2] = 0
        part2_mask[cond2] = True
        alpha, alpha_mask = self.sum(part1 + part2, (part1_mask & part2_mask), dim=0)
        mask = alpha_mask & self.p(self.MASK, pos)
        alpha.masked_fill_(~mask, float('nan'))
        return alpha, mask

    def better_020(self, pos):
        data1, data1_mask = self.std((self.bn(self.S_FWDS_ADJCLOSE, pos, 10) / self.bn(self.S_FWDS_ADJCLOSE, pos-1, 10) - 1).abs() / self.bn(self.S_DQ_VOLUME, pos, 10), self.bn(self.MASK, pos, 10), dim=0)
        data2, data2_mask = self.mean((self.bn(self.S_FWDS_ADJCLOSE, pos, 10) / self.bn(self.S_FWDS_ADJCLOSE, pos-1, 10) - 1).abs() / self.bn(self.S_DQ_VOLUME, pos, 10), self.bn(self.MASK, pos, 10), dim=0)
        alpha = (data1 / data2)
        mask = data1_mask & data2_mask
        alpha.masked_fill_(~mask, float('nan'))
        return alpha, mask

    def old_027(self, pos):
        part1, part1_mask = self.bn(self.S_FWDS_ADJOPEN, pos, 20).clone(), self.bn(self.MASK, pos, 20)
        part1.masked_fill_(~part1_mask, float('nan'))
        part2, part2_mask = self.bn(self.S_FWDS_ADJOPEN, pos-1, 20).clone(), self.bn(self.MASK, pos-1, 20)
        part2.masked_fill_(~part2_mask, float('nan'))
        condition = (part1 >= part2)
        temp = t.maximum(self.bn(self.S_FWDS_ADJOPEN, pos, 20) - self.bn(self.S_FWDS_ADJLOW, pos, 20), self.bn(self.S_FWDS_ADJOPEN, pos, 20) - self.bn(self.S_FWDS_ADJOPEN, pos-1, 20))
        temp_mask = self.bn(self.MASK, pos, 20) & self.bn(self.MASK, pos-1, 20)
        temp[condition] = 0
        temp_mask[condition] = True
        alpha, alpha_mask = self.sum(temp, temp_mask, dim=0)
        mask = alpha_mask & self.p(self.MASK, pos)
        alpha.masked_fill_(~mask, float('nan'))
        return alpha, mask

    def better_024(self, pos):
        part1, part1_mask = self.max(self.bn(self.S_DQ_AVGPRICE, pos, 5) - self.bn(self.S_FWDS_ADJCLOSE, pos, 5), self.bn(self.MASK, pos, 5), dim=0)  # [stock]
        part2, part2_mask = self.min(self.bn(self.S_DQ_AVGPRICE, pos, 5) - self.bn(self.S_FWDS_ADJCLOSE, pos, 5), self.bn(self.MASK, pos, 5), dim=0)  # [stock]
        part3 = self.p(self.S_DQ_VOLUME, pos) - self.p(self.S_DQ_VOLUME, pos-5)
        part3_mask = self.p(self.MASK, pos) & self.p(self.MASK, pos-5) 
        alpha = (part1 + part2) * part3
        mask = part1_mask & part2_mask & part3_mask
        alpha.masked_fill_(~mask, float('nan'))
        return alpha, mask

    def better_026(self, pos):
        data = (self.en(self.S_FWDS_ADJHIGH, pos) - self.en(self.S_FWDS_ADJLOW, pos)) / self.en(self.S_FWDS_ADJCLOSE, pos) # [ewma, stock]
        data_mask = self.en(self.MASK, pos) # [ewma, stock]
        sma1 = self.ewma(data, data_mask, alpha=2.0/5) # [ewma, stock]
        sma1_mask = self.process_nan_infinite_and_mask(sma1, t.ones_like(sma1).bool()) # [ewma, stock]
        sma2 = self.ewma(sma1, sma1_mask, alpha=2.0/20) # [ewma, stock]
        alpha = (sma1[-1] / sma2[-1])
        mask = self.p(self.MASK, pos)
        alpha.masked_fill_(~mask, float('nan'))
        return alpha, mask

    def better_023(self, pos):
        alpha, alpha_mask = self.mean((self.bn(self.S_DQ_AVGPRICE, pos, 4) / self.bn(self.S_FWDS_ADJCLOSE, pos-1, 4) - 1) * t.log2(self.bn(self.S_DQ_AMOUNT, pos, 4) + 1), (self.bn(self.MASK, pos, 4) & self.bn(self.MASK, pos-1, 4)), dim=0)
        alpha.masked_fill_(~alpha_mask, float('nan'))
        return alpha, alpha_mask

    def better_018(self, pos):
        part0, part0_mask = self.rolling_mean(self.eg(self.S_DQ_AMOUNT, pos, 10), self.eg(self.MASK, pos, 10), 10)  # [ewma, stock]
        part1 = (self.en(self.S_DQ_AMOUNT, pos) - part0) / part0  # [ewma, stock]
        part1_mask = self.en(self.MASK, pos) & part0_mask # [ewma, stock]
        alpha = part1[5:] - part1[:-5]
        alpha_mask = part1_mask[5:] & part1_mask[:-5]
        alpha = self.ewma(alpha, alpha_mask, alpha=1.0/20)[-1]  # [stock]
        mask = self.p(self.MASK, pos)
        alpha.masked_fill_(~mask, float('nan'))
        return alpha, mask

    def better_017(self, pos):
        part0, part0_mask = self.rolling_mean(self.eg(self.S_DQ_AMOUNT, pos, 6), self.eg(self.MASK, pos, 6), 6)  # [ewma, stock]
        part1 = (self.en(self.S_DQ_AMOUNT, pos) - part0) / part0  # [ewma, stock]
        part1_mask = self.en(self.MASK, pos) & part0_mask # [ewma, stock]
        alpha = part1[3:] - part1[:-3]
        alpha_mask = part1_mask[3:] & part1_mask[:-3]
        alpha = self.ewma(alpha, alpha_mask, alpha=1.0/12)[-1]  # [stock]
        mask = self.p(self.MASK, pos)
        alpha.masked_fill_(~mask, float('nan'))
        return alpha, mask

    def better_016(self, pos):
        part0, part0_mask = self.rolling_mean(self.eg(self.S_DQ_AVGPRICE, pos, 6), self.eg(self.MASK, pos, 6), 6)  # [ewma, stock]
        part1 = (self.en(self.S_DQ_AVGPRICE, pos) - part0) / part0  # [ewma, stock]
        part1_mask = self.en(self.MASK, pos) & part0_mask # [ewma, stock]
        alpha = part1[3:] - part1[:-3]
        alpha_mask = part1_mask[3:] & part1_mask[:-3]
        alpha = self.ewma(alpha, alpha_mask, alpha=1.0/12)[-1]  # [stock]
        mask = self.p(self.MASK, pos)
        alpha.masked_fill_(~mask, float('nan'))
        return alpha, mask

    def better_019(self, pos):
        log_close = t.log(self.en(self.S_DQ_VOLUME, pos) + 1)
        log_close_mask = self.en(self.MASK, pos)
        data = self.ewma(log_close, log_close_mask, alpha=2/13)
        data_mask = self.process_nan_infinite_and_mask(data, t.ones_like(data).bool())
        data = self.ewma(data, data_mask, alpha=2/13)
        data_mask = self.process_nan_infinite_and_mask(data, t.ones_like(data).bool())
        data = self.ewma(data, data_mask, alpha=2/13)
        data_mask = self.process_nan_infinite_and_mask(data, t.ones_like(data).bool())
        alpha = data[-1] / data[-2] - 1
        mask = data_mask[-1] & data_mask[-2] & self.p(self.MASK, pos)
        alpha.masked_fill_(~mask, float('nan'))
        return alpha, mask

    def better_010(self, pos):
        df1, df1_mask = self.bn(self.S_DQ_AVGPRICE, pos, 11).clone(), self.bn(self.MASK, pos, 11) # [11, stock]
        df1.masked_fill_(~df1_mask, float('nan'))
        df1 = df1[1:] < df1[:-1] # [10, stock]
        df1_mask = t.ones_like(df1).bool()  # [10, stock]
        sumif = t.abs(self.bn(self.S_DQ_AVGPRICE, pos, 10) / self.bn(self.S_DQ_AVGPRICE, pos-1, 10) - 1) / t.log(self.bn(self.S_DQ_AMOUNT, pos, 10) + 1)  # [10, stock]
        sumif_mask = self.bn(self.MASK, pos, 10) & self.bn(self.MASK, pos-1, 10)  # [10, stock]
        sumif.masked_fill_(~df1, 0)  # [10, stock]
        sumif_mask = self.process_nan_infinite_and_mask(sumif, t.ones_like(sumif).bool())
        sumif, sumif_mask = self.sum(sumif, sumif_mask, dim=0)  # [stock]
        count, count_mask = self.sum(df1, df1_mask, dim=0) # [stock]
        alpha = (sumif / count) # [stock]
        mask = sumif_mask & count_mask & self.p(self.MASK, pos) # [stock]
        alpha.masked_fill_(~mask, float('nan'))
        return alpha, mask

    def better_015(self, pos):
        part0, part0_mask = self.rolling_mean(self.eg(self.S_DQ_AVGPRICE, pos, 10), self.eg(self.MASK, pos, 10), 10)  # [ewma, stock]
        part1 = (self.en(self.S_DQ_AVGPRICE, pos) - part0) / part0  # [ewma, stock]
        part1_mask = self.en(self.MASK, pos) & part0_mask # [ewma, stock]
        alpha = part1[5:] - part1[:-5]
        alpha_mask = part1_mask[5:] & part1_mask[:-5]
        alpha = self.ewma(alpha, alpha_mask, alpha=1.0/20)[-1]  # [stock]
        mask = self.p(self.MASK, pos)
        alpha.masked_fill_(~mask, float('nan'))
        return alpha, mask

    def better_009(self, pos):
        df1, df1_mask = self.bn(self.S_DQ_AVGPRICE, pos, 21).clone(), self.bn(self.MASK, pos, 21) # [21, stock]
        df1.masked_fill_(~df1_mask, float('nan'))
        df1 = df1[1:] < df1[:-1] # [20, stock]
        df1_mask = t.ones_like(df1).bool()  # [20, stock]
        sumif = t.abs(self.bn(self.S_DQ_AVGPRICE, pos, 20) / self.bn(self.S_DQ_AVGPRICE, pos-1, 20) - 1) / t.log(self.bn(self.S_DQ_AMOUNT, pos, 20) + 1)  # [20, stock]
        sumif_mask = self.bn(self.MASK, pos, 20) & self.bn(self.MASK, pos-1, 20)  # [20, stock]
        sumif.masked_fill_(~df1, 0)  # [20, stock]
        sumif_mask = self.process_nan_infinite_and_mask(sumif, t.ones_like(sumif).bool())
        sumif, sumif_mask = self.sum(sumif, sumif_mask, dim=0)  # [stock]
        count, count_mask = self.sum(df1, df1_mask, dim=0) # [stock]
        alpha = (sumif / count) # [stock]
        mask = sumif_mask & count_mask & self.p(self.MASK, pos) # [stock]
        alpha.masked_fill_(~mask, float('nan'))
        return alpha, mask

    def better_022(self, pos):
        alpha = (self.p(self.S_DQ_AVGPRICE, pos) / self.p(self.S_FWDS_ADJCLOSE, pos-1) - 1) * t.log2(self.p(self.S_DQ_AMOUNT, pos) + 1)
        mask = self.p(self.MASK, pos) & self.p(self.MASK, pos-1)
        alpha.masked_fill_(~mask, float('nan'))
        return alpha, mask

    def better_014(self, pos):
        alpha, alpha_mask = self.mean((self.bn(self.S_DQ_AVGPRICE, pos, 4) / self.bn(self.S_FWDS_ADJCLOSE, pos, 4) - 1) * t.log2(self.bn(self.S_DQ_AMOUNT, pos, 4) + 1), self.bn(self.MASK, pos, 4), dim=0)  # [stock]
        alpha.masked_fill_(~alpha_mask, float('nan'))
        return alpha, alpha_mask

    def better_021(self, pos):
        alpha = self.p(self.S_DQ_AVGPRICE, pos) / self.p(self.S_FWDS_ADJCLOSE, pos-1)
        mask = self.p(self.MASK, pos) & self.p(self.MASK, pos-1)
        alpha.masked_fill_(~mask, float('nan'))
        return alpha, mask

    def better_006(self, pos):
        alpha, alpha_mask = self.mean((2 * self.bn(self.S_DQ_AVGPRICE, pos, 5) - self.bn(self.S_FWDS_ADJLOW, pos, 5) - self.bn(self.S_FWDS_ADJHIGH, pos, 5)) / (self.bn(self.S_FWDS_ADJHIGH, pos, 5) - self.bn(self.S_FWDS_ADJLOW, pos, 5)), self.bn(self.MASK, pos, 5), dim=0)  # [stock]
        mask = self.process_nan_infinite_and_mask(alpha, alpha_mask)
        alpha.masked_fill_(~mask, float('nan'))
        return alpha, mask

    def better_013(self, pos):
        alpha = (self.p(self.S_DQ_AVGPRICE, pos) / self.p(self.S_FWDS_ADJCLOSE, pos) - 1) * t.log2(self.p(self.S_DQ_AMOUNT, pos) + 1)
        mask = self.p(self.MASK, pos)
        alpha.masked_fill_(~mask, float('nan'))
        return alpha, mask

    def better_002(self, pos):
        data, data_mask = self.mean(self.bn(self.S_DQ_VOLUME, pos, 5), self.bn(self.MASK, pos, 5), dim=0)
        alpha = (self.p(self.S_FWDS_ADJOPEN, pos) / self.p(self.S_FWDS_ADJCLOSE, pos-1) - 1) * self.p(self.S_DQ_VOLUME, pos) / data
        mask = self.p(self.MASK, pos) & self.p(self.MASK, pos-1) & data_mask
        alpha.masked_fill_(~mask, float('nan'))
        return alpha, mask

    def better_008(self, pos):
        alpha = (self.p(self.S_FWDS_ADJHIGH, pos) - self.p(self.S_FWDS_ADJLOW, pos)) / self.p(self.S_FWDS_ADJCLOSE, pos) / t.log2(self.p(self.S_DQ_VOLUME, pos) + 1)
        mask = self.p(self.MASK, pos)
        alpha.masked_fill_(~mask, float('nan'))
        return alpha, mask

    def better_007(self, pos):
        alpha = (self.p(self.S_FWDS_ADJHIGH, pos) - self.p(self.S_FWDS_ADJLOW, pos)) / self.p(self.S_FWDS_ADJCLOSE, pos) * (self.p(self.S_DQ_VOLUME, pos) - self.p(self.S_DQ_VOLUME, pos-1))
        mask = self.p(self.MASK, pos) & self.p(self.MASK, pos-1)
        alpha.masked_fill_(~mask, float('nan'))
        return alpha, mask

    def better_005(self, pos):
        part1 = (2 * self.p(self.S_DQ_AVGPRICE, pos) - self.p(self.S_FWDS_ADJLOW, pos) - self.p(self.S_FWDS_ADJHIGH, pos)) / (self.p(self.S_FWDS_ADJHIGH, pos) - self.p(self.S_FWDS_ADJLOW, pos))
        part2 = (2 * self.p(self.S_DQ_AVGPRICE, pos-5) - self.p(self.S_FWDS_ADJLOW, pos-5) - self.p(self.S_FWDS_ADJHIGH, pos-5)) / (self.p(self.S_FWDS_ADJHIGH, pos-5) - self.p(self.S_FWDS_ADJLOW, pos-5))
        alpha = -(part1 - part2)
        mask = self.p(self.MASK, pos) & self.p(self.MASK, pos-5)
        alpha.masked_fill_(~mask, float('nan'))
        return alpha, mask

    def better_004(self, pos):
        part1 = (2 * self.p(self.S_DQ_AVGPRICE, pos) - self.p(self.S_FWDS_ADJLOW, pos) - self.p(self.S_FWDS_ADJHIGH, pos)) / (self.p(self.S_FWDS_ADJHIGH, pos) - self.p(self.S_FWDS_ADJLOW, pos))
        part2 = (2 * self.p(self.S_DQ_AVGPRICE, pos-1) - self.p(self.S_FWDS_ADJLOW, pos-1) - self.p(self.S_FWDS_ADJHIGH, pos-1)) / (self.p(self.S_FWDS_ADJHIGH, pos-1) - self.p(self.S_FWDS_ADJLOW, pos-1))
        alpha = -(part1 - part2)
        mask = self.p(self.MASK, pos) & self.p(self.MASK, pos-1)
        alpha.masked_fill_(~mask, float('nan'))
        return alpha, mask

    def better_003(self, pos):
        alpha = (self.p(self.S_FWDS_ADJOPEN, pos) / self.p(self.S_FWDS_ADJCLOSE, pos-1) - 1) * (self.p(self.S_DQ_AVGPRICE, pos) - self.p(self.S_FWDS_ADJCLOSE, pos)) * (self.p(self.S_FWDS_ADJHIGH, pos) - self.p(self.S_FWDS_ADJLOW, pos)) / self.p(self.S_FWDS_ADJCLOSE, pos)
        mask = self.p(self.MASK, pos) & self.p(self.MASK, pos-1)
        alpha.masked_fill_(~mask, float('nan'))
        return alpha, mask

    def better_001(self, pos):
        alpha = (self.p(self.S_FWDS_ADJOPEN, pos) / self.p(self.S_FWDS_ADJCLOSE, pos-1) - 1) * t.log2(self.p(self.S_DQ_VOLUME, pos) + 1)
        mask = self.p(self.MASK, pos) & self.p(self.MASK, pos-1)
        alpha.masked_fill_(~mask, float('nan'))
        return alpha, mask

    def add_027(self, pos):
        part0 = (self.bn(self.S_FWDS_ADJHIGH, pos, 20) - self.bn(self.S_FWDS_ADJLOW, pos, 20)) / self.bn(self.S_FWDS_ADJCLOSE, pos, 20)
        part0_mask = self.bn(self.MASK, pos, 20)
        part1 = self.bn(self.S_DQ_VOLUME, pos, 20)
        part1_mask = self.bn(self.MASK, pos, 20)
        alpha, alpha_mask = self.corr(part0, part1, (part0_mask & part1_mask), dim=0)
        alpha.masked_fill_(~alpha_mask, float('nan'))
        return alpha, alpha_mask

    def old_045(self, pos):
        data1, data1_mask = self.rolling_sum((self.rg(self.S_FWDS_ADJHIGH, pos, 5, 9) + self.rg(self.S_FWDS_ADJLOW, pos, 5, 9)) / 2, self.rg(self.MASK, pos, 5, 9), 5)  # [9, stock]
        data2, data2_mask = self.rolling_mean(self.rg(self.S_DQ_VOLUME, pos, 10, 13), self.rg(self.MASK, pos, 10, 13), 10)  # [13, stock]
        data2, data2_mask = self.rolling_sum(data2, data2_mask, 5)  # [9, stock]
        rank1, rank1_mask = self.corr(data1, data2, (data1_mask & data2_mask), dim=0)  # [stock]
        rank1_mask = self.process_nan_infinite_and_mask(rank1, rank1_mask)
        rank1, rank1_mask = self.rank(rank1, rank1_mask, dim=0)  # [stock]
        rank2, rank2_mask = self.corr(self.bn(self.S_FWDS_ADJLOW, pos, 6), self.bn(self.S_DQ_VOLUME, pos, 6), self.bn(self.MASK, pos, 6), dim=0)  # [stock]
        rank2_mask = self.process_nan_infinite_and_mask(rank2, rank2_mask)
        rank2, rank2_mask = self.rank(rank2, rank2_mask, dim=0)  # [stock]
        #rank1.masked_fill_(rank1_mask, float('nan'))
        #rank2.masked_fill_(rank2_mask, float('nan'))
        alpha = (rank1 < rank2).float()
        mask = self.p(self.MASK, pos)
        alpha.masked_fill_(~mask, 1)
        return alpha, mask

    def add_026(self, pos):
        part0 = (self.bn(self.S_FWDS_ADJHIGH, pos, 10) - self.bn(self.S_FWDS_ADJLOW, pos, 10)) / self.bn(self.S_FWDS_ADJCLOSE, pos, 10)
        part0_mask = self.bn(self.MASK, pos, 10)
        part1 = self.bn(self.S_DQ_VOLUME, pos, 10)
        part1_mask = self.bn(self.MASK, pos, 10)
        alpha, alpha_mask = self.corr(part0, part1, (part0_mask & part1_mask), dim=0)
        alpha.masked_fill_(~alpha_mask, float('nan'))
        return alpha, alpha_mask

    def add_025(self, pos):
        part0 = (self.bn(self.S_FWDS_ADJHIGH, pos, 5) - self.bn(self.S_FWDS_ADJLOW, pos, 5)) / self.bn(self.S_FWDS_ADJCLOSE, pos, 5)
        part0_mask = self.bn(self.MASK, pos, 5)
        part1 = self.bn(self.S_DQ_VOLUME, pos, 5)
        part1_mask = self.bn(self.MASK, pos, 5)
        alpha, alpha_mask = self.corr(part0, part1, (part0_mask & part1_mask), dim=0)
        alpha.masked_fill_(~alpha_mask, float('nan'))
        return alpha, alpha_mask

    def old_041(self, pos):
        data1, data1_mask = self.rolling_mean(self.bn(self.S_FWDS_ADJCLOSE, pos, 25), self.bn(self.MASK, pos, 25), 20) # [5, stock]
        part1, part1_mask = self.rank(data1[-6, :], data1_mask[-6, :], dim=0)
        part2, part2_mask = self.corr(self.bn(self.S_FWDS_ADJCLOSE, pos, 5), self.bn(self.S_DQ_VOLUME, pos, 5), self.bn(self.MASK, pos, 5), 0)
        data31, data31_mask = self.rolling_sum(self.rg(self.S_FWDS_ADJCLOSE, pos, 5, 5), self.rg(self.MASK, pos, 5, 5), 5)
        data32, data32_mask = self.rolling_sum(self.rg(self.S_FWDS_ADJCLOSE, pos, 20, 5), self.rg(self.MASK, pos, 20, 5), 20)
        part3, part3_mask = self.corr(data31, data32, data31_mask & data32_mask, 0)
        alpha = (-part1*part2*part3)
        mask = part1_mask & part2_mask & part3_mask
        alpha.masked_fill_(~mask, float('nan'))
        return alpha, mask

    def old_034(self, pos):
        part1, part1_mask = self.rolling_mean(self.rg(self.S_DQ_VOLUME, pos, 10, 21), self.rg(self.MASK, pos, 10, 21), 10)  # [21, stock]
        part1, part1_mask = self.rolling_sum(part1, part1_mask, 15)  # [7, stock]
        part1, part1_mask = self.corr(self.bn(self.S_FWDS_ADJCLOSE, pos, 7), part1, (part1_mask & self.bn(self.MASK, pos, 7)), dim=0) # [stock]
        rank1, rank1_mask = self.rank(part1, part1_mask, dim=0)  # [stock]
        rank2, rank2_mask = (self.bn(self.S_FWDS_ADJHIGH, pos, 11) * 0.1 + self.bn(self.S_DQ_AVGPRICE, pos, 11) * 0.9), self.bn(self.MASK, pos, 11)  # [11, stock]
        rank2, rank2_mask = self.rank(rank2, rank2_mask, dim=1)  # [11, stock]
        rank3, rank3_mask = self.bn(self.S_DQ_VOLUME, pos, 11), self.bn(self.MASK, pos, 11)  # [11, stock]
        rank3, rank3_mask = self.rank(rank3, rank3_mask, dim=1)  # [11, stock]
        rank4, rank4_mask = self.corr(rank2, rank3, (rank2_mask & rank3_mask), dim=0)  # [stock]
        rank4_mask = self.process_nan_infinite_and_mask(rank4, rank4_mask)
        rank4, rank4_mask = self.rank(rank4, rank4_mask, dim=0)  # [stock]
        #rank1.masked_fill_(rank1_mask, float('nan'))
        #rank4.masked_fill_(rank4_mask, float('nan'))
        alpha = (~(rank1 < rank4)).float()
        mask = self.p(self.MASK, pos)
        alpha.masked_fill_(~mask, 1)
        return alpha, mask

    def add_001(self, pos):
        data1 = t.log(self.bn(self.S_DQ_VOLUME, pos, 10) + 1) - t.log(self.bn(self.S_DQ_VOLUME, pos-1, 10) + 1)
        data1_mask = self.bn(self.MASK, pos, 10) & self.bn(self.MASK, pos-1, 10)
        data2 = ((self.bn(self.S_FWDS_ADJCLOSE, pos, 10) - self.bn(self.S_FWDS_ADJOPEN, pos, 10)) / self.bn(self.S_FWDS_ADJOPEN, pos, 10))
        data2_mask = self.bn(self.MASK, pos, 10)
        alpha, alpha_mask = self.corr(data1, data2, (data1_mask & data2_mask), dim=0)
        alpha.masked_fill_(~alpha_mask, float('nan'))
        return alpha, alpha_mask

    def cs_rank_close(self, pos):
        inc_close = t.clamp(self.p(self.S_FWDS_ADJCLOSE, pos) / self.p(self.LAST_CLOSE, pos) - 1, min=-0.1, max=0.1)  # [stock]
        alpha, alpha_mask = self.rank(inc_close, self.p(self.MASK, pos), dim=0, descending=True)  # [stock]
        alpha.masked_fill_(~alpha_mask, float('nan'))
        return alpha, alpha_mask

    def cs_rank_open(self, pos):
        inc_open = t.clamp(self.p(self.S_FWDS_ADJOPEN, pos) / self.p(self.LAST_CLOSE, pos) - 1, min=-0.1, max=0.1)
        alpha, alpha_mask = self.rank(inc_open, self.p(self.MASK, pos), dim=0, descending=True)  # [stock]
        alpha.masked_fill_(~alpha_mask, float('nan'))
        return alpha, alpha_mask

    def cs_rank_high(self, pos):
        inc_high = self.p(self.S_FWDS_ADJHIGH, pos) / self.p(self.LAST_CLOSE, pos) - 1
        alpha, alpha_mask = self.rank(inc_high, self.p(self.MASK, pos), dim=0, descending=True)  # [stock]
        alpha.masked_fill_(~alpha_mask, float('nan'))
        return alpha, alpha_mask

    def cs_rank_avg(self, pos):
        inc_avg = self.p(self.S_DQ_AVGPRICE, pos) / self.p(self.LAST_CLOSE, pos) - 1
        alpha, alpha_mask = self.rank(inc_avg, self.p(self.MASK, pos), dim=0, descending=True)  # [stock]
        alpha.masked_fill_(~alpha_mask, float('nan'))
        return alpha, alpha_mask

    def cs_rank_low(self, pos):
        inc_low = self.p(self.S_FWDS_ADJLOW, pos) / self.p(self.LAST_CLOSE, pos) - 1
        alpha, alpha_mask = self.rank(inc_low, self.p(self.MASK, pos), dim=0, descending=True)  # [stock]
        alpha.masked_fill_(~alpha_mask, float('nan'))
        return alpha, alpha_mask

    def cs_rank_amount(self, pos):
        alpha, alpha_mask = self.rank(self.p(self.S_DQ_AMOUNT, pos), self.p(self.MASK, pos), dim=0, descending=True)  # [stock]
        alpha.masked_fill_(~alpha_mask, float('nan'))
        return alpha, alpha_mask

    def CR_(self, pos):
        inc_close = t.clamp(self.p(self.S_FWDS_ADJCLOSE, pos) / self.p(self.LAST_CLOSE, pos) - 1, min=-0.1, max=0.1)  # [stock]
        mask = self.p(self.MASK, pos)
        inc_close.masked_fill_(~mask, float('-inf'))
        CRi, CR = t.unique((inc_close * 100).long(), return_counts=True, sorted=True)
        CRi = CRi[1:]
        CR = CR[1:]
        CR = CR / CR.sum(dim=0)
        CRs = t.zeros((21,), dtype=CR.dtype, device=CR.device)
        CRs[CRi + 10] = CR
        CRs = CRs.unsqueeze(0).repeat(inc_close.shape[0], 1)  # [stock, 21]
        mask = mask.unsqueeze(-1)  # [stock, 1]
        CRs.masked_fill_(~mask, float('nan'))
        return CRs, mask

    def OR_(self, pos):
        inc_close = t.clamp(self.p(self.S_FWDS_ADJOPEN, pos) / self.p(self.LAST_CLOSE, pos) - 1, min=-0.1, max=0.1)  # [stock]
        mask = self.p(self.MASK, pos)
        inc_close.masked_fill_(~mask, float('-inf'))
        CRi, CR = t.unique((inc_close * 100).long(), return_counts=True, sorted=True)
        CRi = CRi[1:]
        CR = CR[1:]
        CR = CR / CR.sum(dim=0)
        CRs = t.zeros((21,), dtype=CR.dtype, device=CR.device)
        CRs[CRi + 10] = CR
        CRs = CRs.unsqueeze(0).repeat(inc_close.shape[0], 1)  # [stock, 21]
        mask = mask.unsqueeze(-1)  # [stock, 1]
        CRs.masked_fill_(~mask, float('nan'))
        return CRs, mask

    def get_features(self, date):
        pos = self.all_dates[date]
        print('date:', date, pos, flush=True, file=sys.stderr)
        t1 = time.time()
        #for pos in range(200, 1100):
        #for pos in range(pos, pos+1):
        fea_lists = []
        stocks = []
        dates = []
        for date, pos in self.all_dates.items():
            #if pos < 200:
            #    continue
            if date < '20210101' or date > '20210831':
            #if date < '20110101' or date > '20201231':
                continue
            #self.rolling_mean(pos, 60, 100, 4)
            #self.rolling_tsrank(self.S_FWDS_ADJCLOSE, pos, 10, 100)

            np.set_printoptions(threshold=np.inf, suppress=True, linewidth=200)
            fea_list = []
            for func in self.testset:
                result, result_mask = getattr(self, func)(pos)
                if func == 'OR_' or func == 'CR_':
                    fea_list.append(result.t())
                else:
                    fea_list.append(result.unsqueeze(0))

                
                #self.cmp(func, date, result, result_mask)
            fea_tensor = t.cat(fea_list).t()
            fea_lists.append(fea_tensor.to('cpu'))
            for sid in self.all_stocks_test:
                stocks.append(sid)
                dates.append(date)
        res_tensor = t.cat(fea_lists).to('cpu')
        print(res_tensor.shape, len(stocks), len(dates), flush=True, sep='\t')
        result_dict = {}
        result_dict['x'] = res_tensor[:, 1:]
        result_dict['y'] = res_tensor[:, 0]
        result_dict['stocks'] = stocks
        result_dict['date'] = dates

        pd.DataFrame(result_dict['x'].numpy()).to_csv('tmpx.csv')
        pd.DataFrame(result_dict['y'].numpy()).to_csv('tmpy.csv')

        fp = open("/da3/public/tensor_data_test.pickle", 'wb')
        pickle.dump(result_dict, fp,  protocol=4)


        t2 = time.time()
        print('t:', t2 - t1, flush=True)

    def cmp(self, prefix, date, x, x_mask):
        #return
        test_pos = self.all_dates_test[date]
        if prefix == 'CR_' or prefix == 'OR_':
            for idx in range(-10, 11):
                nidx = idx + 10
                nprefix = prefix + str(idx)
                bm = self.test_results[nprefix][test_pos]
                #print('test_sample:', nprefix, date, test_pos, '[', ', '.join(map(str, x[:10, nidx].cpu().numpy())), '], [', ', '.join(map(str, bm[:10].cpu().numpy())), ']', flush=True, file=sys.stderr)
        else:
            bm = self.test_results[prefix][test_pos]
            #print('test_sample:', prefix, date, test_pos, '[', ', '.join(map(str, x[:10].cpu().numpy())), '], [', ', '.join(map(str, bm[:10].cpu().numpy())), ']', flush=True, file=sys.stderr)
            #print('cmp0:', x.shape, bm.shape, x, bm, flush=True, file=sys.stderr)
            print(x)
            mse = (x - bm) ** 2
            mse.masked_fill_(((~x_mask) & t.isnan(bm)), 0.0)
            mse.masked_fill_((t.isnan(x) & t.isnan(bm)), 0.0)
            mse.masked_fill_((t.isinf(x) & t.isinf(bm)), 0.0)
            top, idx = mse.topk(5) # [stock]
            #print('cmp01', top.shape, idx.shape, top, idx, flush=True, file=sys.stderr)
            topx = t.gather(x, -1, idx) # [stock]
            topbm = t.gather(bm, -1, idx) # [stock]
            info = t.stack([top, idx, topx, topbm], dim=-1).cpu().numpy()
            for i in range(len(info)):
                if info[i, 0] < 1e-10:
                    continue
                #print('\tcmp:', prefix, date, i, self.all_stocks_test[idx[i].cpu().numpy()], '\t'.join(map(str, info[i])), flush=True, sep='\t')

        
    def load_test(self, df, device):
        self.all_stocks_test = list(df['S_INFO_WINDCODE'].unique())
        #with open('/da2/anjingwen-s/bm/daily_data/Ashares2train_tushare_test.pickle.202101', 'rb') as f:
        with open('/da2/anjingwen-s/bm/daily_data/Ashares2train_tushare_test.pickle.202101.2', 'rb') as f:
            df2 = pickle.load(f)
        self.all_dates_test = list(df2['date'].unique())
        self.all_dates_test = { self.all_dates_test[i]:i for i in range(len(self.all_dates_test)) }
        self.testset = ['target_01', 'best_014', 'extra_005', 'best_001', 'best_002', 'best_003', 'stock018', 'extra_006', 'old_035', 
                    'best_006', 'original_005', 'stock022', 'original_006', 'change_004', 'change_003', 'change_002', 
                    'original_001', 'change_005', 'original_004', 'change_001', 'add_002', 'original_002',
                    'original_003', 'add_021', 'add_020', 'add_030', 'add_029', 'add_028', 'stock016', 'add_011',
                    'add_019', 'add_018', 'add_013', 'add_010', 'add_017', 'add_024', 'add_016', 'add_023', 'add_006',
                    'stock002', 'stock001', 'add_008', 'add_015', 'add_014', 'add_007', 'add_012', 'add_003', 'add_009',
                    'extra_008', 'extra_009', 'add_005', 'add_004', 'stock009', 'extra_002', 'extra_014', 'extra_013', 
                    'old_036', 'extra_012', 'stock011', 'extra_007', 'extra_011', 'extra_010', 'extra_004', 'stock015', 
                    'stock007', 'old_037', 'stock017', 'extra_003', 'stock021', 'extra_001', 'stock008', 'stock019', 
                    'stock020', 'stock010', 'old_032', 'stock012', 'stock014', 'stock004', 'best_021', 'stock006', 
                    'stock013', 'stock003', 'stock005', 'best_020', 'best_019', 'best_018', 'best_012', 'best_017', 
                    'best_015', 'best_016', 'best_008', 'best_013', 'best_009', 'best_011', 'old_042', 'better_012', 
                    'old_040', 'best_010', 'better_011', 'old_030', 'best_007', 'old_044', 'best_005', 'best_004', 
                    'old_043', 'better_025', 'old_039', 'old_033', 'better_028', 'old_031', 'better_027', 'old_038', 
                    'old_029', 'old_028', 'better_020', 'old_027', 'better_024', 'better_026', 'better_023', 'better_018', 
                    'better_017', 'better_016', 'better_019', 'better_010', 'better_015', 'better_009', 'better_022', 
                    'better_014', 'better_021', 'better_006', 'better_013', 'better_002', 'better_008', 'better_007', 
                    'better_005', 'better_004', 'better_003', 'better_001', 'add_027', 'old_045', 'add_026', 'add_025', 
                    'old_041', 'add_002', 'old_034', 'add_001', 
                    'cs_rank_amount', 'cs_rank_close', 'cs_rank_open', 'CR_', 'OR_',
        ]


        self.test_results = {}
        for fea in self.testset:
            if fea == 'old_045' or fea == 'old_034':
                self.test_results[fea] = t.from_numpy(df2.pivot(index='date', columns=['stock'], values=fea).reindex(columns=self.all_stocks_test).to_numpy().astype(np.bool)).to(device).float()
            elif fea == 'CR_' or fea == 'OR_':
                for idx in range(-10, 11):
                    nfea = fea + str(idx)
                    self.test_results[nfea] = t.tensor(df2.pivot(index='date', columns=['stock'], values=nfea).reindex(columns=self.all_stocks_test).to_numpy()).to(device)
            else:
                self.test_results[fea] = t.tensor(df2.pivot(index='date', columns=['stock'], values=fea).reindex(columns=self.all_stocks_test).to_numpy()).to(device)

    
def extractor(idx, id_queue, out_queue, fktr):
    while True:
        data = id_queue.get()
        #print('date get:', date, flush=True)
        if data is None:
            out_queue.put(None)
            break
        date, = data
        fktr.get_features(date)

if __name__ == "__main__":
    multiprocessing.set_start_method('spawn')
    cp.cuda.Device(0).use()
                        

    #data_path = '/da2/anjingwen-s/bm/daily_data/tushare_neutral2.pickle'
    #ochl_path = None
    data_path = None
    ochl_path = '/da3/public/tushare_dym.data'

    fktr = Factors(pickle_file=data_path, ochl_file=ochl_path, device='cuda:0')
    #fktr = Factors(pickle_file=data_path, ochl_file=ochl_path, device='cpu')

    cpus = 1
    id_queue = Queue(1000)
    out_queue = Queue(10000)
    workers = []
    for i in range(cpus):
        worker = Process(target=extractor, args=(i, id_queue, out_queue, fktr))
        workers.append(worker)
        worker.start()

    cache = {}
    for date in fktr.all_dates:
        #if date <= '20211001':
        if date != '20210104':
            continue
        id_queue.put((date,))

    for i in range(cpus):
        id_queue.put(None)
    for w in workers:
        w.join()
   
