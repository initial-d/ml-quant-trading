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
        self.df = df.loc[ (df['date'] >= begindate) & (df['date'] <= enddate) ]
        self.df.dropna(axis=0, subset=['self.target_01'], inplace=True)
        self.df.fillna(0, inplace=True)
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
        batch_data = np.zeros((batch_size, 144), dtype=np.float32)
        batch_label = np.zeros((batch_size, 1), dtype=np.float32)
        #for i in range(n):
        i = self.k * self.Batch_size
        self.k = (self.k + 1) % self.n
        batch_data = self.df.iloc[i:i+batch_size, :]
        featurearray = batch_data[self.col].to_numpy()
        
        batch_data = featurearray[:, 1:]
        batch_data[batch_data > 1e+1] = 0.0
        batch_data[batch_data < -1e+1] = 0.0
        batch_label[:, 0] = np.log(featurearray[:, 0] + 1.0)
        #print(batch_data)
        #print(batch_label)

        t_batch_data = torch.from_numpy(batch_data)
        #print(self.label.shape)
        
        l_batch_data = torch.from_numpy(batch_label)
        return t_batch_data, l_batch_data

    def get_predict_batch_data(self, batch_size=-1):
        #indices = torch.randperm(len(self.x))[:batch_size] 
        #return self.x[indices], self.y[indices]
        batch_data = np.zeros((batch_size, 144), dtype=np.float32)
        batch_label = np.zeros((batch_size, 1), dtype=np.float32)
        #for i in range(n):
        i = self.k * self.Batch_size
        self.k = (self.k + 1) % self.n
        batch_data = self.df.iloc[i:i+batch_size, :]
        featurearray = batch_data[self.col].to_numpy()
        
        date = batch_data['date'].to_numpy()
        stocks = batch_data['stocks'].to_numpy()

        batch_data = featurearray[:, 1:]
        batch_data[batch_data > 1e+1] = 0.0
        batch_data[batch_data < -1e+1] = 0.0
        batch_label[:, 0] = np.log(featurearray[:, 0] + 1.0)

        t_batch_data = torch.from_numpy(batch_data)
        #print(self.label.shape)
        
        l_batch_data = torch.from_numpy(batch_label)
        return t_batch_data, l_batch_data, date, stocks



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
    # dataset =  Np_Idx_DataSetFile_recall('recall_train_data_map_id.txt', file_len=100000)
    # t2 = 0.0

    # for i in range(5000):
    #     start = time.time()
    #     data = dataset.get_batch_data(128)
    #     end = time.time()
    #     t2 += end -start
    #     print(end - start)
    pass
