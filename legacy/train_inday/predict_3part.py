from model import *
from dataset import *
import torch.nn.functional as F
from torch.utils.data.dataloader import default_collate
import torch
import numpy as np
import os
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error, explained_variance_score, r2_score

import sys
# # deal with url request exception
# def collate_fn_filter(batch):
#     batch = list(filter(lambda x: len(x) == 9, batch))
#     if len(batch) == 0:
#         return torch.Tensor()
#     return default_collate(batch)

device = torch.device('cpu')


global_lr = 0.01
model = StockPred_v10().to(device)
#model = Feature_200_KLDivLoss().to(device)
model.load_state_dict(torch.load(sys.argv[1]))
params = list(model.named_parameters())
#model.load_state_dict(torch.load('model_force.pt.101')) #regre+clsif best
#model.load_state_dict(torch.load('model_force.pt.6')) # regre best
#model.load_state_dict(torch.load('model_force.pt.26')) #good
#model.load_state_dict(torch.load('model_force.pt.9')) # regre best 2 (merge 19_20)
#model.load_state_dict(torch.load('/da1/public/duyimin/trade_predict/model_force.pt.90'))
#optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
# scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=300, gamma = 0.93)
n_epochs = 30000

loss_stat = []
#dataset = StockData('f_train', 13177512)


# pre load label list:
# label_list = []
# with open('/da2/search/wanghexiang/stock_data/test_data', 'r') as f:
#     for line in f:
#         if len(label_list) < 1280000:
#             ll = line.strip('\n\r').split('\t')[-1].split('|')[-1]
#             label_list.append(float(ll))
valid_batch = 25600
batch_size = 128
counter = 1000

#baseline day data
#_valid_data = StockData_torch_fea_test('./tensor_data_test.pickle', device)


#_valid_data = StockData_torch_fea_test('./tensor_data_test_torch_fea.pickle', device)
#_valid_data = StockData_mini_batch_tensor_gbm_pre('/da1/public/duyimin/data/tensor_data_concat_test.pickle', device)
#_valid_data = StockData_mini_batch_tensor_gbm_pre('/da1/public/duyimin/data/tensor_data_concat3_test.pickle', device)
_valid_data = StockData_mini_batch_tensor_gbm_pre('/da1/public/duyimin/data/tensor_data_concat3_k15_test.pickle', device)
#_valid_data = StockData_mini_batch_tensor_gbm_pre('/da1/public/duyimin/data/tensor_data_concat3_k15_newest.pickle', device)
#_valid_data = StockData_mini_batch_day_inday_tensor('./tensor_data_newest_torch_fea.pickle', device)

#_valid_data = StockData_mini_batch_day_inday_tensor('./tensor_data_merge_gbm_19_20.pickle', device)
#_valid_data = StockData_day_inday_tensor('./tensor_data_new_test.pickle', device)


#print('starting test model !')
pre_list = []
label_list = []
#_valid_data = StockData('f_test', 3254549)
total_valid = int(np.ceil(_valid_data.f_len / valid_batch))
#print(_valid_data.f_len)
fo = open("./tushare_day", "w")
t_test_data, l_test_data, date, stocks, today_close, halfday_close = _valid_data.x, _valid_data.y, _valid_data.date, _valid_data.stocks, _valid_data.today_close, _valid_data.halfday_close
label_list = (l_test_data.numpy().reshape(-1)).tolist()
date_list = date
stocks_list = stocks
model.eval()
with torch.no_grad():
    #pre = model.predict(t_test_data.to(device))
    ##w = torch.cat((torch.ones(30) * -1, torch.zeros(1), torch.ones(30)), 0)
    #out_put = pre.detach().cpu().numpy()
    #pre_list = np.reshape(out_put,-1).tolist()

    pre = model.predict(t_test_data.to(device))
    #pre = torch.exp(pre)
    #w = torch.cat((torch.ones(30) * -1, torch.zeros(1), torch.ones(30)), 0)
    #w = torch.Tensor(range(-20,21)).to(device)
    #pre = torch.matmul(pre, w)
    out_put = pre.detach().cpu().numpy()
    pre_list = np.reshape(out_put,-1).tolist()


    for i in range(len(label_list)):
        fo.write(str(stocks[i]) + "\t" + str(date[i]) + "\t" + str(label_list[i]) + "\t" + str(pre_list[i]) + "\t" + str(today_close[i]) + "\t" + str(halfday_close[i])  + '\n' )
        #fo.write(str(label_list[i]) + "\t" + str(date[i]) + "\t" + str(stocks[i]) + '\n' )
fo.close()
