from model import *
from dataset import *
import torch.nn.functional as F
from torch.utils.data.dataloader import default_collate
import torch
import random
import numpy as np
import os
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error, explained_variance_score, r2_score

# # deal with url request exception
# def collate_fn_filter(batch):
#     batch = list(filter(lambda x: len(x) == 9, batch))
#     if len(batch) == 0:
#         return torch.Tensor()
#     return default_collate(batch)

model_saving_path = './lr0001/model.pt'
forced_saving_path = './lr0001/model_force.pt'
#device = torch.device('cuda:0')
device = torch.device('cpu')


global_lr = 0.01
#model = StockPred_v4().to(device)
model = StockPred_v9().to(device)
#model = Feature_KLDivLoss().to(device)
#model.load_state_dict(torch.load('/da1/public/duyimin/trade_predict/model_force.pt.90'))
#optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
# scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=300, gamma = 0.93)
optimizer = torch.optim.Adam(model.parameters(),lr=0.0001)
criterion1 = torch.nn.MSELoss()
criterion2 = torch.nn.BCEWithLogitsLoss()
criterion = torch.nn.KLDivLoss()
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

#day baseline data
#dataset = StockData_torch_fea_test('./tensor_data.pickle', device)

dataset = StockData_torch_fea_test('./tensor_data_torch_fea.pickle', device)
#_valid_data = StockData_mini_batch_day_inday_tensor('./tensor_data_minute_2021.pickle', device)
for epoch in np.arange(n_epochs):

    dataset.idx = 0

    epoch_loss = 0
    total_batches = int(np.ceil(len(dataset) / batch_size))
    print("start train epoch: ", epoch)
    print(" ")
    print("total batch num: ", int(np.ceil(total_batches / counter)))
    batch_num = 0 
    train_loss = 0.0
    model.train()
    for batch_num in range(total_batches):
        # start = time.time()
        t_data, l_data = dataset.get_batch_data(batch_size)

        #c_data = l_data.clone()
        #nc_data = c_data.numpy()
        #nc_data = np.where(nc_data > 0, 1.0, 0.0)
        #cl_data = torch.from_numpy(nc_data).to(torch.float32)

        optimizer.zero_grad()
        pre = model.forward(t_data.to(device))
        pre = pre.squeeze(-1)

        loss = criterion1(pre, l_data)
        #loss1 = criterion1(pre, l_data)
        #loss2 = criterion2(pre, cl_data)
        #loss = 0.9 * loss1 + 0.1 * loss2
        loss.backward()
 
        #torch.nn.utils.clip_grad_norm_(model.parameters(), 1)
        optimizer.step()

        train_loss += loss.cpu().detach().numpy().item()
        if batch_num % counter == 0 and batch_num != 0:
            avg_train_loss = train_loss / counter
            tt = time.localtime()
            print(time.strftime('%Y-%m-%d %H:%M:%S',tt), 'Epoch:', epoch, ' Batch num:', int(batch_num / counter), ' Average loss:', avg_train_loss, 'LR: ',  optimizer.param_groups[0]['lr'])
            epoch_loss += train_loss
            train_loss = 0

    if 1: 
        #print('starting test model !')
        #if 1:
        #    _valid_data.idx = 0
        #    pre_list = []
        #    label_list = []
        #    #_valid_data = StockData('f_test', 3254549)
        #    total_valid = int(np.ceil(_valid_data.f_len / valid_batch))
        #    print(_valid_data.f_len)
        #    fo = open("./lr0001/tushare_day." + str(epoch), "w")
        #    for j in range(total_valid):
        #        t_test_data, l_test_data, date, stocks, today_close, halfday_close = _valid_data.get_predict_batch_data(valid_batch)
        #        label_list = (l_test_data.numpy().reshape(-1)).tolist()
        #        date_list = (date.reshape(-1)).tolist()
        #        stocks_list = (stocks.reshape(-1)).tolist()
        #        model.eval()
        #        with torch.no_grad():
        #            #pre = model.valid_batch(t_test_data.to(device))
        #            pre = model.predict(t_test_data.to(device))
        #            out_put = pre.detach().cpu().numpy()
        #            pre_list = np.reshape(out_put,-1).tolist()
        #
        #            for i in range(len(label_list)):
        #                fo.write(str(stocks[i]) + "\t" + str(date[i]) + "\t" + str(label_list[i]) + "\t" + str(pre_list[i]) + "\t" + str(today_close[i]) + "\t" + str(halfday_close[i])  + '\n' )
        #                #fo.write(str(label_list[i]) + "\t" + str(date[i]) + "\t" + str(stocks[i]) + '\n' )
        #    fo.close()
    
        torch.save(model.state_dict(), forced_saving_path + '.' + str(epoch))
        print('Finish saving epoch model !')
