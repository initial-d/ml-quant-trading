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

model_saving_path = './lr0001_self/model.pt'
forced_saving_path = './lr0001_self/model_force.pt'
#device = torch.device('cuda:0')
device = torch.device('cpu')


global_lr = 0.01
model0 = StockPred_v4().to(device)
model1 = StockPred_v4().to(device)
model2 = StockPred_v4().to(device)
model3 = StockPred_v4().to(device)
model4 = StockPred_v4().to(device)
model5 = StockPred_v4().to(device)
model6 = StockPred_v4().to(device)
model7 = StockPred_v4().to(device)
model8 = StockPred_v4().to(device)
model9 = StockPred_v4().to(device)
model10 = StockPred_v4().to(device)
model11 = StockPred_v4().to(device)
model12 = StockPred_v4().to(device)
model13 = StockPred_v4().to(device)
model14 = StockPred_v4().to(device)
model_list = [model0, model1, model2, model3, model4, model5, model6, model7, model8, model9, model10, model11, model12, model13, model14]
#model.load_state_dict(torch.load('/da1/public/duyimin/trade_predict/model_force.pt.90'))
#optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
# scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=300, gamma = 0.93)
optimizer0 = torch.optim.Adam(model0.parameters(),lr=0.0001)
optimizer1 = torch.optim.Adam(model1.parameters(),lr=0.0001)
optimizer2 = torch.optim.Adam(model2.parameters(),lr=0.0001)
optimizer3 = torch.optim.Adam(model3.parameters(),lr=0.0001)
optimizer4 = torch.optim.Adam(model4.parameters(),lr=0.0001)
optimizer5 = torch.optim.Adam(model5.parameters(),lr=0.0001)
optimizer6 = torch.optim.Adam(model6.parameters(),lr=0.0001)
optimizer7 = torch.optim.Adam(model7.parameters(),lr=0.0001)
optimizer8 = torch.optim.Adam(model8.parameters(),lr=0.0001)
optimizer9 = torch.optim.Adam(model9.parameters(),lr=0.0001)
optimizer10 = torch.optim.Adam(model10.parameters(),lr=0.0001)
optimizer11 = torch.optim.Adam(model11.parameters(),lr=0.0001)
optimizer12 = torch.optim.Adam(model12.parameters(),lr=0.0001)
optimizer13 = torch.optim.Adam(model13.parameters(),lr=0.0001)
optimizer14 = torch.optim.Adam(model14.parameters(),lr=0.0001)

opt_list = [optimizer0, optimizer1, optimizer2, optimizer3, optimizer4, optimizer5, optimizer6, optimizer7, optimizer8, optimizer9, optimizer10, optimizer11, optimizer12, optimizer13, optimizer14]


criterion1 = torch.nn.MSELoss()
criterion2 = torch.nn.BCEWithLogitsLoss()
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



for epoch in np.arange(n_epochs):

    id = epoch % 15
    #id = random.randint(0, 14)
    print(str(id) + ' part')
    dataset = StockData_mini_batch_day_inday_tensor('./tensor_data_inday_train_' + str(id) + '.pickle', device)
    #_valid_data = StockData_mini_batch_day_inday_tensor('./tensor_data_halfday_del8_test.pickle', device)
    dataset.idx = 0
    #_valid_data.idx = 0

    epoch_loss = 0
    total_batches = int(np.ceil(len(dataset) / batch_size))
    print("start train epoch: ", epoch)
    print("total batch num: ", int(np.ceil(total_batches / counter)))
    batch_num = 0 
    train_loss = 0.0
    model_list[id].train()
    for batch_num in range(total_batches):
        # start = time.time()
        t_data, l_data = dataset.get_batch_data(batch_size)

        c_data = l_data.clone()
        nc_data = c_data.numpy()
        nc_data = np.where(nc_data > 0, 1.0, 0.0)
        cl_data = torch.from_numpy(nc_data).to(torch.float32)

        opt_list[id].zero_grad()
        pre = model_list[id].forward(t_data.to(device))

        loss1 = criterion1(pre, l_data)
        loss2 = criterion2(pre, cl_data)
        loss = 0.9 * loss1 + 0.1 * loss2
        loss.backward()
 
        #torch.nn.utils.clip_grad_norm_(model.parameters(), 1)
        opt_list[id].step()

        train_loss += loss.cpu().detach().numpy().item()
        if batch_num % counter == 0 and batch_num != 0:
            avg_train_loss = train_loss / counter
            tt = time.localtime()
            print(time.strftime('%Y-%m-%d %H:%M:%S',tt), 'Epoch:', epoch, ' Batch num:', int(batch_num / counter), ' Average loss:', avg_train_loss, 'LR: ',  opt_list[id].param_groups[0]['lr'])
            epoch_loss += train_loss
            train_loss = 0

    if epoch > 0 and epoch % 14 == 0: 
        print('starting test model !')
        for i in range(15):
            _valid_data = StockData_mini_batch_day_inday_tensor('./tensor_data_inday_test_' + str(i) + '.pickle', device)
            _valid_data.idx = 0
            pre_list = []
            label_list = []
            #_valid_data = StockData('f_test', 3254549)
            total_valid = int(np.ceil(_valid_data.f_len / valid_batch))
            print(_valid_data.f_len)
            fo = open("./lr0001_self/tushare_day." + str(epoch // 15) + '.' + str(i), "w")
            for j in range(total_valid):
                t_test_data, l_test_data, date, stocks, today_close, halfday_close = _valid_data.get_predict_batch_data(valid_batch)
                label_list = (l_test_data.numpy().reshape(-1)).tolist()
                date_list = (date.reshape(-1)).tolist()
                stocks_list = (stocks.reshape(-1)).tolist()
                model_list[i].eval()
                with torch.no_grad():
                    pre = model_list[i].valid_batch(t_test_data.to(device))
                    out_put = pre.detach().cpu().numpy()
                    pre_list = np.reshape(out_put,-1).tolist()
        
                    for ii in range(len(label_list)):
                        fo.write(str(stocks[ii]) + "\t" + str(date[ii]) + "\t" + str(label_list[ii]) + "\t" + str(pre_list[ii]) + "\t" + str(today_close[ii]) + "\t" + str(halfday_close[ii])  + '\n' )
                        #fo.write(str(label_list[i]) + "\t" + str(date[i]) + "\t" + str(stocks[i]) + '\n' )
            fo.close()
    
            torch.save(model_list[i].state_dict(), forced_saving_path + '.' + str(epoch // 15) + '.' + str(i))
        print('Finish saving epoch model !')
