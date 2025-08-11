from model import *
from dataset import *
import torch.nn.functional as F
from torch.utils.data.dataloader import default_collate
import torch
import numpy as np
import os
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error, explained_variance_score, r2_score

# # deal with url request exception
# def collate_fn_filter(batch):
#     batch = list(filter(lambda x: len(x) == 9, batch))
#     if len(batch) == 0:
#         return torch.Tensor()
#     return default_collate(batch)

model_saving_path = '/da1/public/duyimin/trade_predict/model.pt'
forced_saving_path = '/da1/public/duyimin/trade_predict/model_force.pt'
#device = torch.device('cuda:3')
device = torch.device('cpu')

dir_path = '/da1/public/duyimin/trade_predict/'

global_lr = 0.01
model = StockPred_v3().to(device)
#optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
# scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=300, gamma = 0.93)
optimizer = torch.optim.Adam(model.parameters(),lr=0.001)
criterion = torch.nn.MSELoss()
n_epochs = 10

loss_stat = []
#dataset = StockData('f_train', 13177512)


# pre load label list:
# label_list = []
# with open('/da2/search/wanghexiang/stock_data/test_data', 'r') as f:
#     for line in f:
#         if len(label_list) < 1280000:
#             ll = line.strip('\n\r').split('\t')[-1].split('|')[-1]
#             label_list.append(float(ll))


batch_size = 128
counter = 1000

#train_pattern = [('Ashares2train_2011_2012.pickle', '2011-01-01 09:30', '2012-12-31 14:50', 'train'), \
#('Ashares2train_2013_2014.pickle', '2013-01-01 09:30', '2014-12-31 14:50', 'train'), \
#('Ashares2train_2015_2016.pickle', '2015-01-01 09:30', '2016-12-31 14:50', 'train'), \
#('Ashares2train_2017_2018.pickle', '2017-01-01 09:30', '2018-12-31 14:50', 'train')]
train_pattern = [('Ashares2train_2020_2021.pickle', '2020-01-01 09:30', '2020-12-31 14:50', 'train')]

test_pattern = [('Ashares2train_2020_2021.pickle', '2021-01-01 09:30', '2021-12-31 14:50', 'test')]

for epoch in np.arange(n_epochs):

    print("start train epoch: ", epoch)
    print(" ")
    
    for (pkl, begin, end, flag) in train_pattern:
        print("training: " + begin + "=>" + end)
        dataset = StockData_mini_batch_tensor(pkl, begin, end, flag)
        #epoch_loss = 0
        total_batches = int(np.ceil(len(dataset) / batch_size))
        print("total batch num: ", int(np.ceil(total_batches / counter)))
        batch_num = 0 
        train_loss = 0.0
        model.train()
        for batch_num in range(total_batches):
            # start = time.time()
            t_data, l_data = dataset.get_batch_data(min(batch_size, dataset.f_len - batch_size * batch_num))

            optimizer.zero_grad()
            pre = model.forward(t_data.to(device))

            loss = criterion(pre, l_data.to(device))
            loss.backward()

            #torch.nn.utils.clip_grad_norm_(model.parameters(), 1)
            optimizer.step()

            train_loss += loss.cpu().detach().numpy().item()
            if batch_num % counter == 0 and batch_num != 0:
                avg_train_loss = train_loss / counter
                tt = time.localtime()
                print(time.strftime('%Y-%m-%d %H:%M:%S',tt), 'Epoch:', epoch, ' Batch num:', int(batch_num / counter), ' Average loss:', avg_train_loss, 'LR: ',  optimizer.param_groups[0]['lr'])
                #epoch_loss += train_loss
                train_loss = 0

        
    print('starting test model !')
    valid_batch = 256
    fo = open("test.txt." + str(epoch), "w")
    for (pkl, begin, end, flag) in test_pattern:
        print("testing: " + begin + "=>" + end)
        pre_list = []
        label_list = []
        #_valid_data = StockData('f_test', 3254549)
        _valid_data = StockData_mini_batch_tensor(pkl, begin, end, flag)
        total_valid = int(np.ceil(_valid_data.f_len / valid_batch))
        for i in range(total_valid):
            t_test_data, l_test_data, date, stocks = _valid_data.get_predict_batch_data(min(valid_batch, _valid_data.f_len - valid_batch * i))
            label_list = (l_test_data.numpy().reshape(-1)).tolist()
            date_list = (date.reshape(-1)).tolist()
            stocks_list = (stocks.reshape(-1)).tolist()
            model.eval()
            with torch.no_grad():
                pre = model.valid_batch(t_test_data.to(device))
                out_put = pre.detach().cpu().numpy()
                pre_list = np.reshape(out_put,-1).tolist()

                for i in range(len(label_list)):
                    fo.write(str(label_list[i])  + "\t" + str(stocks[i]) + "\t" + str(date[i]) + "\t" + str(label_list[i])  + "\t" + str(pre_list[i]) + '\n' )
                    #fo.write(str(label_list[i]) + "\t" + str(date[i]) + "\t" + str(stocks[i]) + '\n' )
    fo.close()
    #y_true = np.array(label_list)
    #y_scores = np.array(pre_list)
    #mse = mean_squared_error(y_true, y_scores)
    #rmse = np.sqrt(mean_squared_error(y_true, y_scores))
    #mae = mean_absolute_error(y_true, y_scores)
    #r2 = r2_score(y_true, y_scores)
    #print(("Epoch: %d:  MSE: %.4f, RMSE: %.4f, MAE: %.4f, R2: %.4f") % (epoch, mse, rmse, mae, r2))
    # if auc > best_auc :
    #     best_auc = auc
    #torch.save(model.state_dict(), model_saving_path)
    #     print('Finish saving model !')

    torch.save(model.state_dict(), forced_saving_path)
    print('Finish saving epoch model !')
