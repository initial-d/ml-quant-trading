# train dataset adding  id seq feature 
from model import *
from dataset import *
import torch.nn.functional as F
from torch.utils.data.dataloader import default_collate
import torch
import numpy as np
import os
from sklearn.metrics import roc_auc_score

model_saving_path = '/da2/search/wanghexiang/stock_data/save/model_v2.pt'
forced_saving_path = '/da2/search/wanghexiang/stock_data/save/model_force_v2.pt'
device = torch.device('cuda:3')
dir_path = '/da2/search/wanghexiang/stock_data/split'


global_lr = 0.01
model = StockPred_v2().to(device)
# optimizer = torch.optim.SGD(model.parameters(), lr=global_lr)
# scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=300, gamma = 0.93)
optimizer = torch.optim.Adam(model.parameters(),lr=0.00
criterion = nn.BCEWithLogitsLoss()
n_epochs = 10


loss_stat = []
best_loss = 100000
dataset = IdStockData('shuf_id_train_data', 89079682)


# pre load label list:
# label_list = []
# with open('/da2/search/wanghexiang/stock_data/test_data', 'r') as f:
#     for line in f:
#         if len(label_list) < 1280000:
#             ll = line.strip('\n\r').split('\t')[-1].split('|')[-1]
#             label_list.append(float(ll))

batch_size = 128
counter = 1000
test_model_counter = 200 * counter
best_auc = 0.0

for epoch in np.arange(n_epochs):
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
        id_t_data, t_data, l_data = dataset.get_batch_data(128)

        optimizer.zero_grad()
        pre = model.forward(id_t_data.to(device), t_data.to(device))

        loss = criterion(pre, l_data.to(device))
        loss.backward()
        optimizer.step()

        train_loss += loss.cpu().detach().numpy().item()
        if batch_num % counter == 0 and batch_num != 0:
            avg_train_loss = train_loss / counter
            tt = time.localtime()
            print(time.strftime('%Y-%m-%d %H:%M:%S',tt), 'Epoch:', epoch, ' Batch num:', int(batch_num / counter), ' Average loss:', avg_train_loss, 'LR: ',  optimizer.param_groups[0]['lr'])
            epoch_loss += train_loss
            train_loss = 0

        
        if batch_num != 0 and batch_num % test_model_counter == 0:
            print('starting test model !')
            pre_list = []
            label_list = []
            _valid_data = IdStockData('id_test_data', 6301621)
            valid_batch = 256
            total_valid = int(np.ceil(_valid_data.f_len / valid_batch))
            for i in range(20000):
                id_t_test_data, t_test_data, l_test_data = _valid_data.get_batch_data(256)
                label_list += l_test_data.numpy().reshape(-1).tolist()
                model.eval()
                with torch.no_grad():
                    pre = model.valid_batch(id_t_test_data.to(device), t_test_data.to(device))
                    pre = F.sigmoid(pre)
                    out_put = pre.detach().cpu().numpy()
                    pre_list += np.reshape(out_put,[256]).tolist()

            y_true = np.array(label_list)
            y_scores = np.array(pre_list)
            auc = roc_auc_score(y_true, y_scores)
            print(("Epoch: %d:  AUC: %.4f") % (epoch, auc))
            if auc > best_auc :
                best_auc = auc
                torch.save(model.state_dict(), model_saving_path)
                print('Finish saving model !')
    

    torch.save(model.state_dict(), forced_saving_path)
    print('Finish saving epoch model !')
