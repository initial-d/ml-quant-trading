# -*- coding:utf-8 -*-

import os

#os.environ["MODIN_ENGINE"] = "ray"

import numpy as np

from math import ceil
from sklearn import metrics
import time
#import datetime
from time import strftime
from collections import defaultdict
from datetime import date, datetime, timedelta
import codecs


#IN = codecs.open('d1.fea.train.bin.model.0', 'r', encoding='utf-8')  #utf-8
#IN = codecs.open('model', 'r', encoding='utf-8')  #utf-8
IN = codecs.open('model', 'r', encoding='utf-8')  #utf-8

line = IN.readline()
wl = []
while line:
    line = line.rstrip()
    #hs300_list.append(line) 
    wl.append(list(map(lambda x:float(x), line.split(' '))))
    line = IN.readline()
IN.close()

layer1_weight = np.transpose(np.around(np.array(wl[0], dtype=np.float32), decimals=7).reshape(159, 512))
layer1_bias = np.around(np.array(wl[1], dtype=np.float32), decimals=7)
layer2_weight = np.transpose(np.around(np.array(wl[2], dtype=np.float32), decimals=7).reshape(512, 256))
layer2_bias = np.around(np.array(wl[3], dtype=np.float32), decimals=7)
layer3_weight = np.transpose(np.around(np.array(wl[4], dtype=np.float32), decimals=7).reshape(256, 128))
layer3_bias = np.around(np.array(wl[5], dtype=np.float32), decimals=7)
layer4_weight = np.transpose(np.around(np.array(wl[6], dtype=np.float32), decimals=7).reshape(128, 64))
layer4_bias = np.around(np.array(wl[7], dtype=np.float32), decimals=7)
layer5_weight = np.transpose(np.around(np.array(wl[8], dtype=np.float32), decimals=7).reshape(64, 1))
layer5_bias = np.around(np.array(wl[9], dtype=np.float32), decimals=7)


#layer1_weight = np.around(np.array(wl[0], dtype=np.float32), decimals=7).reshape(512, 772)
#layer1_bias = np.around(np.array(wl[1], dtype=np.float32), decimals=7)
#layer2_weight = np.around(np.array(wl[2], dtype=np.float32), decimals=7).reshape(256, 512)
#layer2_bias = np.around(np.array(wl[3], dtype=np.float32), decimals=7)
#layer3_weight = np.around(np.array(wl[4], dtype=np.float32), decimals=7).reshape(128, 256)
#layer3_bias = np.around(np.array(wl[5], dtype=np.float32), decimals=7)
#layer4_weight = np.around(np.array(wl[6], dtype=np.float32), decimals=7).reshape(64, 128)
#layer4_bias = np.around(np.array(wl[7], dtype=np.float32), decimals=7)
#layer5_weight = np.around(np.array(wl[8], dtype=np.float32), decimals=7).reshape(1, 64)
#layer5_bias = np.around(np.array(wl[9], dtype=np.float32), decimals=7)

#layer3_weight = np.array(wl[4]).reshape(128, 256)
#layer3_bias = np.array(wl[5])
#layer4_weight = np.array(wl[6]).reshape(64, 128)
#layer4_bias = np.array(wl[7])
#layer5_weight = np.array(wl[8]).reshape(1, 64)
#layer5_bias = np.array(wl[9])

#fp = open(os.path.join('/da2/search/duyimin', 'test_feature_label'), 'r')
#f_len = 1488408

fp = open(os.path.join('/home/duyimin/trade_predict', 'f_test'), 'r')
f_len = 3254549

def get_batch_data(batch_size):
    batch_data = np.around(np.zeros((batch_size, 159), dtype=np.float32), decimals=7)
    batch_label = np.around(np.zeros((batch_size, 1), dtype=np.float32), decimals=7)
    for i in range(batch_size):
        line = fp.readline().strip('\n\r')
        while ((line is None)):
            if not line :
                fp.seek(0, 0)
            line = fp.readline().strip('\n\r')
        
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


        #line_data, label = line.split('\t')
        #data = np.array([float(item) for item in line_data.split(' ')], dtype=np.float32)
        #batch_data[i] = data
        ##batch_data[i][0:1] = data[0:1]
        #batch_label[i][0] = np.log(float(label) + 1.0)

    return batch_data, batch_label

def sigmoid(z):
    return 1/(1 + np.exp(-z))

valid_batch = 1
total_valid = int(np.ceil(f_len / valid_batch))
for i in range(total_valid):
    t_test_data, l_test_data = get_batch_data(valid_batch)
    #print("input##########")
    #print(t_test_data.tolist())
    #print(t_test_data)
    #print('weight#########')
    #print(layer1_weight.tolist())
    #print(layer1_weight)
    #print(layer2_weight.tolist())
    #print(layer2_weight)
    #print('bias#########')
    #print(layer1_bias.tolist())
    #print(layer2_bias.tolist())
    #print(t_test_data[0].tolist())
    #print(layer1_bias.tolist())
    #print(sigmoid((layer1_weight.dot(t_test_data[0]))).tolist())
    #print(sigmoid((layer1_weight.dot(t_test_data[0])) + layer1_bias).tolist())
    #print('out#########')
    lout1 = sigmoid(np.transpose(layer1_weight.dot(np.transpose(t_test_data))) + np.transpose(layer1_bias))
    #lout1 = np.transpose(layer1_weight.dot(np.transpose(t_test_data))) + np.transpose(layer1_bias)
    #print(lout1.tolist())
    #print(lout1)
    lout2 = sigmoid(np.transpose(layer2_weight.dot(np.transpose(lout1))) + np.transpose(layer2_bias))
    #lout2 = sigmoid(np.transpose(layer2_weight.dot(np.transpose(lout1))))
    #print(lout2.tolist())
    #print(lout2)
    lout3 = sigmoid(np.transpose(layer3_weight.dot(np.transpose(lout2))) + np.transpose(layer3_bias))
    lout4 = sigmoid(np.transpose(layer4_weight.dot(np.transpose(lout3))) + np.transpose(layer4_bias))
    lout5 = np.transpose(layer5_weight.dot(np.transpose(lout4))) + np.transpose(layer5_bias)
    print(lout5.tolist()[0][0])

