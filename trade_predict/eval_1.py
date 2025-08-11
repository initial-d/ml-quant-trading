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


IN = codecs.open('model', 'r', encoding='utf-8')  #utf-8

line = IN.readline()
wl = []
while line:
    line = line.rstrip()
    #hs300_list.append(line) 
    wl.append(list(map(lambda x:float(x), line.split(' '))))
    line = IN.readline()
IN.close()

layer1_weight = np.transpose(np.array(wl[0], dtype=np.float32).reshape(772, 512))
layer1_bias = np.array(wl[1], dtype=np.float32)
layer2_weight = np.array(wl[2]).reshape(512, 256)
layer2_bias = np.array(wl[3])
layer3_weight = np.array(wl[4]).reshape(256, 128)
layer3_bias = np.array(wl[5])
layer4_weight = np.array(wl[6]).reshape(128, 64)
layer4_bias = np.array(wl[7])
layer5_weight = np.array(wl[8]).reshape(1, 64)
layer5_bias = np.array(wl[9])

fp = open(os.path.join('/da2/search/duyimin', 'test_feature_label'), 'r')
f_len = 1488408

def get_batch_data(batch_size):
    batch_data = np.zeros((batch_size, 772), dtype=np.float32)
    batch_label = np.zeros((batch_size, 1), dtype=np.float32)
    for i in range(batch_size):
        line = fp.readline().strip('\n\r')
        while ((line is None) or len(line.split('\t')) != 2):
            if not line :
                fp.seek(0, 0)
            line = fp.readline().strip('\n\r')
        
        line_data, label = line.split('\t')
        data = np.array([float(item) for item in line_data.split(' ')], dtype=np.float32)
        batch_data[i] = data
        batch_label[i][0] = np.log(float(label) + 1.0)

    return batch_data, batch_label

def sigmoid(z):
    return 1/(1 + np.exp(-z))

valid_batch = 1
total_valid = int(np.ceil(f_len / valid_batch))
for i in range(total_valid):
    t_test_data, l_test_data = get_batch_data(1)
    #print(t_test_data[0].tolist())
    #print(layer1_bias.tolist())
    #print(sigmoid((layer1_weight.dot(t_test_data[0]))).tolist())
    #print(sigmoid((layer1_weight.dot(t_test_data[0])) + layer1_bias).tolist())
    print(((layer1_weight.dot(t_test_data[0])) + layer1_bias).tolist())

