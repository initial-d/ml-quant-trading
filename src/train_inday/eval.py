# -*- coding:utf-8 -*-

import os

#os.environ["MODIN_ENGINE"] = "ray"

import numpy as np
import pandas as pd
#import modin.pandas as pd
import xgboost as xgb
import configparser

from math import ceil
from sklearn import metrics
import time
#import datetime
from time import strftime
from xgboost import plot_tree
from scipy.spatial.distance import cosine
import joblib
from collections import defaultdict
from datetime import date, datetime, timedelta


def time_increase(begin_time,days):
    ts = time.strptime(str(begin_time),"%Y-%m-%d")
    ts = time.mktime(ts)
    dateArray = datetime.datetime.utcfromtimestamp(ts)
    date_increase = (dateArray+datetime.timedelta(days=days)).strftime("%Y-%m-%d")
    return  date_increase
 

class TrainningModel(object):


    def __init__(self, data_path):

        self.pdata = data_path
        self.f1 = self.pdata + '/daily_data_with_fh.txt.sort_by_day_amount'
        self.f2 = self.pdata + '/AShareMerge_1.pickle'
        #self.f4 = self.pdata + '/Ashares2train_origin.pickle'
        self.f4 = self.pdata + '/Ashares2train.pickle'
        self.f5 = self.pdata + '/jointquantprocessed/JointIndustry2train.pickle'
        #self.model = os.path.dirname(os.path.dirname(__file__)) + '/data/0001_20190410.model'
        self.model = os.path.dirname(os.path.dirname(__file__)) + '/data/0001_20171229.model'
        self.presentdate = int(strftime("%Y%m%d") + '10')
        #self.load_feature_list(self.pdata + '/fea.ini')
        self.xgb_param = {'max_depth':6, 'eta':0.01,  'min_child_weight':1, 'gamma':0.0, 'subsample':0.65, 'colsample_bytree':0.65, 'lambda':0.5, 'alpha':0.5, 'scale_pos_weight':1, 'objective':'binary:logistic', 'eval_metric':'mae'} 


    def load_feature_list(self, featurelist_fn):

        cf = configparser.ConfigParser()
        cf.read(featurelist_fn)
        feaitems = cf.items('allfeatures')
        tmpfeature = dict((i[0],i[1]) for i in feaitems)
        fealist = tmpfeature['fealist']
        fealist = fealist.split(',')
        print("fealist训练特征如下：", fealist)
        self.col = fealist.copy()
        print(self.f4)
        #self.df = pd.read_pickle(self.f4)
        #df1 = pd.read_pickle('data/2019/Ashares2train_origin.pickle')
        #self.df = pd.concat([self.df,df1])


    def print_result(self, y_test, y_pred, top=-1):
        #y_pred_binary = (y_pred >= 0.5)*1
        y_pred_binary = [1 if y > 0.5 else 0 for y in y_pred]
        #print(sorted(y_pred, key=lambda e:e, reverse=True)[:100])
        if top > 1:
            th = sorted(y_pred, key=lambda e:e, reverse=True)[top]
            #y_pred_binary = (y_pred >= th)*1
            print('分类阈值: ' +  str(th) + '  top:' + str(top))
            y_pred_binary = [1 if y > th else 0 for y in y_pred]
            pass
        print('AUC: %.4f' % metrics.roc_auc_score(y_test,y_pred))
        #print('ACC: %.4f' % metrics.accuracy_score(y_test,y_pred_binary))
        print('Recall: %.4f' % metrics.recall_score(y_test,y_pred_binary))
        #print('F1-score: %.4f' % metrics.f1_score(y_test,y_pred_binary))
        print('Precesion: %.4f' % metrics.precision_score(y_test,y_pred_binary))
        tn, fp, fn, tp = metrics.confusion_matrix(y_test,y_pred_binary).ravel()
        print('tn, fp, fn, tp: %s %s %s %s' % (tn, fp, fn, tp) )
        print('sensitivity: %.4f' % ( 1.*tp / (tp+fp) ) )
        print('specificity: %.4f' % ( 1.*tn / (tn+fp) ) )


    def predictbyday(self):

        #finaldata = finaldata[(finaldata['date'] > begindate)]
        print(df)
        df = self.df.dropna()

        print("start training")
        featurearray = df[self.col].to_numpy()
        y_train = df['self.target_01']
        print(y_train.to_numpy())
        print(featurearray[:,0])
        y_train = featurearray[:,0]
        y_train = np.where(y_train>0.002, 1,0)
        print(y_train)
    
        y_ext = []
        y_pred_ext = []
        #y_pred_ext = np.ndarray()
        num_round = 3
        gap = 200000
        day = 5000
        n = len(y_train)/day
        for i in range(20,int(n)):
            start = i*day - gap
            end   = start + gap
            print('start:%s end:%s' % (start, end))
            dtrain = xgb.DMatrix(featurearray[start:end,1:-3], y_train[start:end])
            #dtest = xgb.DMatrix(featurearray[end:end+day,1:], y_train[end:end+day])
            dtest = xgb.DMatrix(featurearray[end+3000:end+day,1:-3], y_train[end+3000:end+day])
            bst = xgb.train(self.xgb_param, dtrain, num_round)
            y_pred = bst.predict(dtest)
            y_test = dtest.get_label()
            self.print_result(y_test,y_pred)
            #print(type(y_pred))
            y_ext.extend(list(y_test))
            for i in y_pred:
                y_pred_ext.append(i)
            #print(y_pred_ext[:20])
            print('total:', '##################')
            self.print_result(y_ext, y_pred_ext)


    def predictbyyear(self):
        
        df = self.df.dropna(axis=0, subset=['self.target_01'])
        #df = self.df.dropna()
        #df = self.df

        begindate = '2020-01-02 09:30'
        enddate   = '2020-12-31 14:50'

        train_df = df.loc[ (df['date'] >= begindate) & (df['date'] <= enddate) ]
        df1 = train_df.loc[ (df['self.target_01'] >= 0.01) ]
        df2 = train_df.loc[ (df['self.target_01'] < 0) ]
        train_df = pd.concat([df1,df2],axis=0)

        i = 1
        print(train_df['self.target_01'])
        print(train_df['date'])
        print(train_df['stocks'])
        #train_df.drop(columns=['self.stock006', 'self.best_010'])
        featurearray = train_df[self.col].to_numpy()
        y_train = featurearray[:,0]
        y_train = np.where(y_train>0.01, 1,0)
        print(y_train)
        x_train = featurearray[:,1:-3]
 
        #test
        tt = y_train[y_train == 1]
        print(len(y_train))
        print(len(tt))
        print("#######")

    
        begindate = '2021-01-04 09:30'
        enddate   = '2021-03-31 14:50'
        test_df = df.loc[ (df['date'] >= begindate) & (df['date'] <= enddate) ]
        #test_df = train_df
        
        print(test_df)
        featurearray = test_df[self.col].to_numpy()
        y_test = featurearray[:,0]
        y_test = np.where(y_test>0.01, 1,0)
        print(y_test)
        print(len(y_test))
        x_test = featurearray[:,1:-3]
    
        dtrain = xgb.DMatrix(x_train, y_train)
        dtest = xgb.DMatrix(x_test, y_test)
        num_round = 3

        bst = xgb.train(self.xgb_param, dtrain, num_round)
        
        joblib.dump(bst, self.pdata + '/xgb_model')
        #bst = joblib.load(self.pdata + '/xgb_model')

        y_pred = bst.predict(dtest)
        y_test = dtest.get_label()

        #y_pred_tmp = []
        #y_test_tmp = []
        #high = sorted(y_pred, key=lambda e:e, reverse=True)[30000]
        #low = sorted(y_pred, key=lambda e:e, reverse=True)[2000000]
        #for i in range(0, len(y_pred)):
        #    if y_pred[i] > high and y_pred[i] < low:
        #        continue
        #    else:
        #       y_pred_tmp.append(y_pred[i])
        #       y_test_tmp.append(y_test[i])
        #y_pred = np.array(y_pred_tmp)
        #y_test = np.array(y_test_tmp)  

        #sorted(y_pred, key=lambda e:e, reverse=True)[0:125000]
        #sorted(y_test, key=lambda e:e, reverse=True)[0:125000]

        #self.print_result(y_test, y_pred)
        self.print_result(y_test, y_pred, 500)
        self.print_result(y_test, y_pred, 1000)
        self.print_result(y_test, y_pred, 2000)
        self.print_result(y_test, y_pred, 3000)
        self.print_result(y_test, y_pred, 4000)
        self.print_result(y_test, y_pred, 5000)
        self.print_result(y_test, y_pred, 6000)
        self.print_result(y_test, y_pred, 7000)
        self.print_result(y_test, y_pred, 8000)
        self.print_result(y_test, y_pred, 9000)
        self.print_result(y_test, y_pred, 10000)
        self.print_result(y_test, y_pred, 50000)
        self.print_result(y_test, y_pred, 100000)
        self.print_result(y_test, y_pred, 500000)
        self.print_result(y_test, y_pred, 800000)
        self.print_result(y_test, y_pred, 1000000)
        #plot_tree(bst)

    def get_cos(self, x):
        #print(x)
        return 1 - cosine((x['self.target_01'] + 1).apply(np.log), x['y_pred'])

    def get_pearson(self, x):
        #print(x)
        #return (x['self.target_01'] + 1).apply(np.log).corr( x['y_pred'].apply(np.log) )
        #return (x['self.target_01'] + 1).apply(np.log).corr( x['y_pred'] )
        return x['target_01'].corr( x['y_pred'] )

    def print_cross(self, x, f):
        #print(x)
        #f.write(x['date'].iloc[0] + "\t" + ' '.join(x['stocks'].astype(str).str.zfill(6).str.cat(x['y_pred'].apply(np.square).astype(str), sep='_').values) + '\n')
        f.write(x['date'].iloc[0] + "\t" + ' '.join(x['stocks'].astype(str).str.zfill(6).str.cat(x['y_pred'].astype(str), sep='_').values) + '\n') # not square
        #return x['stocks'].str.cat(x['y_pred'], sep='_').agg('str': lambda x: ', '.join(x))

    def predictbyyear_return_valid(self):
        
        test_df = pd.read_csv('f_train.eval.0', header=None,sep='\t',names=['y_test', 'self.target_01', 'y_pred', 'stocks', 'date'], index_col=False)
        y_test = test_df['y_test']
        y_pred = test_df['y_pred']
        self.print_result(y_test, y_pred)

        #test_df['date'] = test_df['ts'].map(lambda x: time.strftime('%Y-%m-%d %H:%M', time.localtime(int(x)*60)))
        #test_df['date'] = test_df['ts'].map(lambda x: datetime.strptime(x, '%Y-%m-%d %H:%M:%S').strftime("%Y-%m-%d %H:%M"))
	

        #self.print_result(y_test, y_pred)
        ret = 1.0
        ret_avg = 0.0
        ret_cnt = 0 
        th = 0.5
        th = sorted(y_pred, key=lambda e:e, reverse=True)[100000]

        trade_df = test_df.loc[ (test_df['y_pred'] > th)]
        starttime =  "2021-01-04 09:30:00";
        endtime =  "2021-03-31 14:50:00";
        day_return = defaultdict(lambda:1)
        day_daily_retrun = defaultdict(list)

        while starttime <= endtime:
            curtime = starttime
            dt = datetime.strptime(curtime, '%Y-%m-%d %H:%M:%S');
            minute = dt.strftime("%Y-%m-%d %H:%M")
            day = dt.strftime("%Y%m%d")
     
            cur_df = trade_df.loc[ (trade_df['date'] == minute) ]
            if len(cur_df) > 0:
                r = 0
                k = 0
                cur_df.sort_values(by=['y_pred'], ascending=False, inplace=True)
                for i, (index, row) in enumerate(cur_df.iterrows()):
                    r += row['self.target_01']
                    k = i+1
                    print(minute, row['stocks'], row['self.target_01'], row['y_pred'], row['self.target_01'])
                    if i > 30:
                        break
                ret_avg += r/k
                trade_ratio = 1.0
                ret =  trade_ratio * ret * ( 1 + r/k) + ( 1 - trade_ratio ) * ret
                ret_cnt += 1
                day_return[day] = trade_ratio * day_return[day] * ( 1 + r/k ) + ( 1 - trade_ratio) * day_return[day]
                day_daily_retrun[day].append(day_return[day])

            starttime = str((dt + timedelta(seconds=600)).strftime("%Y-%m-%d %H:%M:%S"))
        print("平均收益率: %.6f\t%.6f\t%.6f" % (ret_avg/ret_cnt, ret_avg, ret_cnt))
        print("累计收益: %.6f" % (ret))
        for k,v in day_return.items():
            x = np.array(list( day_daily_retrun[k]))
            sharp_ratio = x.mean()/x.std()
            print('%s\t%.2f\t%.2f' % (k,v, sharp_ratio))
        
        #plot_tree(bst)


    def predictbyyear_cos(self):
        
        #df = self.df.dropna(axis=0, subset=['self.target_01'])
        #df = self.df
        
        test_df = pd.read_csv('predict_file', header=None,sep='\t',names=['y_test', 'self.target_01', 'y_pred', 'stocks', 'date'], index_col=False)
        #test_df = pd.read_csv('predict_file', header=None,sep='\t',names=['y_test', 'self.target_01', 'y_pred', 'stocks', 'date'], index_col=False)
        y_test = test_df['y_test']
        y_pred = test_df['y_pred']

        #test_df['date'] = test_df['ts'].map(lambda x: time.strftime('%Y-%m-%d %H:%M', time.localtime(int(x)*60)))
	
        test_df.sort_values(by=['date', 'stocks'], ascending=True, inplace=True)
        cos = test_df.groupby(['date']).apply(self.get_cos)

        #cross_group = test_df.groupby(['date'])
        #f = open("cross_data", 'w+')
        #cross_df = cross_group.apply(self.print_cross, f)
        #f.close()
        
        #df_09 = test_df[test_df['date'].str.contains(' 09:')].groupby(['date']).apply(self.get_pearson)
        #df_10 = test_df[test_df['date'].str.contains(' 10:')].groupby(['date']).apply(self.get_pearson)
        #df_11 = test_df[test_df['date'].str.contains(' 11:')].groupby(['date']).apply(self.get_pearson)
        #df_13 = test_df[test_df['date'].str.contains(' 13:')].groupby(['date']).apply(self.get_pearson)
        #df_14 = test_df[test_df['date'].str.contains(' 14:')].groupby(['date']).apply(self.get_pearson)
       	#print('09') 
       	#print(df_09.describe()) 
       	#print('10') 
       	#print(df_10.describe()) 
       	#print('11') 
       	#print(df_11.describe()) 
       	#print('13') 
       	#print(df_13.describe()) 
       	#print('14') 
       	#print(df_14.describe()) 
        print('day')
       	print(cos.describe()) 

        #self.print_result(y_test, y_pred)
        #self.print_result(y_test, y_pred)
        #plot_tree(bst)

    def predictbyyear_pearson(self):
        
        #df = self.df.dropna(axis=0, subset=['self.target_01'])
        #df = self.df
        
        #test_df = pd.read_csv('olhc.2021.fea.bin.eval.0', header=None,sep=' ',names=['y_test', 'stocks', 'ts', 'self.target_01', 'y_pred'], index_col=False)
        #test_df = pd.read_csv('d1.fea.test.bin.eval.0', header=None,sep='\t',names=['y_test', 'stocks', 'date', 'self.target_01', 'y_pred'], index_col=False)
        #test_df = pd.read_csv('new_pre', header=None,sep='\t',names=['y_test', 'stocks', 'date', 'self.target_01', 'y_pred'], index_col=False)
        test_df = pd.read_csv('new_pre', header=None,sep='\t',names=['stocks', 'date', 'target_01', 'y_pred', 'dclose', 'hcloes'], index_col=False)
        #test_df = pd.read_csv('olhc.2021.fea.bin.eval.1', header=None,sep='\t',names=['y_test', 'stocks', 'date', 'self.target_01', 'y_pred'], index_col=False)
        #test_df = test_df.loc[test_df.target_01 > 0]
        #test_df = test_df.loc[(test_df.y_pred > 0) & (test_df.y_pred < 1)]
        y_test = test_df['target_01']
        y_pred = test_df['y_pred']

        #test_df['date'] = test_df['date'].map(lambda x: time.strftime('%Y-%m-%d %H:%M', time.localtime(int(x)*60)))
	
        test_df.sort_values(by=['date', 'stocks'], ascending=True, inplace=True)
        cos = test_df.groupby(['date']).apply(self.get_pearson)

        #cross_group = test_df.groupby(['date'])
        #f = open("cross_data", 'w+')
        #cross_df = cross_group.apply(self.print_cross, f)
        #f.close()
        
        #df_09 = test_df[test_df['date'].str.contains(' 09:')].groupby(['date']).apply(self.get_pearson)
        #df_10 = test_df[test_df['date'].str.contains(' 10:')].groupby(['date']).apply(self.get_pearson)
        #df_11 = test_df[test_df['date'].str.contains(' 11:')].groupby(['date']).apply(self.get_pearson)
        #df_13 = test_df[test_df['date'].str.contains(' 13:')].groupby(['date']).apply(self.get_pearson)
        #df_14 = test_df[test_df['date'].str.contains(' 14:')].groupby(['date']).apply(self.get_pearson)
       	#print('09') 
       	#print(df_09.describe()) 
       	#print('10') 
       	#print(df_10.describe()) 
       	#print('11') 
       	#print(df_11.describe()) 
       	#print('13') 
       	#print(df_13.describe()) 
       	#print('14') 
       	#print(df_14.describe()) 
        #print('day')
       	print(cos.describe()) 

        #self.print_result(y_test, y_pred)
        #self.print_result(y_test, y_pred)
        #plot_tree(bst)
    
    def backtesting_by_year(self, train_start_date, train_end_date, test_start_date, test_end_date):

        df = self.df.dropna()

        train_df = df.loc[ (df['date'] >= train_start_date) & (df['date'] <= train_end_date) ]
        featurearray = train_df[self.col].to_numpy()
        y_train = featurearray[:,0]
        y_train = np.where(y_train>0.005, 1,0)
        print(y_train)
        print(train_df)
        x_train = featurearray[:,1:-3]

        test_df = df.loc[ (df['date'] >= test_start_date) & (df['date'] <= test_end_date) ]
        #test_df = train_df
        
        print(test_df)
        featurearray = test_df[self.col].to_numpy()
        y_test = featurearray[:,0]
        y_test = np.where(y_test>0.005, 1,0)
        print(y_test)
        print(len(y_test))
        x_test = featurearray[:,1:-3]
  
        dtrain = xgb.DMatrix(x_train, y_train)
        dtest = xgb.DMatrix(x_test, y_test)
        num_round = 3
        bst = xgb.train(self.xgb_param, dtrain, num_round)
        y_pred = bst.predict(dtest)
        y_test = dtest.get_label()
        self.print_result(y_test, y_pred, 1000)
        test_df['pred'] = y_pred
        test_df.sort_values(by=['stocks', 'date'], ascending=True, inplace=True)
        test_df['return'] = (test_df['self.close'].shift(-1) - test_df['self.close']) / test_df['self.close']
        th = sorted(y_pred, key=lambda e:e, reverse=True)[2000]
        trade_df = test_df.loc[ (test_df['pred'] > 0.5) ]
        print(trade_df)
        ret = 1
        year_return = {}
        for i in range(1, 20000):
            d = time_increase(test_start_date,i)
            if d > test_end_date: break
            cur_df = trade_df.loc[trade_df['date'] == d]
            year = d.split('-')[0]
            if year not in year_return:
                year_return[year] = 1
            if(len(cur_df)):
                print('%s\t%s\t%s' % (d, len(cur_df), ret) )
                n = len(cur_df)
                k = 0
                r = 0
                for index, row in cur_df.iterrows():
                    print(row['stocks'], row['pred'], row['return'])
                    if (row['return'] > -0.11 and row['return'] < 0.11):
                        r += row['return']
                        k +=1
                if k:
                    ret =  0.9989 * 0.5 * ret * (1+r/k) + 0.5 * ret
                    year_return[year] = 0.9989 * 0.5 * year_return[year] * (1+r/k) + 0.5 * year_return[year]
        for k,v in year_return.items():
            print(k,v)
                    




if __name__ == '__main__':

    train_model = TrainningModel('data/inday')
    #train_model.predictbyday()
    #train_model.predictbyyear_cos()
    train_model.predictbyyear_pearson()
    #train_model.predictbyyear_return_valid()
	
    train_start = '2011-01-01'
    train_end   = '2014-12-31'
    test_start  = '2015-01-01'
    test_end    = '2021-12-31'
    #train_model.backtesting_by_year(train_start, train_end, test_start, test_end)

