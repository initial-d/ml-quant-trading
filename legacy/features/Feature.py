# -*- coding:utf-8 -*-


import numpy as np
import pandas as pd
#import modin.pandas as pd
from scipy import stats
import configparser


def ts_rank(data): return (np.argsort(data)[-1] + 1) / len(data)
def func_highday(na): return len(na) - na.argmax()
def func_lowday(na): return len(na) - na.argmin()


def func_rolling_sum(d, v): return d.rolling(v, min_periods=v).sum()
def func_rolling_std(d, v): return d.rolling(v, min_periods=v).std()
def func_rolling_max(d, v): return d.rolling(v, min_periods=v).max()
def func_rolling_min(d, v): return d.rolling(v, min_periods=v).min()
def func_rolling_mean(d, v): return d.rolling(v, min_periods=v).mean()


def func_rolling_corr(d1, d2, v): return d1.rolling(v, min_periods=v).corr(d2)
def func_ewma(d, alpha=0): return d.ewm(alpha=alpha).mean()


def func_decaylinear(na):
    n = len(na)
    decay_weights = np.arange(1, n+1, 1)
    decay_weights = decay_weights / decay_weights.sum()
    return (na * decay_weights).sum()


def func_grad(data):
    xx = range(1, len(data)+1)
    ts = stats.linregress(xx, data)
    if ts.pvalue < 0.05:
        return ts.slope
    else:
        return 0


# conditions
def condnew(DQ):
    amount60 = DQ.rolling(60).mean()
    def f(x): return (-x).rank()
    return amount60.apply(f) <= 1800.0


def condlimit(CLO):
    return abs(CLO/CLO.shift() - 1) <= 0.098


class Func(object):


    # bast features
    # Alpha2  (-1 * DELTA((((CLOSE - LOW) - (HIGH - CLOSE)) / (HIGH - LOW)), 1))

    def best_001(self, type=1):
        part = -((2*self.S_FWDS_ADJCLOSE-self.S_FWDS_ADJLOW-self.S_FWDS_ADJHIGH)/(self.S_FWDS_ADJHIGH-self.S_FWDS_ADJLOW))
        alpha = part.rolling(5).apply(ts_rank)
        return alpha

# Alpha2  (-1 * DELTA((((CLOSE - LOW) - (HIGH - CLOSE)) / (HIGH - LOW)), 1))
    def best_002(self, type=1):
        part = -((2*self.S_FWDS_ADJCLOSE-self.S_FWDS_ADJLOW-self.S_FWDS_ADJHIGH)/(self.S_FWDS_ADJHIGH-self.S_FWDS_ADJLOW))
        alpha = part.rolling(10).apply(ts_rank)
        return alpha

# Alpha2  (-1 * DELTA((((CLOSE - LOW) - (HIGH - CLOSE)) / (HIGH - LOW)), 1))
    def best_003(self, type=1):
        part = -((2*self.S_FWDS_ADJCLOSE-self.S_FWDS_ADJLOW-self.S_FWDS_ADJHIGH)/(self.S_FWDS_ADJHIGH-self.S_FWDS_ADJLOW))
        alpha = part.rolling(20).apply(ts_rank)
        return alpha
        alpha[alpha >= 1.0E+38] = 1.0E+38
        alpha[alpha <= -1.0E+38] = -1.0E+38
        if type:
            return alpha[self.condperiod]
        else:
            return alpha[self.condperiod].iloc[-1, :]

# Alpha2  (-1 * DELTA((((CLOSE - LOW) - (HIGH - CLOSE)) / (HIGH - LOW)), 1))
    def best_004(self, type=1):
        alpha = -((2*self.S_FWDS_ADJCLOSE-self.S_FWDS_ADJLOW-self.S_FWDS_ADJHIGH)/(self.S_FWDS_ADJHIGH-self.S_FWDS_ADJLOW)).diff(3)
        return alpha
        alpha[alpha >= 1.0E+38] = 1.0E+38
        alpha[alpha <= -1.0E+38] = -1.0E+38
        if type:
            return alpha[self.condperiod]
        else:
            return alpha[self.condperiod].iloc[-1, :]

# Alpha2  (-1 * DELTA((((CLOSE - LOW) - (HIGH - CLOSE)) / (HIGH - LOW)), 1))
    def best_005(self, type=1):
        alpha = -((2*self.S_FWDS_ADJCLOSE-self.S_FWDS_ADJLOW-self.S_FWDS_ADJHIGH)/(self.S_FWDS_ADJHIGH-self.S_FWDS_ADJLOW)).diff(5)
        return alpha
        alpha[alpha >= 1.0E+38] = 1.0E+38
        alpha[alpha <= -1.0E+38] = -1.0E+38
        if type:
            return alpha[self.condperiod]
        else:
            return alpha[self.condperiod].iloc[-1, :]

# Alpha2  (-1 * DELTA((((CLOSE - LOW) - (HIGH - CLOSE)) / (HIGH - LOW)), 1))
    def best_006(self, type=1):
        alpha = -((2*self.S_FWDS_ADJCLOSE-self.S_FWDS_ADJLOW-self.S_FWDS_ADJHIGH)/(self.S_FWDS_ADJHIGH-self.S_FWDS_ADJLOW))
        alpha = func_ewma(alpha, alpha=1/10.0)
        return alpha
        alpha[alpha >= 1.0E+38] = 1.0E+38
        alpha[alpha <= -1.0E+38] = -1.0E+38
        if type:
            return alpha[self.condperiod]
        else:
            return alpha[self.condperiod].iloc[-1, :]

# Alpha2  (-1 * DELTA((((CLOSE - LOW) - (HIGH - CLOSE)) / (HIGH - LOW)), 1))
    def best_007(self, type=1):
        alpha = (-((2*self.S_FWDS_ADJCLOSE-self.S_FWDS_ADJLOW-self.S_FWDS_ADJHIGH)/(self.S_FWDS_ADJHIGH-self.S_FWDS_ADJLOW))).rank(pct=True)
        return alpha
        alpha[alpha >= 1.0E+38] = 1.0E+38
        alpha[alpha <= -1.0E+38] = -1.0E+38
        if type:
            return alpha[self.condperiod]
        else:
            return alpha[self.condperiod].iloc[-1, :]

    def best_008(self, type=1):
        alpha = func_rolling_mean(self.S_DQ_AVGPRICE / self.S_FWDS_ADJCLOSE, 3)
        return alpha
        alpha[alpha >= 1.0E+38] = 1.0E+38
        alpha[alpha <= -1.0E+38] = -1.0E+38
        if type:
            return alpha[self.condperiod]
        else:
            return alpha[self.condperiod].iloc[-1, :]

    def best_009(self, type=1):
        part1 = func_ewma(self.S_DQ_AVGPRICE, alpha=1/5.0)
        part2 = func_ewma(self.S_FWDS_ADJCLOSE, alpha=1/5.0)
        alpha = part1 / part2
        return alpha
        alpha[alpha >= 1.0E+38] = 1.0E+38
        alpha[alpha <= -1.0E+38] = -1.0E+38
        if type:
            return alpha[self.condperiod]
        else:
            return alpha[self.condperiod].iloc[-1, :]

    def best_010(self, type=1):
        alpha = (self.S_DQ_AVGPRICE / self.S_FWDS_ADJCLOSE - 1) * self.S_DQ_VOLUME
        return alpha
        alpha[alpha >= 1.0E+38] = 1.0E+38
        alpha[alpha <= -1.0E+38] = -1.0E+38
        if type:
            return alpha[self.condperiod]
        else:
            return alpha[self.condperiod].iloc[-1, :]

    # Alpha15  OPEN/DELAY(CLOSE,1)-1
    def best_011(self, type=1):
        alpha = (self.S_FWDS_ADJOPEN/self.S_FWDS_ADJCLOSE.shift()-1).rank(pct=True)
        return alpha
        alpha[alpha >= 1.0E+38] = 1.0E+38
        alpha[alpha <= -1.0E+38] = -1.0E+38
        #result[result > 0.08] = np.nan
        if type:
            return alpha[self.condperiod]
        else:
            return alpha[self.condperiod].iloc[-1, :]

    # Alpha15  OPEN/DELAY(CLOSE,1)-1
    def best_012(self, type=1):
        alpha = func_rolling_mean(self.S_FWDS_ADJOPEN/self.S_FWDS_ADJCLOSE.shift()-1, 5)
        return alpha
        alpha[alpha >= 1.0E+38] = 1.0E+38
        alpha[alpha <= -1.0E+38] = -1.0E+38
        #result[result > 0.08] = np.nan
        if type:
            return alpha[self.condperiod]
        else:
            return alpha[self.condperiod].iloc[-1, :]

    # Alpha15  OPEN/DELAY(CLOSE,1)-1
    def best_013(self, type=1):
        alpha = func_ewma(self.S_FWDS_ADJOPEN/self.S_FWDS_ADJCLOSE.shift()-1, alpha=1/5.0)
        return alpha
        alpha[alpha >= 1.0E+38] = 1.0E+38
        alpha[alpha <= -1.0E+38] = -1.0E+38
        #result[result > 0.08] = np.nan
        if type:
            return alpha[self.condperiod]
        else:
            return alpha[self.condperiod].iloc[-1, :]

    # Alpha5  (-1 * TSMAX(CORR(TSRANK(VOLUME, 5), TSRANK(HIGH, 5), 5), 3))
    def best_014(self, type=1):
        ts_volume = self.S_DQ_VOLUME.rolling(7).apply(ts_rank)
        ts_high = ((self.S_FWDS_ADJHIGH - self.S_FWDS_ADJLOW)/self.S_FWDS_ADJCLOSE).rolling(7).apply(ts_rank)
        # alpha=func_rolling_corr(ts_high,ts_volume,5)
        alpha = ts_high.rolling(5).corr(ts_volume)
        return alpha
        alpha[alpha >= 1.0E+38] = 1.0E+38
        alpha[alpha <= -1.0E+38] = -1.0E+38
        #result[result > 0.08] = np.nan
        if type:
            return alpha[self.condperiod]
        else:
            return alpha[self.condperiod].iloc[-1, :]

    def best_015(self, type=1):
        alpha = ((self.S_FWDS_ADJHIGH - self.S_FWDS_ADJLOW)/self.S_FWDS_ADJCLOSE).rank(pct=True)
        return alpha
        alpha[alpha >= 1.0E+38] = 1.0E+38
        alpha[alpha <= -1.0E+38] = -1.0E+38
        if type:
            return alpha[self.condperiod]
        else:
            return alpha[self.condperiod].iloc[-1, :]

    def best_016(self, type=1):
        alpha = ((self.S_FWDS_ADJHIGH - self.S_FWDS_ADJLOW)/self.S_FWDS_ADJCLOSE)/(1 + np.sqrt(self.S_DQ_VOLUME))
        return alpha
        alpha[alpha >= 1.0E+38] = 1.0E+38
        alpha[alpha <= -1.0E+38] = -1.0E+38
        if type:
            return alpha[self.condperiod]
        else:
            return alpha[self.condperiod].iloc[-1, :]

    # 均线破位
    def best_017(self, type=1):
        part0 = func_ewma(self.S_FWDS_ADJCLOSE, alpha=1/5.0)
        alpha = part0 - part0.shift()
        return alpha
        alpha[alpha >= 1.0E+38] = 1.0E+38
        alpha[alpha <= -1.0E+38] = -1.0E+38
        if type:
            return alpha[self.condperiod]
        else:
            return alpha[self.condperiod].iloc[-1, :]

    # 均线破位
    def best_018(self, type=1):
        part0 = func_ewma(self.S_FWDS_ADJCLOSE, alpha=1/10.0)
        alpha = part0 - part0.shift()
        return alpha
        alpha[alpha >= 1.0E+38] = 1.0E+38
        alpha[alpha <= -1.0E+38] = -1.0E+38
        if type:
            return alpha[self.condperiod]
        else:
            return alpha[self.condperiod].iloc[-1, :]

    # 均线破位
    def best_019(self, type=1):
        part0 = func_ewma(self.S_FWDS_ADJCLOSE, alpha=1/20.0)
        alpha = part0 - part0.shift()
        return alpha
        alpha[alpha >= 1.0E+38] = 1.0E+38
        alpha[alpha <= -1.0E+38] = -1.0E+38
        if type:
            return alpha[self.condperiod]
        else:
            return alpha[self.condperiod].iloc[-1, :]

    # 赚钱效应
    def best_020(self, type=1):
        alpha = func_ewma((self.S_FWDS_ADJCLOSE - self.S_DQ_AVGPRICE)*self.S_DQ_VOLUME, alpha=1/5.0)
        return alpha
        alpha[alpha >= 1.0E+38] = 1.0E+38
        alpha[alpha <= -1.0E+38] = -1.0E+38
        if type:
            return alpha[self.condperiod]
        else:
            return alpha[self.condperiod].iloc[-1, :]

    # Alpha7  ((RANK(MAX((VWAP - CLOSE), 3)) + RANK(MIN((VWAP - CLOSE), 3))) * RANK(DELTA(VOLUME, 3)))
    def best_021(self, type=1):
        part1 = (func_rolling_max((self.S_DQ_AVGPRICE-self.S_FWDS_ADJCLOSE)*self.S_DQ_VOLUME, 3))
        part2 = (func_rolling_min((self.S_DQ_AVGPRICE-self.S_FWDS_ADJCLOSE)*self.S_DQ_VOLUME, 3))
        part3 = (self.S_DQ_VOLUME.diff(3))
        alpha = part1+part2*part3
        return alpha
        alpha[alpha >= 1.0E+38] = 1.0E+38
        alpha[alpha <= -1.0E+38] = -1.0E+38
        if type:
            return alpha[self.condperiod]
        else:
            return alpha[self.condperiod].iloc[-1, :]

    # Alpha1  (-1 * CORR(RANK(DELTA(LOG(VOLUME), 1)), RANK(((CLOSE - OPEN) / OPEN)), 6))
    def stock001(self, type=1):
        data1 = np.log(self.S_DQ_VOLUME + 1).diff(periods=1).rank(pct=True)
        data2 = ((self.S_FWDS_ADJCLOSE - self.S_FWDS_ADJOPEN)/self.S_FWDS_ADJOPEN).rank(pct=True)
        alpha = func_rolling_corr(data1, data2, 4)
        return alpha
        alpha[alpha >= 1.0E+38] = 1.0E+38
        alpha[alpha <= -1.0E+38] = -1.0E+38
        if type:
            return alpha[self.condperiod]
        else:
            return alpha[self.condperiod].iloc[-1, :]

    # Alpha1  (-1 * CORR(RANK(DELTA(LOG(VOLUME), 1)), RANK(((CLOSE - OPEN) / OPEN)), 6))
    def stock002(self, type=1):
        data1 = np.log(self.S_DQ_VOLUME + 1).diff(periods=1)
        data2 = ((self.S_FWDS_ADJCLOSE - self.S_FWDS_ADJOPEN)/self.S_FWDS_ADJOPEN)
        alpha = func_rolling_corr(data1, data2, 6)
        return alpha
        alpha[alpha >= 1.0E+38] = 1.0E+38
        alpha[alpha <= -1.0E+38] = -1.0E+38
        if type:
            return alpha[self.condperiod]
        else:
            return alpha[self.condperiod].iloc[-1, :]

    # Alpha17  RANK((VWAP - MAX(VWAP, 15)))^DELTA(CLOSE, 5)
    def stock003(self, type=1):
        temp1 = func_rolling_max(self.S_DQ_AVGPRICE, 15)
        temp2 = (self.S_FWDS_ADJCLOSE-temp1)
        part1 = temp2.rank(pct=True)
        part2 = self.S_FWDS_ADJCLOSE.diff(5)
        alpha = part1**part2
        return alpha
        alpha[alpha >= 1.0E+38] = 1.0E+38
        alpha[alpha <= -1.0E+38] = -1.0E+38
        if type:
            return alpha[self.condperiod]
        else:
            return alpha[self.condperiod].iloc[-1, :]

    # Alpha22  SMEAN(((CLOSE-MEAN(CLOSE,6))/MEAN(CLOSE,6)-DELAY((CLOSE-MEAN(CLOSE,6))/MEAN(CLOSE,6),3)),12,1)
    def stock004(self, type=1):
        part0 = func_rolling_mean(self.S_FWDS_ADJCLOSE, 6)
        part1 = (self.S_FWDS_ADJCLOSE - part0)/part0
        alpha = part1-part1.shift(3)
        alpha = func_ewma(alpha, alpha=1.0/12)
        return alpha
        alpha[alpha >= 1.0E+38] = 1.0E+38
        alpha[alpha <= -1.0E+38] = -1.0E+38
        if type:
            return alpha[self.condperiod]
        else:
            return alpha[self.condperiod].iloc[-1, :]

    # Alpha29  (CLOSE-DELAY(CLOSE,6))/DELAY(CLOSE,6)*VOLUME
    def stock005(self, type=1):
        delay6 = self.S_FWDS_ADJCLOSE.shift(6)
        alpha = (self.S_FWDS_ADJCLOSE-delay6)*(self.S_DQ_VOLUME + 1)/delay6
        return alpha
        alpha[alpha >= 1.0E+38] = 1.0E+38
        alpha[alpha <= -1.0E+38] = -1.0E+38
        if type:
            return alpha[self.condperiod]
        else:
            return alpha[self.condperiod].iloc[-1, :]

    # Alpha31  (CLOSE-MEAN(CLOSE,12))/MEAN(CLOSE,12)*100
    def stock006(self, type=1):
        part = func_rolling_mean(self.S_FWDS_ADJCLOSE, 12)
        alpha = (self.S_FWDS_ADJCLOSE - part)/part
        return alpha
        alpha[alpha >= 1.0E+38] = 1.0E+38
        alpha[alpha <= -1.0E+38] = -1.0E+38
        if type:
            return alpha[self.condperiod]
        else:
            return alpha[self.condperiod].iloc[-1, :]

    # Alpha33  ((((-1 * TSMIN(LOW, 5)) + DELAY(TSMIN(LOW, 5), 5)) * RANK(((SUM(RET, 240) - SUM(RET, 20)) / 220))) *TSRANK(VOLUME, 5))
    def stock007(self, type=1):
        ret = self.S_FWDS_ADJCLOSE.pct_change()
        temp1 = func_rolling_min(self.S_FWDS_ADJLOW, 5)
        part1 = temp1.shift(5)-temp1
        temp2 = (func_rolling_sum(ret, 60)-func_rolling_sum(ret, 20))/40
        part2 = temp2.rank(pct=True)
        part3 = self.S_DQ_VOLUME.rank(pct=True)
        alpha = part1 * part2 * part3
        return alpha
        alpha[alpha >= 1.0E+38] = 1.0E+38
        alpha[alpha <= -1.0E+38] = -1.0E+38
        if type:
            return alpha[self.condperiod]
        else:
            return alpha[self.condperiod].iloc[-1, :]

    # Alpha37  (-1 * RANK(((SUM(OPEN, 5) * SUM(RET, 5)) - DELAY((SUM(OPEN, 5) * SUM(RET, 5)), 10))))
    def stock008(self, type=1):
        ret = self.S_FWDS_ADJCLOSE.pct_change()
        temp = func_rolling_sum(self.S_FWDS_ADJOPEN, 5)*func_rolling_sum(ret, 5)
        part = temp - temp.shift(10)
        alpha = -part.rank(pct=True)
        return alpha
        alpha[alpha >= 1.0E+38] = 1.0E+38
        alpha[alpha <= -1.0E+38] = -1.0E+38
        if type:
            return alpha[self.condperiod]
        else:
            return alpha[self.condperiod].iloc[-1, :]

    # Alpha46  (MEAN(CLOSE,3)+MEAN(CLOSE,6)+MEAN(CLOSE,12)+MEAN(CLOSE,24))/(4*CLOSE)
    def stock009(self, type=1):
        part1 = func_rolling_mean(self.S_FWDS_ADJCLOSE, 3)
        part2 = func_rolling_mean(self.S_FWDS_ADJCLOSE, 6)
        part3 = func_rolling_mean(self.S_FWDS_ADJCLOSE, 12)
        part4 = func_rolling_mean(self.S_FWDS_ADJCLOSE, 24)
        alpha = (part1+part2+part3+part4)*0.25/self.S_FWDS_ADJCLOSE
        return alpha
        alpha[alpha >= 1.0E+38] = 1.0E+38
        alpha[alpha <= -1.0E+38] = -1.0E+38
        if type:
            return alpha[self.condperiod]
        else:
            return alpha[self.condperiod].iloc[-1, :]

    # Alpha70  STD(AMOUNT,6)
    def stock010(self, type=1):
        alpha = func_rolling_std(np.log(self.S_DQ_AMOUNT + 1), 6)
        return alpha
        alpha[alpha >= 1.0E+38] = 1.0E+38
        alpha[alpha <= -1.0E+38] = -1.0E+38
        if type:
            return alpha[self.condperiod]
        else:
            return alpha[self.condperiod].iloc[-1, :]

    # Alpha78  ((HIGH+LOW+CLOSE)/3-MA((HIGH+LOW+CLOSE)/3,12))/(0.015*MEAN(ABS(CLOSE-MEAN((HIGH+LOW+CLOSE)/3,12)),12))
    def stock011(self, type=1):
        data1 = (self.S_FWDS_ADJHIGH+self.S_FWDS_ADJLOW+self.S_FWDS_ADJCLOSE)/3-func_rolling_mean((self.S_FWDS_ADJHIGH+self.S_FWDS_ADJLOW+self.S_FWDS_ADJCLOSE)/3, 12)
        data2 = abs(self.S_FWDS_ADJCLOSE - func_rolling_mean((self.S_FWDS_ADJHIGH+self.S_FWDS_ADJLOW+self.S_FWDS_ADJCLOSE)/3, 12))
        data3 = func_rolling_mean(data2, 12) * 0.015
        alpha = (data1 / data3)
        return alpha
        alpha[alpha >= 1.0E+38] = 1.0E+38
        alpha[alpha <= -1.0E+38] = -1.0E+38
        if type:
            return alpha[self.condperiod]
        else:
            return alpha[self.condperiod].iloc[-1, :]

    # Alpha102  SMA(MAX(VOLUME-DELAY(VOLUME,1),0),6,1)/SMA(ABS(VOLUME-DELAY(VOLUME,1)),6,1)*100
    def stock012(self, type=1):
        temp1 = self.S_DQ_VOLUME-self.S_DQ_VOLUME.shift()
        part1 = np.maximum(temp1, 0)
        part1 = func_ewma(part1, alpha=1.0/6)
        temp2 = temp1.abs()
        part2 = func_ewma(temp2, alpha=1.0/6)
        alpha = part1*100/part2
        return alpha
        alpha[alpha >= 1.0E+38] = 1.0E+38
        alpha[alpha <= -1.0E+38] = -1.0E+38
        if type:
            return alpha[self.condperiod]
        else:
            return alpha[self.condperiod].iloc[-1, :]

    # Alpha120  (RANK((VWAP - CLOSE)) / RANK((VWAP + CLOSE)))
    def stock013(self, type=1):
        data1 = (self.S_DQ_AVGPRICE-self.S_FWDS_ADJCLOSE).diff()
        data2 = (self.S_DQ_AVGPRICE+self.S_FWDS_ADJCLOSE).diff()
        alpha = (data1/data2)
        return alpha
        alpha[alpha >= 1.0E+38] = 1.0E+38
        alpha[alpha <= -1.0E+38] = -1.0E+38
        if type:
            return alpha[self.condperiod]
        else:
            return alpha[self.condperiod].iloc[-1, :]

    # Alpha134  (CLOSE-DELAY(CLOSE,12))/DELAY(CLOSE,12)*VOLUME

    def stock014(self, type=1):
        alpha = (self.S_FWDS_ADJCLOSE / self.S_FWDS_ADJCLOSE.shift(12) - 1) * self.S_DQ_VOLUME
        return alpha
        alpha[alpha >= 1.0E+38] = 1.0E+38
        alpha[alpha <= -1.0E+38] = -1.0E+38
        if type:
            return alpha[self.condperiod]
        else:
            return alpha[self.condperiod].iloc[-1, :]

    # Alpha144  SUMIF(ABS(CLOSE/DELAY(CLOSE,1)-1)/AMOUNT,20,CLOSE<DELAY(CLOSE,1))/COUNT(CLOSE<DELAY(CLOSE,1),20)
    def stock015(self, type=1):
        df1 = self.S_FWDS_ADJCLOSE < self.S_FWDS_ADJCLOSE.shift()
        sumif = func_rolling_sum(((abs(self.S_FWDS_ADJCLOSE / self.S_FWDS_ADJCLOSE.shift() - 1)/np.log(self.S_DQ_AMOUNT + 1))[df1].fillna(0)), 20)
        count = func_rolling_sum(df1, 20)
        alpha = (sumif / count)
        return alpha
        alpha[alpha >= 1.0E+38] = 1.0E+38
        alpha[alpha <= -1.0E+38] = -1.0E+38
        if type:
            return alpha[self.condperiod]
        else:
            return alpha[self.condperiod].iloc[-1, :]

    # Alpha159  ((CLOSE-SUM(MIN(LOW,DELAY(CLOSE,1)),6))/SUM(MAX(HGIH,DELAY(CLOSE,1))-MIN(LOW,DELAY(CLOSE,1)),6)*12*24+(CLOSE-SUM(MIN(LOW,DELAY(CLOSE,1)),12))/SUM(MAX(HGIH,DELAY(CLOSE,1))-MIN(LOW,DELAY(CLOSE,1)),12)*6*24+(CLOSE-SUM(MIN(LOW,DELAY(CLOSE,1)),24))/SUM(MAX(HGIH,DELAY(CLOSE,1))-MIN(LOW,DELAY(CLOSE,1)),24)*6*24)*100/(6*12+6*24+12*24)
    def stock016(self, type=1):

        # ((CLOSE-SUM(mitt,6))/SUM(matt-mitt,6)*12*24+(CLOSE-SUM(mitt,12))/SUM(matt-mitt,12)*6*24+(CLOSE-SUM(mitt,24))/SUM(matt-mitt,24)*6*24)*100/(6*12+6*24+12*24)
        data1 = np.minimum(self.S_FWDS_ADJLOW, self.S_FWDS_ADJCLOSE.shift(1))
        data2 = np.maximum(self.S_FWDS_ADJHIGH, self.S_FWDS_ADJCLOSE.shift(1))
        part1 = (self.S_FWDS_ADJCLOSE - func_rolling_sum(data1, 6))/func_rolling_sum(data2-data1, 6)*12*24
        part2 = (self.S_FWDS_ADJCLOSE - func_rolling_sum(data1, 12))/func_rolling_sum(data2-data1, 12)*6*24
        part3 = (self.S_FWDS_ADJCLOSE - func_rolling_sum(data1, 24))/func_rolling_sum(data2-data1, 24)*6*24
        alpha = (part1+part2+part3)*100/(6*12+6*24+12*24)
        return alpha
        alpha[alpha >= 1.0E+38] = 1.0E+38
        alpha[alpha <= -1.0E+38] = -1.0E+38
        if type:
            return alpha[self.condperiod]
        else:
            return alpha[self.condperiod].iloc[-1, :]

    # Alpha163  RANK(((((-1 * RET) * MEAN(VOLUME,20)) * VWAP) * (HIGH - CLOSE)))
    def stock017(self, type=1):
        alpha = (-1)*(self.S_FWDS_ADJCLOSE/self.S_FWDS_ADJCLOSE.shift()-1)*func_rolling_mean(self.S_DQ_VOLUME, 20)*self.S_DQ_AVGPRICE*(self.S_FWDS_ADJHIGH-self.S_FWDS_ADJCLOSE)
        return alpha
        alpha[alpha >= 1.0E+38] = 1.0E+38
        alpha[alpha <= -1.0E+38] = -1.0E+38
        if type:
            return alpha[self.condperiod]
        else:
            return alpha[self.condperiod].iloc[-1, :]

    # Alpha170  ((((RANK((1 / CLOSE)) * VOLUME) / MEAN(VOLUME,20)) * ((HIGH * RANK((HIGH - CLOSE))) / (SUM(HIGH, 5) /5))) - RANK((VWAP - DELAY(VWAP, 5))))
    def stock018(self, type=1):
        data1 = (1/self.S_FWDS_ADJCLOSE).rank(pct=True)
        data2 = func_rolling_mean(self.S_DQ_VOLUME, 20)
        part1 = (data1*self.S_DQ_VOLUME)/data2
        data3 = (self.S_FWDS_ADJHIGH-self.S_FWDS_ADJCLOSE).rank(pct=True)
        data4 = func_rolling_mean(self.S_FWDS_ADJHIGH, 5)
        part2 = (data3*self.S_FWDS_ADJHIGH)/data4
        part3 = (self.S_DQ_AVGPRICE-self.S_DQ_AVGPRICE.shift(5)).rank(pct=True)
        alpha = part1 * part2 - part3
        return alpha
        alpha[alpha >= 1.0E+38] = 1.0E+38
        alpha[alpha <= -1.0E+38] = -1.0E+38
        if type:
            return alpha[self.condperiod]
        else:
            return alpha[self.condperiod].iloc[-1, :]

    # Alpha171  ((-1 * ((LOW - CLOSE) * (OPEN^5))) / ((CLOSE - HIGH) * (CLOSE^5)))
    def stock019(self, type=1):
        data1 = -1*(self.S_FWDS_ADJLOW-self.S_FWDS_ADJCLOSE)*(self.S_FWDS_ADJOPEN**5)
        data2 = (self.S_FWDS_ADJCLOSE-self.S_FWDS_ADJHIGH)*(self.S_FWDS_ADJCLOSE**5)
        alpha = (data1/data2)
        return alpha
        alpha[alpha >= 1.0E+38] = 1.0E+38
        alpha[alpha <= -1.0E+38] = -1.0E+38
        if type:
            return alpha[self.condperiod]
        else:
            return alpha[self.condperiod].iloc[-1, :]

    # Alpha178  (CLOSE-DELAY(CLOSE,1))/DELAY(CLOSE,1)*VOLUME

    def stock020(self, type=1):
        alpha = ((self.S_FWDS_ADJCLOSE-self.S_FWDS_ADJCLOSE.shift())/self.S_FWDS_ADJCLOSE.shift()*self.S_DQ_VOLUME)
        return alpha
        alpha[alpha >= 1.0E+38] = 1.0E+38
        alpha[alpha <= -1.0E+38] = -1.0E+38
        if type:
            return alpha[self.condperiod]
        else:
            return alpha[self.condperiod].iloc[-1, :]

    # Alpha188  ((HIGH-LOW–SMA(HIGH-LOW,11,2))/SMA(HIGH-LOW,11,2))*100
    def stock021(self, type=1):
        sma = func_ewma(self.S_FWDS_ADJHIGH - self.S_FWDS_ADJLOW, alpha=2/11)
        alpha = ((self.S_FWDS_ADJHIGH - self.S_FWDS_ADJLOW - sma)/sma*100)
        return alpha
        alpha[alpha >= 1.0E+38] = 1.0E+38
        alpha[alpha <= -1.0E+38] = -1.0E+38
        if type:
            return alpha[self.condperiod]
        else:
            return alpha[self.condperiod].iloc[-1, :]

    # Alpha191  ((CORR(MEAN(VOLUME,20), LOW, 5) + ((HIGH + LOW) / 2)) - CLOSE)
    def stock022(self, type=1):
        volume_avg = func_rolling_mean(self.S_DQ_VOLUME, 20)
        corr = func_rolling_corr(volume_avg, self.S_FWDS_ADJLOW, 5)
        alpha = corr + (self.S_FWDS_ADJHIGH + self.S_FWDS_ADJLOW)/2 - self.S_FWDS_ADJCLOSE
        return alpha
        alpha[alpha >= 1.0E+38] = 1.0E+38
        alpha[alpha <= -1.0E+38] = -1.0E+38
        if type:
            return alpha[self.condperiod]
        else:
            return alpha[self.condperiod].iloc[-1, :]


# extra features
# Alpha2  (-1 * DELTA((((CLOSE - LOW) - (HIGH - CLOSE)) / (HIGH - LOW)), 1))

    def extra_001(self, type=1):
        alpha = -((2*self.S_FWDS_ADJCLOSE-self.S_FWDS_ADJLOW-self.S_FWDS_ADJHIGH)/(self.S_FWDS_ADJHIGH-self.S_FWDS_ADJLOW)).diff()
        return alpha
        alpha[alpha >= 1.0E+38] = 1.0E+38
        alpha[alpha <= -1.0E+38] = -1.0E+38
        if type:
            return alpha[self.condperiod]
        else:
            return alpha[self.condperiod].iloc[-1, :]

    # Alpha7  ((RANK(MAX((VWAP - CLOSE), 3)) + RANK(MIN((VWAP - CLOSE), 3))) * RANK(DELTA(VOLUME, 3)))
    def extra_002(self, type=1):
        part1 = (func_rolling_max(self.S_DQ_AVGPRICE-self.S_FWDS_ADJCLOSE, 3)).rank(pct=True)
        part2 = (func_rolling_min(self.S_DQ_AVGPRICE-self.S_FWDS_ADJCLOSE, 3)).rank(pct=True)
        part3 = (self.S_DQ_VOLUME.diff(3)).rank(pct=True)
        alpha = part1+part2*part3
        return alpha
        alpha[alpha >= 1.0E+38] = 1.0E+38
        alpha[alpha <= -1.0E+38] = -1.0E+38
        if type:
            return alpha[self.condperiod]
        else:
            return alpha[self.condperiod].iloc[-1, :]

    # Alpha15  OPEN/DELAY(CLOSE,1)-1
    def extra_003(self, type=1):
        alpha = self.S_FWDS_ADJOPEN/self.S_FWDS_ADJCLOSE.shift()-1
        return alpha
        alpha[alpha >= 1.0E+38] = 1.0E+38
        alpha[alpha <= -1.0E+38] = -1.0E+38
        #result[result > 0.08] = np.nan
        if type:
            return alpha[self.condperiod]
        else:
            return alpha[self.condperiod].iloc[-1, :]

    # Alpha111  SMA(VOL*((CLOSE-LOW)-(HIGH-CLOSE))/(HIGH-LOW),11,2)-SMA(VOL*((CLOSE-LOW)-(HIGH-CLOSE))/(HIGH-LOW),4,2)
    def extra_004(self, type=1):
        data1 = self.S_DQ_VOLUME*(2*self.S_FWDS_ADJCLOSE-self.S_FWDS_ADJLOW-self.S_FWDS_ADJHIGH)/(self.S_FWDS_ADJHIGH-self.S_FWDS_ADJLOW)
        x = func_ewma(data1, alpha=2.0/9)
        y = func_ewma(data1, alpha=2.0/4)
        alpha = (x-y)
        return alpha
        alpha[alpha >= 1.0E+38] = 1.0E+38
        alpha[alpha <= -1.0E+38] = -1.0E+38
        if type:
            return alpha[self.condperiod]
        else:
            return alpha[self.condperiod].iloc[-1, :]

    # Alpha114  ((RANK(DELAY(((HIGH - LOW) / (SUM(CLOSE, 5) / 5)), 2)) * RANK(RANK(VOLUME))) / (((HIGH - LOW) /(SUM(CLOSE, 5) / 5)) / (VWAP - CLOSE)))
    def extra_005(self, type=1):
        data1 = (self.S_FWDS_ADJHIGH-self.S_FWDS_ADJLOW)/func_rolling_mean(self.S_FWDS_ADJCLOSE, 20)
        rank1 = (data1.shift(2)).rank(pct=True)
        rank2 = self.S_DQ_VOLUME.rank(pct=True).rank(pct=True)
        data2 = (data1/(self.S_DQ_AVGPRICE-self.S_FWDS_ADJCLOSE))
        alpha = (rank1*rank2)/data2
        return alpha
        alpha[alpha >= 1.0E+38] = 1.0E+38
        alpha[alpha <= -1.0E+38] = -1.0E+38
        if type:
            return alpha[self.condperiod]
        else:
            return alpha[self.condperiod].iloc[-1, :]

    def extra_006(self, type=1):
        alpha = (self.S_DQ_VOLUME/func_rolling_mean(self.S_DQ_VOLUME, 20)).rank(pct=True)
        return alpha
        alpha[alpha >= 1.0E+38] = 1.0E+38
        alpha[alpha <= -1.0E+38] = -1.0E+38
        if type:
            return alpha[self.condperiod]
        else:
            return alpha[self.condperiod].iloc[-1, :]

    # Alpha109  SMA(HIGH-LOW,5,2)/SMA(SMA(HIGH-LOW,10,2),20,2)
    def extra_007(self, type=1):
        data = self.S_FWDS_ADJHIGH-self.S_FWDS_ADJLOW
        sma1 = func_ewma(data, alpha=2.0/5)
        sma2 = func_ewma(sma1, alpha=2.0/20)
        alpha = (sma1/sma2)
        return alpha
        alpha[alpha >= 1.0E+38] = 1.0E+38
        alpha[alpha <= -1.0E+38] = -1.0E+38
        if type:
            return alpha[self.condperiod]
        else:
            return alpha[self.condperiod].iloc[-1, :]

    # Alpha110  SUM(MAX(0,HIGH-DELAY(CLOSE,1)),5)/SUM(MAX(0,DELAY(CLOSE,1)-LOW),10)*100
    def extra_008(self, type=1):
        data1 = np.maximum(self.S_FWDS_ADJHIGH-self.S_FWDS_ADJCLOSE.shift(), 0)
        data2 = np.maximum(self.S_FWDS_ADJCLOSE.shift()-self.S_FWDS_ADJLOW, 0)
        sum1 = func_rolling_sum(data1, 5)
        sum2 = func_rolling_sum(data2, 10)
        alpha = sum1/sum2*100
        return alpha
        alpha[alpha >= 1.0E+38] = 1.0E+38
        alpha[alpha <= -1.0E+38] = -1.0E+38
        if type:
            return alpha[self.condperiod]
        else:
            return alpha[self.condperiod].iloc[-1, :]

    # Alpha160  SMA((CLOSE<=DELAY(CLOSE,1)?STD(CLOSE,20):0),20,1)
    def extra_009(self, type=1):
        data1 = func_rolling_std(self.S_FWDS_ADJCLOSE, 20)
        cond = self.S_FWDS_ADJCLOSE <= self.S_FWDS_ADJCLOSE.shift(1)
        data1[~cond] = 0
        alpha = func_ewma(data1, alpha=1/5) - func_ewma(data1, alpha=1/20)
        return alpha
        alpha[alpha >= 1.0E+38] = 1.0E+38
        alpha[alpha <= -1.0E+38] = -1.0E+38
        if type:
            return alpha[self.condperiod]
        else:
            return alpha[self.condperiod].iloc[-1, :]

    # Alpha18  CLOSE/DELAY(CLOSE,5)
    def extra_010(self, type=1):
        delay5 = self.S_FWDS_ADJCLOSE.shift(5)
        alpha = self.S_FWDS_ADJCLOSE / delay5
        return alpha
        alpha[alpha >= 1.0E+38] = 1.0E+38
        alpha[alpha <= -1.0E+38] = -1.0E+38
        if type:
            return alpha[self.condperiod]
        else:
            return alpha[self.condperiod].iloc[-1, :]

    def extra_011(self, type=1):
        alpha = self.S_DQ_AMOUNT.copy()
        return alpha
        alpha[alpha >= 1.0E+38] = 1.0E+38
        alpha[alpha <= -1.0E+38] = -1.0E+38
        if type:
            return alpha[self.condperiod]
        else:
            return alpha[self.condperiod].iloc[-1, :]

    def extra_012(self, type=1):
        alpha = self.S_FWDS_ADJHIGH/self.S_FWDS_ADJOPEN
        return alpha
        alpha[alpha >= 1.0E+38] = 1.0E+38
        alpha[alpha <= -1.0E+38] = -1.0E+38
        if type:
            return alpha[self.condperiod]
        else:
            return alpha[self.condperiod].iloc[-1, :]

    def extra_013(self, type=1):
        alpha = self.S_DQ_AVGPRICE / self.S_FWDS_ADJCLOSE
        return alpha
        alpha[alpha >= 1.0E+38] = 1.0E+38
        alpha[alpha <= -1.0E+38] = -1.0E+38
        if type:
            return alpha[self.condperiod]
        else:
            return alpha[self.condperiod].iloc[-1, :]

    def extra_014(self, type=1):
        alpha = (self.S_FWDS_ADJHIGH - self.S_FWDS_ADJLOW)/self.S_FWDS_ADJCLOSE
        return alpha
        alpha[alpha >= 1.0E+38] = 1.0E+38
        alpha[alpha <= -1.0E+38] = -1.0E+38
        if type:
            return alpha[self.condperiod]
        else:
            return alpha[self.condperiod].iloc[-1, :]

# extra add features

    # Alpha1  (-1 * CORR(RANK(DELTA(LOG(VOLUME), 1)), RANK(((CLOSE - OPEN) / OPEN)), 6))
    def add_001(self, type=1):
        data1 = np.log(self.S_DQ_VOLUME + 1).diff(periods=1)
        data2 = ((self.S_FWDS_ADJCLOSE - self.S_FWDS_ADJOPEN)/self.S_FWDS_ADJOPEN)
        alpha = func_rolling_corr(data1, data2, 10)
        return alpha
        alpha[alpha >= 1.0E+38] = 1.0E+38
        alpha[alpha <= -1.0E+38] = -1.0E+38
        if type:
            return alpha[self.condperiod]
        else:
            return alpha[self.condperiod].iloc[-1, :]

    # Alpha1  (-1 * CORR(RANK(DELTA(LOG(VOLUME), 1)), RANK(((CLOSE - OPEN) / OPEN)), 6))
    def add_002(self, type=1):
        data1 = np.log(self.S_DQ_VOLUME + 1).diff(periods=1)
        data2 = ((self.S_FWDS_ADJCLOSE - self.S_FWDS_ADJOPEN)/self.S_FWDS_ADJOPEN)
        alpha = func_rolling_corr(data1, data2, 20)
        return alpha
        alpha[alpha >= 1.0E+38] = 1.0E+38
        alpha[alpha <= -1.0E+38] = -1.0E+38
        if type:
            return alpha[self.condperiod]
        else:
            return alpha[self.condperiod].iloc[-1, :]

    # Alpha22  SMEAN(((CLOSE-MEAN(CLOSE,6))/MEAN(CLOSE,6)-DELAY((CLOSE-MEAN(CLOSE,6))/MEAN(CLOSE,6),3)),12,1)
    def add_003(self, type=1):
        part0 = func_rolling_mean(self.S_FWDS_ADJCLOSE, 10)
        part1 = (self.S_FWDS_ADJCLOSE - part0)/part0
        alpha = part1-part1.shift(5)
        alpha = func_ewma(alpha, alpha=1.0/20)
        return alpha
        alpha[alpha >= 1.0E+38] = 1.0E+38
        alpha[alpha <= -1.0E+38] = -1.0E+38
        if type:
            return alpha[self.condperiod]
        else:
            return alpha[self.condperiod].iloc[-1, :]

    # Alpha29  (CLOSE-DELAY(CLOSE,6))/DELAY(CLOSE,6)*VOLUME
    def add_004(self, type=1):
        delay6 = self.S_FWDS_ADJCLOSE.shift(10)
        alpha = (self.S_FWDS_ADJCLOSE-delay6)*(self.S_DQ_VOLUME + 1)/delay6
        return alpha
        alpha[alpha >= 1.0E+38] = 1.0E+38
        alpha[alpha <= -1.0E+38] = -1.0E+38
        if type:
            return alpha[self.condperiod]
        else:
            return alpha[self.condperiod].iloc[-1, :]

    # Alpha29  (CLOSE-DELAY(CLOSE,6))/DELAY(CLOSE,6)*VOLUME
    def add_005(self, type=1):
        delay6 = self.S_FWDS_ADJCLOSE.shift(20)
        alpha = (self.S_FWDS_ADJCLOSE-delay6)*(self.S_DQ_VOLUME + 1)/delay6
        return alpha
        alpha[alpha >= 1.0E+38] = 1.0E+38
        alpha[alpha <= -1.0E+38] = -1.0E+38
        if type:
            return alpha[self.condperiod]
        else:
            return alpha[self.condperiod].iloc[-1, :]

    # Alpha37  (-1 * RANK(((SUM(OPEN, 5) * SUM(RET, 5)) - DELAY((SUM(OPEN, 5) * SUM(RET, 5)), 10))))
    def add_006(self, type=1):
        ret = self.S_FWDS_ADJCLOSE.pct_change()
        temp = func_rolling_sum(self.S_FWDS_ADJOPEN, 5)*func_rolling_sum(ret, 5)
        alpha = temp - temp.shift(10)
        return alpha
        alpha[alpha >= 1.0E+38] = 1.0E+38
        alpha[alpha <= -1.0E+38] = -1.0E+38
        if type:
            return alpha[self.condperiod]
        else:
            return alpha[self.condperiod].iloc[-1, :]

    # Alpha70  STD(AMOUNT,6)
    def add_007(self, type=1):
        alpha = func_rolling_std(np.log(self.S_DQ_AMOUNT + 1), 10)
        return alpha
        alpha[alpha >= 1.0E+38] = 1.0E+38
        alpha[alpha <= -1.0E+38] = -1.0E+38
        if type:
            return alpha[self.condperiod]
        else:
            return alpha[self.condperiod].iloc[-1, :]

    # Alpha70  STD(AMOUNT,6)
    def add_008(self, type=1):
        alpha = func_rolling_std(np.log(self.S_DQ_AMOUNT + 1), 20)
        return alpha
        alpha[alpha >= 1.0E+38] = 1.0E+38
        alpha[alpha <= -1.0E+38] = -1.0E+38
        if type:
            return alpha[self.condperiod]
        else:
            return alpha[self.condperiod].iloc[-1, :]

    # Alpha102  SMA(MAX(VOLUME-DELAY(VOLUME,1),0),6,1)/SMA(ABS(VOLUME-DELAY(VOLUME,1)),6,1)*100
    def add_009(self, type=1):
        temp1 = self.S_DQ_VOLUME-self.S_DQ_VOLUME.shift()
        part1 = np.maximum(temp1, 0)
        part1 = func_ewma(part1, alpha=1.0/12)
        temp2 = temp1.abs()
        part2 = func_ewma(temp2, alpha=1.0/12)
        alpha = part1*100/part2
        return alpha
        alpha[alpha >= 1.0E+38] = 1.0E+38
        alpha[alpha <= -1.0E+38] = -1.0E+38
        if type:
            return alpha[self.condperiod]
        else:
            return alpha[self.condperiod].iloc[-1, :]

    # Alpha144  SUMIF(ABS(CLOSE/DELAY(CLOSE,1)-1)/AMOUNT,20,CLOSE<DELAY(CLOSE,1))/COUNT(CLOSE<DELAY(CLOSE,1),20)
    def add_010(self, type=1):
        df1 = self.S_FWDS_ADJCLOSE < self.S_FWDS_ADJCLOSE.shift()
        sumif = func_rolling_sum(((abs(self.S_FWDS_ADJCLOSE / self.S_FWDS_ADJCLOSE.shift() - 1)/np.log(self.S_DQ_AMOUNT + 1))[df1].fillna(0)), 10)
        count = func_rolling_sum(df1, 10)
        alpha = (sumif / count)
        return alpha
        alpha[alpha >= 1.0E+38] = 1.0E+38
        alpha[alpha <= -1.0E+38] = -1.0E+38
        if type:
            return alpha[self.condperiod]
        else:
            return alpha[self.condperiod].iloc[-1, :]

    # Alpha170  ((((RANK((1 / CLOSE)) * VOLUME) / MEAN(VOLUME,20)) * ((HIGH * RANK((HIGH - CLOSE))) / (SUM(HIGH, 5) /5))) - RANK((VWAP - DELAY(VWAP, 5))))
    def add_011(self, type=1):
        data1 = (1/self.S_FWDS_ADJCLOSE)  # .rank(axis=1,pct=True)
        data2 = func_rolling_mean(self.S_DQ_VOLUME, 20)
        part1 = (data1*self.S_DQ_VOLUME)/data2
        data3 = (self.S_FWDS_ADJHIGH-self.S_FWDS_ADJCLOSE)  # .rank(axis=1,pct=True)
        data4 = func_rolling_mean(self.S_FWDS_ADJHIGH, 5)
        part2 = (data3*self.S_FWDS_ADJHIGH)/data4
        part3 = (self.S_DQ_AVGPRICE-self.S_DQ_AVGPRICE.shift(5))  # .rank(pct=True)
        alpha = part1 * part2 - part3
        return alpha
        alpha[alpha >= 1.0E+38] = 1.0E+38
        alpha[alpha <= -1.0E+38] = -1.0E+38
        if type:
            return alpha[self.condperiod]
        else:
            return alpha[self.condperiod].iloc[-1, :]

    # Alpha188  ((HIGH-LOW–SMA(HIGH-LOW,11,2))/SMA(HIGH-LOW,11,2))*100
    def add_012(self, type=1):
        sma = func_ewma(self.S_FWDS_ADJHIGH - self.S_FWDS_ADJLOW, alpha=1/11)
        alpha = ((self.S_FWDS_ADJHIGH - self.S_FWDS_ADJLOW - sma)/sma*100)
        return alpha
        alpha[alpha >= 1.0E+38] = 1.0E+38
        alpha[alpha <= -1.0E+38] = -1.0E+38
        if type:
            return alpha[self.condperiod]
        else:
            return alpha[self.condperiod].iloc[-1, :]

    # Alpha7  ((RANK(MAX((VWAP - CLOSE), 3)) + RANK(MIN((VWAP - CLOSE), 3))) * RANK(DELTA(VOLUME, 3)))
    def add_013(self, type=1):
        part1 = (func_rolling_max(self.S_DQ_AVGPRICE-self.S_FWDS_ADJCLOSE, 3))
        part2 = (func_rolling_min(self.S_DQ_AVGPRICE-self.S_FWDS_ADJCLOSE, 3))
        part3 = (self.S_DQ_VOLUME.diff(3))
        alpha = part1+part2*part3
        return alpha
        alpha[alpha >= 1.0E+38] = 1.0E+38
        alpha[alpha <= -1.0E+38] = -1.0E+38
        if type:
            return alpha[self.condperiod]
        else:
            return alpha[self.condperiod].iloc[-1, :]

    # Alpha111  SMA(VOL*((CLOSE-LOW)-(HIGH-CLOSE))/(HIGH-LOW),11,2)-SMA(VOL*((CLOSE-LOW)-(HIGH-CLOSE))/(HIGH-LOW),4,2)
    def add_014(self, type=1):
        data1 = self.S_DQ_VOLUME*(2*self.S_FWDS_ADJCLOSE-self.S_FWDS_ADJLOW-self.S_FWDS_ADJHIGH)/(self.S_FWDS_ADJHIGH-self.S_FWDS_ADJLOW)
        x = func_ewma(data1, alpha=1.0/10)
        y = func_ewma(data1, alpha=1.0/5)
        alpha = (x-y)
        return alpha
        alpha[alpha >= 1.0E+38] = 1.0E+38
        alpha[alpha <= -1.0E+38] = -1.0E+38
        if type:
            return alpha[self.condperiod]
        else:
            return alpha[self.condperiod].iloc[-1, :]

    # Alpha111  SMA(VOL*((CLOSE-LOW)-(HIGH-CLOSE))/(HIGH-LOW),11,2)-SMA(VOL*((CLOSE-LOW)-(HIGH-CLOSE))/(HIGH-LOW),4,2)
    def add_015(self, type=1):
        data1 = self.S_DQ_VOLUME*(2*self.S_FWDS_ADJCLOSE-self.S_FWDS_ADJLOW-self.S_FWDS_ADJHIGH)/(self.S_FWDS_ADJHIGH-self.S_FWDS_ADJLOW)
        x = func_ewma(data1, alpha=1.0/20)
        y = func_ewma(data1, alpha=1.0/10)
        alpha = (x-y)
        return alpha
        alpha[alpha >= 1.0E+38] = 1.0E+38
        alpha[alpha <= -1.0E+38] = -1.0E+38
        if type:
            return alpha[self.condperiod]
        else:
            return alpha[self.condperiod].iloc[-1, :]

    # Alpha114  ((RANK(DELAY(((HIGH - LOW) / (SUM(CLOSE, 5) / 5)), 2)) * RANK(RANK(VOLUME))) / (((HIGH - LOW) /(SUM(CLOSE, 5) / 5)) / (VWAP - CLOSE)))
    def add_016(self, type=1):
        data1 = (self.S_FWDS_ADJHIGH-self.S_FWDS_ADJLOW)/func_rolling_mean(self.S_FWDS_ADJCLOSE, 20)
        rank1 = (data1.shift(2))  # .rank(pct=True)
        rank2 = self.S_DQ_VOLUME  # .rank(pct=True).rank(pct=True)
        data2 = (data1/(self.S_DQ_AVGPRICE-self.S_FWDS_ADJCLOSE))
        alpha = (rank1*rank2)/data2
        return alpha
        alpha[alpha >= 1.0E+38] = 1.0E+38
        alpha[alpha <= -1.0E+38] = -1.0E+38
        if type:
            return alpha[self.condperiod]
        else:
            return alpha[self.condperiod].iloc[-1, :]

    # Alpha114  ((RANK(DELAY(((HIGH - LOW) / (SUM(CLOSE, 5) / 5)), 2)) * RANK(RANK(VOLUME))) / (((HIGH - LOW) /(SUM(CLOSE, 5) / 5)) / (VWAP - CLOSE)))
    def add_017(self, type=1):
        data1 = (self.S_FWDS_ADJHIGH-self.S_FWDS_ADJLOW)/func_rolling_mean(self.S_FWDS_ADJCLOSE, 10)
        rank1 = (data1.shift(3))  # .rank(pct=True)
        rank2 = self.S_DQ_VOLUME  # .rank(pct=True).rank(pct=True)
        data2 = (data1/(self.S_DQ_AVGPRICE-self.S_FWDS_ADJCLOSE))
        alpha = (rank1*rank2)/data2
        return alpha
        alpha[alpha >= 1.0E+38] = 1.0E+38
        alpha[alpha <= -1.0E+38] = -1.0E+38
        if type:
            return alpha[self.condperiod]
        else:
            return alpha[self.condperiod].iloc[-1, :]

    def add_018(self, type=1):
        alpha = (self.S_DQ_VOLUME/func_rolling_mean(self.S_DQ_VOLUME, 10))
        return alpha
        alpha[alpha >= 1.0E+38] = 1.0E+38
        alpha[alpha <= -1.0E+38] = -1.0E+38
        if type:
            return alpha[self.condperiod]
        else:
            return alpha[self.condperiod].iloc[-1, :]

    def add_019(self, type=1):
        alpha = (self.S_DQ_VOLUME/func_rolling_mean(self.S_DQ_VOLUME, 5))
        return alpha
        alpha[alpha >= 1.0E+38] = 1.0E+38
        alpha[alpha <= -1.0E+38] = -1.0E+38
        if type:
            return alpha[self.condperiod]
        else:
            return alpha[self.condperiod].iloc[-1, :]

    # Alpha110  SUM(MAX(0,HIGH-DELAY(CLOSE,1)),5)/SUM(MAX(0,DELAY(CLOSE,1)-LOW),10)*100
    def add_020(self, type=1):
        data1 = np.maximum(self.S_FWDS_ADJHIGH-self.S_FWDS_ADJCLOSE.shift(), 0)
        data2 = np.maximum(self.S_FWDS_ADJCLOSE.shift()-self.S_FWDS_ADJLOW, 0)
        sum1 = func_rolling_sum(data1, 6)
        sum2 = func_rolling_sum(data2, 20)
        alpha = sum1/sum2*100
        return alpha
        alpha[alpha >= 1.0E+38] = 1.0E+38
        alpha[alpha <= -1.0E+38] = -1.0E+38
        if type:
            return alpha[self.condperiod]
        else:
            return alpha[self.condperiod].iloc[-1, :]

    # Alpha160  SMA((CLOSE<=DELAY(CLOSE,1)?STD(CLOSE,20):0),20,1)
    def add_021(self, type=1):
        data1 = func_rolling_std(self.S_FWDS_ADJCLOSE, 10)
        cond = self.S_FWDS_ADJCLOSE <= self.S_FWDS_ADJCLOSE.shift(1)
        data1[~cond] = 0
        alpha = func_ewma(data1, alpha=1/5) - func_ewma(data1, alpha=1/20)
        return alpha
        alpha[alpha >= 1.0E+38] = 1.0E+38
        alpha[alpha <= -1.0E+38] = -1.0E+38
        if type:
            return alpha[self.condperiod]
        else:
            return alpha[self.condperiod].iloc[-1, :]

    # Alpha160  SMA((CLOSE<=DELAY(CLOSE,1)?STD(CLOSE,20):0),20,1)
    def add_022(self, type=1):
        data1 = func_rolling_std(self.S_FWDS_ADJCLOSE, 5)
        cond = self.S_FWDS_ADJCLOSE <= self.S_FWDS_ADJCLOSE.shift(1)
        data1[~cond] = 0
        alpha = func_ewma(data1, alpha=1/5) - func_ewma(data1, alpha=1/20)
        return alpha
        alpha[alpha >= 1.0E+38] = 1.0E+38
        alpha[alpha <= -1.0E+38] = -1.0E+38
        if type:
            return alpha[self.condperiod]
        else:
            return alpha[self.condperiod].iloc[-1, :]

    # Alpha18  CLOSE/DELAY(CLOSE,5)
    def add_023(self, type=1):
        delay5 = self.S_FWDS_ADJCLOSE.shift(10)
        alpha = self.S_FWDS_ADJCLOSE / delay5
        return alpha
        alpha[alpha >= 1.0E+38] = 1.0E+38
        alpha[alpha <= -1.0E+38] = -1.0E+38
        if type:
            return alpha[self.condperiod]
        else:
            return alpha[self.condperiod].iloc[-1, :]

    # Alpha18  CLOSE/DELAY(CLOSE,5)
    def add_024(self, type=1):
        delay5 = self.S_FWDS_ADJCLOSE.shift(20)
        alpha = self.S_FWDS_ADJCLOSE / delay5
        return alpha
        alpha[alpha >= 1.0E+38] = 1.0E+38
        alpha[alpha <= -1.0E+38] = -1.0E+38
        if type:
            return alpha[self.condperiod]
        else:
            return alpha[self.condperiod].iloc[-1, :]

    def add_025(self, type=1):
        part0 = (self.S_FWDS_ADJHIGH - self.S_FWDS_ADJLOW)/self.S_FWDS_ADJCLOSE
        part1 = self.S_DQ_VOLUME
        alpha = func_rolling_corr(part0, part1, 5)
        return alpha
        alpha[alpha >= 1.0E+38] = 1.0E+38
        alpha[alpha <= -1.0E+38] = -1.0E+38
        if type:
            return alpha[self.condperiod]
        else:
            return alpha[self.condperiod].iloc[-1, :]

    def add_026(self, type=1):
        part0 = (self.S_FWDS_ADJHIGH - self.S_FWDS_ADJLOW)/self.S_FWDS_ADJCLOSE
        part1 = self.S_DQ_VOLUME
        alpha = func_rolling_corr(part0, part1, 10)
        return alpha
        alpha[alpha >= 1.0E+38] = 1.0E+38
        alpha[alpha <= -1.0E+38] = -1.0E+38
        if type:
            return alpha[self.condperiod]
        else:
            return alpha[self.condperiod].iloc[-1, :]

    def add_027(self, type=1):
        part0 = (self.S_FWDS_ADJHIGH - self.S_FWDS_ADJLOW)/self.S_FWDS_ADJCLOSE
        part1 = self.S_DQ_VOLUME
        alpha = func_rolling_corr(part0, part1, 20)
        return alpha
        alpha[alpha >= 1.0E+38] = 1.0E+38
        alpha[alpha <= -1.0E+38] = -1.0E+38
        if type:
            return alpha[self.condperiod]
        else:
            return alpha[self.condperiod].iloc[-1, :]

    def add_028(self, type=1):
        alpha = self.S_FWDS_ADJCLOSE/func_ewma(self.S_FWDS_ADJCLOSE, alpha=1/5.0)
        return alpha
        alpha[alpha >= 1.0E+38] = 1.0E+38
        alpha[alpha <= -1.0E+38] = -1.0E+38
        if type:
            return alpha[self.condperiod]
        else:
            return alpha[self.condperiod].iloc[-1, :]

    def add_029(self, type=1):
        alpha = self.S_FWDS_ADJCLOSE/func_ewma(self.S_FWDS_ADJCLOSE, alpha=1/10.0)
        return alpha
        alpha[alpha >= 1.0E+38] = 1.0E+38
        alpha[alpha <= -1.0E+38] = -1.0E+38
        if type:
            return alpha[self.condperiod]
        else:
            return alpha[self.condperiod].iloc[-1, :]

    def add_030(self, type=1):
        alpha = self.S_FWDS_ADJCLOSE/func_ewma(self.S_FWDS_ADJCLOSE, alpha=1/20.0)
        return alpha
        alpha[alpha >= 1.0E+38] = 1.0E+38
        alpha[alpha <= -1.0E+38] = -1.0E+38
        if type:
            return alpha[self.condperiod]
        else:
            return alpha[self.condperiod].iloc[-1, :]

# change features

    # Alpha2  (-1 * DELTA((((CLOSE - LOW) - (HIGH - CLOSE)) / (HIGH - LOW)), 1))
    def change_001(self, type=1):
        result = (2*self.S_FWDS_ADJCLOSE-self.S_FWDS_ADJLOW-self.S_FWDS_ADJHIGH)/(self.S_FWDS_ADJHIGH-self.S_FWDS_ADJLOW)
        alpha = func_rolling_mean(result, 5)
        return alpha
        alpha[alpha >= 1.0E+38] = 1.0E+38
        alpha[alpha <= -1.0E+38] = -1.0E+38
        if type:
            return alpha[self.condperiod]
        else:
            return alpha[self.condperiod].iloc[-1, :]

    # 5日均线突破
    def change_002(self, type=1):
        part0 = self.S_FWDS_ADJCLOSE - func_ewma(self.S_FWDS_ADJCLOSE, alpha=1/5.0)
        part1 = func_rolling_std(self.S_FWDS_ADJCLOSE, 10)
        alpha = part0/part1
        return alpha
        alpha[alpha >= 1.0E+38] = 1.0E+38
        alpha[alpha <= -1.0E+38] = -1.0E+38
        if type:
            return alpha[self.condperiod]
        else:
            return alpha[self.condperiod].iloc[-1, :]

    # 10日均线突破
    def change_003(self, type=1):
        part0 = self.S_FWDS_ADJCLOSE - func_ewma(self.S_FWDS_ADJCLOSE, alpha=1/10.0)
        part1 = func_rolling_std(self.S_FWDS_ADJCLOSE, 10)
        alpha = part0/part1
        return alpha
        alpha[alpha >= 1.0E+38] = 1.0E+38
        alpha[alpha <= -1.0E+38] = -1.0E+38
        if type:
            return alpha[self.condperiod]
        else:
            return alpha[self.condperiod].iloc[-1, :]

    # 20日均线突破
    def change_004(self, type=1):
        part0 = self.S_FWDS_ADJCLOSE - func_ewma(self.S_FWDS_ADJCLOSE, alpha=1/20.0)
        part1 = func_rolling_std(self.S_FWDS_ADJCLOSE, 10)
        alpha = part0/part1
        return alpha
        alpha[alpha >= 1.0E+38] = 1.0E+38
        alpha[alpha <= -1.0E+38] = -1.0E+38
        if type:
            return alpha[self.condperiod]
        else:
            return alpha[self.condperiod].iloc[-1, :]

    # 均线缠绕

    def change_005(self, type=1):
        part0 = func_ewma(self.S_FWDS_ADJCLOSE, alpha=1/5.0)
        part1 = func_ewma(self.S_FWDS_ADJCLOSE, alpha=1/10.0)
        part2 = func_ewma(self.S_FWDS_ADJCLOSE, alpha=1/20.0)
        #alpha = pd.DataFrame(np.zeros((len(self.S_FWDS_ADJCLOSE.index), len(self.S_FWDS_ADJCLOSE.columns))), index=self.S_FWDS_ADJCLOSE.index, columns=self.S_FWDS_ADJCLOSE.columns)
        alpha = pd.Series(np.zeros(len(self.S_FWDS_ADJCLOSE.index)), index=self.S_FWDS_ADJCLOSE.index, dtype=np.float32)
        cond1 = (part0 > part1) & (part1 > part2)
        cond2 = (part0 < part1) & (part1 < part2)
        alpha[cond1] = 1
        alpha[cond2] = -1
        return alpha
        if type:
            return alpha[self.condperiod]
        else:
            return alpha[self.condperiod].iloc[-1, :]

# Original Features
# Alpha3  SUM((CLOSE=DELAY(CLOSE,1)?0:CLOSE-(CLOSE>DELAY(CLOSE,1)?MIN(LOW,DELAY(CLOSE,1)):MAX(HIGH,DELAY(CLOSE,1)))),6)
    def original_001(self, type=1):
        delay1 = self.S_FWDS_ADJCLOSE.shift()
        condition1 = (self.S_FWDS_ADJCLOSE <= delay1)
        condition2 = (self.S_FWDS_ADJCLOSE >= delay1)
        part1 = (self.S_FWDS_ADJCLOSE-np.minimum(self.S_FWDS_ADJLOW, delay1))
        part1[condition1] = 0
        part2 = (self.S_FWDS_ADJCLOSE-np.maximum(self.S_FWDS_ADJHIGH, delay1))
        part2[condition2] = 0
        alpha = func_rolling_sum(part1+part2, 6)
        return alpha
        alpha[alpha >= 1.0E+38] = 1.0E+38
        alpha[alpha <= -1.0E+38] = -1.0E+38
        if type:
            return alpha[self.condperiod]
        else:
            return alpha[self.condperiod].iloc[-1, :]

    # Alpha6  (RANK(SIGN(DELTA((((OPEN * 0.85) + (HIGH * 0.15))), 4)))* -1)
    def original_002(self, type=1):
        n = len(self.S_FWDS_ADJCLOSE)
        condition1 = ((self.S_FWDS_ADJOPEN*0.85+self.S_FWDS_ADJHIGH*0.15).diff(4) <= 0)
        condition2 = ((self.S_FWDS_ADJOPEN*0.85+self.S_FWDS_ADJHIGH*0.15).diff(4) >= 0)
        #indicator1 = pd.DataFrame(np.ones(self.S_FWDS_ADJCLOSE.shape), index=self.S_FWDS_ADJCLOSE.index, columns=self.S_FWDS_ADJCLOSE.columns)
        indicator1 = pd.Series(np.ones(n), index=self.S_FWDS_ADJCLOSE.index)
        #indicator2 = -pd.DataFrame(np.ones(self.S_FWDS_ADJCLOSE.shape), index=self.S_FWDS_ADJCLOSE.index, columns=self.S_FWDS_ADJCLOSE.columns)
        indicator2 = pd.Series(np.ones(n), index=self.S_FWDS_ADJCLOSE.index)
        indicator1[condition1] = 0
        indicator2[condition2] = 0
        alpha = indicator1+indicator2
        return alpha
        alpha[alpha >= 1.0E+38] = 1.0E+38
        alpha[alpha <= -1.0E+38] = -1.0E+38
        if type:
            return alpha[self.condperiod]
        else:
            return alpha[self.condperiod].iloc[-1, :]

    # Alpha8  RANK(DELTA(((((HIGH + LOW) / 2) * 0.2) + (VWAP * 0.8)), 4) * -1)
    def original_003(self, type=1):
        temp = (self.S_FWDS_ADJHIGH+self.S_FWDS_ADJLOW)*0.2/2+self.S_DQ_AVGPRICE*0.8
        alpha = -temp.diff(4)
        return alpha
        alpha[alpha >= 1.0E+38] = 1.0E+38
        alpha[alpha <= -1.0E+38] = -1.0E+38
        if type:
            return alpha[self.condperiod]
        else:
            return alpha[self.condperiod].iloc[-1, :]

    # Alpha9  SMA(((HIGH+LOW)/2-(DELAY(HIGH,1)+DELAY(LOW,1))/2)*(HIGH-LOW)/VOLUME,7,2)
    def original_004(self, type=1):
        temp = ((self.S_FWDS_ADJHIGH+self.S_FWDS_ADJLOW)*0.5-(self.S_FWDS_ADJHIGH.shift()+self.S_FWDS_ADJLOW.shift())*0.5)*(self.S_FWDS_ADJHIGH-self.S_FWDS_ADJLOW)/self.S_DQ_VOLUME
        alpha = func_ewma(temp, alpha=2/7.0)
        return alpha
        alpha[alpha >= 1.0E+38] = 1.0E+38
        alpha[alpha <= -1.0E+38] = -1.0E+38
        if type:
            return alpha[self.condperiod]
        else:
            return alpha[self.condperiod].iloc[-1, :]

    # Alpha10  (RANK(MAX(((RET < 0) ? STD(RET, 20) : CLOSE)^2),5))
    def original_005(self, type=1):
        ret = self.S_FWDS_ADJCLOSE.pct_change()
        part1 = func_rolling_std(ret, 20)
        part1[ret >= 0] = 0
        part2 = self.S_FWDS_ADJCLOSE.copy()
        part2[ret < 0] = 0
        alpha = func_rolling_max((part1+part2)**2, 5)
        return alpha
        alpha[alpha >= 1.0E+38] = 1.0E+38
        alpha[alpha <= -1.0E+38] = -1.0E+38
        if type:
            return alpha[self.condperiod]
        else:
            return alpha[self.condperiod].iloc[-1, :]

    # Alpha11  SUM(((CLOSE-LOW)-(HIGH-CLOSE))./(HIGH-LOW).*VOLUME,6)
    def original_006(self, type=1):
        temp = (2*self.S_FWDS_ADJCLOSE-self.S_FWDS_ADJLOW-self.S_FWDS_ADJHIGH)/(self.S_FWDS_ADJHIGH-self.S_FWDS_ADJLOW)
        alpha = func_rolling_sum(temp*self.S_DQ_VOLUME, 6)
        return alpha
        alpha[alpha >= 1.0E+38] = 1.0E+38
        alpha[alpha <= -1.0E+38] = -1.0E+38
        if type:
            return alpha[self.condperiod]
        else:
            return alpha[self.condperiod].iloc[-1, :]

    # Alpha12  (RANK((OPEN - (SUM(VWAP, 10) / 10)))) * (-1 * (RANK(ABS((CLOSE - VWAP)))))
    def original_007(self, type=1):
        vwap10 = func_rolling_mean(self.S_DQ_AVGPRICE, 10)
        temp1 = self.S_FWDS_ADJOPEN-vwap10
        temp2 = -(self.S_FWDS_ADJCLOSE-self.S_DQ_AVGPRICE).abs()
        alpha = temp1*temp2
        return alpha
        alpha[alpha >= 1.0E+38] = 1.0E+38
        alpha[alpha <= -1.0E+38] = -1.0E+38
        if type:
            return alpha[self.condperiod]
        else:
            return alpha[self.condperiod].iloc[-1, :]

    # Alpha13  (((HIGH * LOW)^0.5) - VWAP)
    def original_008(self, type=1):
        alpha = ((self.S_FWDS_ADJHIGH * self.S_FWDS_ADJLOW)**0.5)-self.S_DQ_AVGPRICE
        return alpha
        alpha[alpha >= 1.0E+38] = 1.0E+38
        alpha[alpha <= -1.0E+38] = -1.0E+38
        if type:
            return alpha[self.condperiod]
        else:
            return alpha[self.condperiod].iloc[-1, :]

    # Alpha16  (-1 * TSMAX(RANK(CORR(RANK(VOLUME), RANK(VWAP), 5)), 5))
    def original_009(self, type=1):
        temp1 = self.S_DQ_VOLUME.rank(pct=True)
        temp2 = self.S_DQ_AVGPRICE.rank(pct=True)
        alpha = func_rolling_corr(temp1, temp2, 5)
        return alpha
        alpha[alpha >= 1.0E+38] = 1.0E+38
        alpha[alpha <= -1.0E+38] = -1.0E+38
        if type:
            return alpha[self.condperiod]
        else:
            return alpha[self.condperiod].iloc[-1, :]

    # Alpha17  RANK((VWAP - MAX(VWAP, 15)))^DELTA(CLOSE, 5)
    def original_010(self, type=1):
        part1 = self.S_DQ_AVGPRICE-func_rolling_max(self.S_DQ_AVGPRICE, 15)
        part2 = self.S_FWDS_ADJCLOSE.diff(5)
        alpha = part1*part2
        return alpha
        alpha[alpha >= 1.0E+38] = 1.0E+38
        alpha[alpha <= -1.0E+38] = -1.0E+38
        if type:
            return alpha[self.condperiod]
        else:
            return alpha[self.condperiod].iloc[-1, :]

    # Alpha19  (CLOSE<DELAY(CLOSE,5)?(CLOSE-DELAY(CLOSE,5))/DELAY(CLOSE,5):(CLOSE=DELAY(CLOSE,5)?0:(CLOSE-DELAY(CLOSE,5))/CLOSE))
    def original_011(self, type=1):
        delay5 = self.S_FWDS_ADJCLOSE.shift(5)
        part1 = self.S_FWDS_ADJCLOSE/delay5 - 1
        part1[self.S_FWDS_ADJCLOSE >= delay5] = 0
        part2 = 1 - delay5/self.S_FWDS_ADJCLOSE
        part2[self.S_FWDS_ADJCLOSE <= delay5] = 0
        alpha = part1 + part2
        return alpha
        alpha[alpha >= 1.0E+38] = 1.0E+38
        alpha[alpha <= -1.0E+38] = -1.0E+38
        if type:
            return alpha[self.condperiod]
        else:
            return alpha[self.condperiod].iloc[-1, :]

    # Alpha21  REGBETA(MEAN(CLOSE,6),SEQUENCE(6))
    def original_012(self, type=1):
        part1 = func_ewma(self.S_DQ_AVGPRICE, alpha=1/10.0)
        alpha = (part1.diff(6)/6.0 + part1.diff(3)/3.0)/2
        return alpha
        alpha[alpha >= 1.0E+38] = 1.0E+38
        alpha[alpha <= -1.0E+38] = -1.0E+38
        if type:
            return alpha[self.condperiod]
        else:
            return alpha[self.condperiod].iloc[-1, :]

    # Alpha23  SMA((CLOSE>DELAY(CLOSE,1)?STD(CLOSE:20),0),20,1)/(SMA((CLOSE>DELAY(CLOSE,1)?STD(CLOSE,20):0),20,1)+SMA((CLOSE<=DELAY(CLOSE,1)?STD(CLOSE,20):0),20,1))*100
    def original_013(self, type=1):
        part1 = func_rolling_std(self.S_FWDS_ADJCLOSE, 20)
        part1[(self.S_FWDS_ADJCLOSE <= self.S_FWDS_ADJCLOSE.shift())] = 0
        part2 = func_ewma(part1, alpha=1/20.0)
        part3 = func_rolling_std(self.S_FWDS_ADJCLOSE, 20)
        part3[(self.S_FWDS_ADJCLOSE > self.S_FWDS_ADJCLOSE.shift())] = 0
        part4 = func_ewma(part3, alpha=1/20.0)
        alpha = part2/(part2 + part4)
        return alpha
        alpha[alpha >= 1.0E+38] = 1.0E+38
        alpha[alpha <= -1.0E+38] = -1.0E+38
        if type:
            return alpha[self.condperiod]
        else:
            return alpha[self.condperiod].iloc[-1, :]

    # Alpha24  SMA(CLOSE-DELAY(CLOSE,5),5,1)
    def original_014(self, type=1):
        delay5 = self.S_FWDS_ADJCLOSE.shift(5)
        result = self.S_FWDS_ADJCLOSE-delay5
        alpha = func_ewma(result, alpha=1.0/5)
        return alpha
        alpha[alpha >= 1.0E+38] = 1.0E+38
        alpha[alpha <= -1.0E+38] = -1.0E+38
        if type:
            return alpha[self.condperiod]
        else:
            return alpha[self.condperiod].iloc[-1, :]

    # Alpha26  ((((SUM(CLOSE, 7) / 7) - CLOSE)) + ((CORR(VWAP, DELAY(CLOSE, 5), 230))))
    def original_015(self, type=1):
        part1 = func_rolling_mean(self.S_FWDS_ADJCLOSE, 7)-self.S_FWDS_ADJCLOSE
        delay5 = self.S_FWDS_ADJCLOSE.shift(5)
        part2 = func_rolling_corr(self.S_DQ_AVGPRICE, delay5, 10)
        alpha = part1+part2
        return alpha
        alpha[alpha >= 1.0E+38] = 1.0E+38
        alpha[alpha <= -1.0E+38] = -1.0E+38
        if type:
            return alpha[self.condperiod]
        else:
            return alpha[self.condperiod].iloc[-1, :]

    # Alpha28  3*SMA((CLOSE-TSMIN(LOW,9))/(TSMAX(HIGH,9)-TSMIN(LOW,9))*100,3,1)-2*SMA(SMA((CLOSE-TSMIN(LOW,9))/(MAX(HIGH,9)-TSMAX(LOW,9))*100,3,1),3,1)
    def original_016(self, type=1):
        temp1 = self.S_FWDS_ADJCLOSE-func_rolling_min(self.S_FWDS_ADJLOW, 9)
        temp2 = func_rolling_max(self.S_FWDS_ADJHIGH, 9)-func_rolling_min(self.S_FWDS_ADJLOW, 9)
        part1 = 3*func_ewma(temp1*100/temp2, alpha=1.0/3)
        temp3 = func_ewma(temp1*100/temp2, alpha=1.0/3)
        part2 = 2*func_ewma(temp3, alpha=1.0/3)
        alpha = part1-part2
        return alpha
        alpha[alpha >= 1.0E+38] = 1.0E+38
        alpha[alpha <= -1.0E+38] = -1.0E+38
        if type:
            return alpha[self.condperiod]
        else:
            return alpha[self.condperiod].iloc[-1, :]

    # Alpha29  (CLOSE-DELAY(CLOSE,6))/DELAY(CLOSE,6)*VOLUME
    def original_017(self, type=1):
        delay6 = self.S_FWDS_ADJCLOSE.shift(6)
        alpha = (self.S_FWDS_ADJCLOSE-delay6)*self.S_DQ_VOLUME/delay6
        return alpha
        alpha[alpha >= 1.0E+38] = 1.0E+38
        alpha[alpha <= -1.0E+38] = -1.0E+38
        if type:
            return alpha[self.condperiod]
        else:
            return alpha[self.condperiod].iloc[-1, :]

    # Alpha32  (-1 * SUM(RANK(CORR(RANK(HIGH), RANK(VOLUME), 3)), 3))
    def original_018(self, type=1):
        temp1 = self.S_FWDS_ADJHIGH
        temp2 = self.S_DQ_VOLUME
        alpha = -func_rolling_sum(func_rolling_corr(temp1, temp2, 4).rank(pct=True), 3)
        return alpha
        alpha[alpha >= 1.0E+38] = 1.0E+38
        alpha[alpha <= -1.0E+38] = -1.0E+38
        if type:
            return alpha[self.condperiod]
        else:
            return alpha[self.condperiod].iloc[-1, :]

    # Alpha33  ((((-1 * TSMIN(LOW, 5)) + DELAY(TSMIN(LOW, 5), 5)) * RANK(((SUM(RET, 240) - SUM(RET, 20)) / 220))) *TSRANK(VOLUME, 5))
    def original_019(self, type=1):
        ret = self.S_FWDS_ADJCLOSE.pct_change()
        temp1 = func_rolling_min(self.S_FWDS_ADJLOW, 5)
        part1 = temp1.shift(5)-temp1
        part2 = (func_rolling_sum(ret, 20)-func_rolling_sum(ret, 10))/10
        part3 = self.S_DQ_VOLUME.rolling(5).apply(ts_rank)
        alpha = part1 * part2 * part3
        return alpha
        alpha[alpha >= 1.0E+38] = 1.0E+38
        alpha[alpha <= -1.0E+38] = -1.0E+38
        if type:
            return alpha[self.condperiod]
        else:
            return alpha[self.condperiod].iloc[-1, :]

    # Alpha36  RANK(SUM(CORR(RANK(VOLUME), RANK(VWAP)), 6), 2)
    def original_020(self, type=1):
        temp1 = self.S_DQ_VOLUME.rank(pct=True)
        temp2 = self.S_DQ_AVGPRICE.rank(pct=True)
        part1 = func_rolling_corr(temp1, temp2, 6)
        alpha = func_rolling_sum(part1, 2)
        return alpha
        alpha[alpha >= 1.0E+38] = 1.0E+38
        alpha[alpha <= -1.0E+38] = -1.0E+38
        if type:
            return alpha[self.condperiod]
        else:
            return alpha[self.condperiod].iloc[-1, :]

    # Alpha38  (((SUM(HIGH, 20) / 20) < HIGH) ? (-1 * DELTA(HIGH, 2)) : 0)
    def original_021(self, type=1):
        alpha = -self.S_FWDS_ADJHIGH.diff(2)
        alpha[func_rolling_mean(self.S_FWDS_ADJHIGH, 20) >= self.S_FWDS_ADJHIGH] = 0
        return alpha
        alpha[alpha >= 1.0E+38] = 1.0E+38
        alpha[alpha <= -1.0E+38] = -1.0E+38
        if type:
            return alpha[self.condperiod]
        else:
            return alpha[self.condperiod].iloc[-1, :]

    # Alpha40  SUM((CLOSE>DELAY(CLOSE,1)?VOLUME:0),26)/SUM((CLOSE<=DELAY(CLOSE,1)?VOLUME:0),26)*100
    def original_022(self, type=1):
        part1 = self.S_DQ_VOLUME.copy()
        part1[self.S_FWDS_ADJCLOSE <= self.S_FWDS_ADJCLOSE.shift()] = 0
        sum1 = func_rolling_sum(part1, 20)
        part2 = self.S_DQ_VOLUME.copy()
        part2[self.S_FWDS_ADJCLOSE > self.S_FWDS_ADJCLOSE.shift()] = 0
        sum2 = func_rolling_sum(part2, 20)
        alpha = sum1/sum2
        return alpha
        alpha[alpha >= 1.0E+38] = 1.0E+38
        alpha[alpha <= -1.0E+38] = -1.0E+38
        if type:
            return alpha[self.condperiod]
        else:
            return alpha[self.condperiod].iloc[-1, :]

    # Alpha41  (RANK(MAX(DELTA((VWAP), 3), 5))* -1)
    def original_023(self, type=1):
        delta_avg = self.S_DQ_AVGPRICE.diff(3)
        alpha = func_rolling_max(delta_avg, 5)
        return alpha
        alpha[alpha >= 1.0E+38] = 1.0E+38
        alpha[alpha <= -1.0E+38] = -1.0E+38
        if type:
            return alpha[self.condperiod]
        else:
            return alpha[self.condperiod].iloc[-1, :]

    # Alpha42  ((-1 * RANK(STD(HIGH, 10))) * CORR(HIGH, VOLUME, 10))
    def original_024(self, type=1):
        part1 = func_rolling_std(self.S_FWDS_ADJHIGH, 10)
        part2 = func_rolling_corr(self.S_FWDS_ADJHIGH, self.S_DQ_VOLUME, 10)
        alpha = -part1*part2
        return alpha
        alpha[alpha >= 1.0E+38] = 1.0E+38
        alpha[alpha <= -1.0E+38] = -1.0E+38
        if type:
            return alpha[self.condperiod]
        else:
            return alpha[self.condperiod].iloc[-1, :]

    # Alpha43  SUM((CLOSE>DELAY(CLOSE,1)?VOLUME:(CLOSE<DELAY(CLOSE,1)?-VOLUME:0)),6)
    def original_025(self, type=1):
        delay1 = self.S_FWDS_ADJCLOSE.shift()
        part1 = self.S_DQ_VOLUME.copy()
        part1[self.S_FWDS_ADJCLOSE <= delay1] = 0
        part2 = -self.S_DQ_VOLUME.copy()
        part2[self.S_FWDS_ADJCLOSE >= delay1] = 0
        alpha = func_rolling_sum(part1 + part2, 6)
        return alpha
        alpha[alpha >= 1.0E+38] = 1.0E+38
        alpha[alpha <= -1.0E+38] = -1.0E+38
        if type:
            return alpha[self.condperiod]
        else:
            return alpha[self.condperiod].iloc[-1, :]

    # Alpha45  (RANK(DELTA((((CLOSE * 0.6) + (OPEN *0.4))), 1)) * RANK(CORR(VWAP, MEAN(VOLUME,150), 15)))
    def original_026(self, type=1):
        part1 = (self.S_FWDS_ADJCLOSE*0.6+self.S_FWDS_ADJOPEN*0.4).diff()
        temp2 = func_rolling_mean(self.S_DQ_VOLUME, 20)
        part2 = func_rolling_corr(self.S_DQ_AVGPRICE, temp2, 10)
        alpha = part1*part2
        return alpha
        alpha[alpha >= 1.0E+38] = 1.0E+38
        alpha[alpha <= -1.0E+38] = -1.0E+38
        if type:
            return alpha[self.condperiod]
        else:
            return alpha[self.condperiod].iloc[-1, :]

    # Alpha47  SMA((TSMAX(HIGH,6)-CLOSE)/(TSMAX(HIGH,6)-TSMIN(LOW,6))*100,9,1)

    def original_027(self, type=1):
        part1 = func_rolling_max(self.S_FWDS_ADJHIGH, 6)-self.S_FWDS_ADJCLOSE
        part2 = func_rolling_max(self.S_FWDS_ADJHIGH, 6) - func_rolling_min(self.S_FWDS_ADJLOW, 6)
        alpha = func_ewma(100*part1/part2, alpha=1.0/9)
        return alpha
        alpha[alpha >= 1.0E+38] = 1.0E+38
        alpha[alpha <= -1.0E+38] = -1.0E+38
        if type:
            return alpha[self.condperiod]
        else:
            return alpha[self.condperiod].iloc[-1, :]

    # Alpha48  (-1*((RANK(((SIGN((CLOSE - DELAY(CLOSE, 1))) + SIGN((DELAY(CLOSE, 1) - DELAY(CLOSE, 2)))) +SIGN((DELAY(CLOSE, 2) - DELAY(CLOSE, 3)))))) * SUM(VOLUME, 5)) / SUM(VOLUME, 20))
    def original_028(self, type=1):
        condition1 = (self.S_FWDS_ADJCLOSE > self.S_FWDS_ADJCLOSE.shift())
        condition2 = (self.S_FWDS_ADJCLOSE < self.S_FWDS_ADJCLOSE.shift())
        condition3 = (self.S_FWDS_ADJCLOSE.shift() > self.S_FWDS_ADJCLOSE.shift(2))
        condition4 = (self.S_FWDS_ADJCLOSE.shift() < self.S_FWDS_ADJCLOSE.shift(2))
        condition5 = (self.S_FWDS_ADJCLOSE.shift(2) > self.S_FWDS_ADJCLOSE.shift(3))
        condition6 = (self.S_FWDS_ADJCLOSE.shift(2) < self.S_FWDS_ADJCLOSE.shift(3))
        indicator1 = pd.DataFrame(np.zeros(self.S_FWDS_ADJCLOSE.shape), index=self.S_FWDS_ADJCLOSE.index, columns=self.S_FWDS_ADJCLOSE.columns)
        indicator1[condition1] = 1
        indicator1[condition2] = -1
        indicator2 = pd.DataFrame(np.zeros(self.S_FWDS_ADJCLOSE.shape), index=self.S_FWDS_ADJCLOSE.index, columns=self.S_FWDS_ADJCLOSE.columns)
        indicator2[condition3] = 1
        indicator2[condition4] = -1
        indicator3 = pd.DataFrame(np.zeros(self.S_FWDS_ADJCLOSE.shape), index=self.S_FWDS_ADJCLOSE.index, columns=self.S_FWDS_ADJCLOSE.columns)
        indicator3[condition5] = 1
        indicator3[condition6] = -1
        alpha = -(indicator1 + indicator2 + indicator3) * func_rolling_sum(self.S_DQ_VOLUME, 5)/func_rolling_sum(self.S_DQ_VOLUME, 20)
        return alpha
        alpha[alpha >= 1.0E+38] = 1.0E+38
        alpha[alpha <= -1.0E+38] = -1.0E+38
        if type:
            return alpha[self.condperiod]
        else:
            return alpha[self.condperiod].iloc[-1, :]


# old features
# Alpha49  SUM(((HIGH+LOW)>=(DELAY(HIGH,1)+DELAY(LOW,1))?0:MAX(ABS(HIGH-DELAY(HIGH,1)),ABS(LOW-DELAY(LOW,1)))),12)/(SUM(((HIGH+LOW)>=(DELAY(HIGH,1)+DELAY(LOW,1))?0:MAX(ABS(HIGH-DELAY(HIGH,1)),ABS(LOW-DELAY(LOW,1)))),12)+SUM(((HIGH+LOW)<=(DELAY(HIGH,1)+DELAY(LOW,1))?0:MAX(ABS(HIGH-DELAY(HIGH,1)),ABS(LOW-DELAY(LOW,1)))),12))

    def old_001(self, type=1):
        part1 = (self.S_FWDS_ADJHIGH - self.S_FWDS_ADJHIGH.shift()).abs()
        part2 = (self.S_FWDS_ADJLOW - self.S_FWDS_ADJLOW.shift()).abs()
        cond1 = part2 >= part1
        part1[cond1] = 0
        part2[~cond1] = 0
        conv1 = part1 + part2
        part3 = self.S_FWDS_ADJHIGH + self.S_FWDS_ADJLOW
        conv1[part3 >= part3.shift()] = 0
        det1 = func_rolling_sum(conv1, 12)
        conv2 = part1 + part2
        conv2[part3 <= part3.shift()] = 0
        det2 = func_rolling_sum(conv2, 12)
        alpha = det1/(det1 + det2)
        return alpha
        alpha[alpha >= 1.0E+38] = 1.0E+38
        alpha[alpha <= -1.0E+38] = -1.0E+38
        if type:
            return alpha[self.condperiod]
        else:
            return alpha[self.condperiod].iloc[-1, :]

    # Alpha52  SUM(MAX(0,HIGH-DELAY((HIGH+LOW+CLOSE)/3,1)),26)/SUM(MAX(0,DELAY((HIGH+LOW+CLOSE)/3,1)-L),26)*100
    def old_002(self, type=1):
        delay = ((self.S_FWDS_ADJHIGH+self.S_FWDS_ADJLOW+self.S_FWDS_ADJCLOSE)/3).shift()
        part1 = (np.maximum(self.S_FWDS_ADJHIGH-delay, 0))
        part2 = (np.maximum(delay-self.S_FWDS_ADJLOW, 0))
        alpha = func_rolling_sum(part1, 10)/func_rolling_sum(part2, 10)
        return alpha
        alpha[alpha >= 1.0E+38] = 1.0E+38
        alpha[alpha <= -1.0E+38] = -1.0E+38
        if type:
            return alpha[self.condperiod]
        else:
            return alpha[self.condperiod].iloc[-1, :]

    # Alpha53  COUNT(CLOSE>DELAY(CLOSE,1),12)/12*100
    def old_003(self, type=1):
        delay = self.S_FWDS_ADJCLOSE.shift()
        condition = self.S_FWDS_ADJCLOSE > delay
        alpha = func_rolling_sum(condition, 12)/12.0
        return alpha
        alpha[alpha >= 1.0E+38] = 1.0E+38
        alpha[alpha <= -1.0E+38] = -1.0E+38
        if type:
            return alpha[self.condperiod]
        else:
            return alpha[self.condperiod].iloc[-1, :]

    # Alpha53  COUNT(CLOSE>DELAY(CLOSE,1),12)/12*100
    def old_004(self, type=1):
        condition = self.S_FWDS_ADJCLOSE > self.S_FWDS_ADJOPEN
        alpha = func_rolling_sum(condition, 12)/12.0
        return alpha
        alpha[alpha >= 1.0E+38] = 1.0E+38
        alpha[alpha <= -1.0E+38] = -1.0E+38
        if type:
            return alpha[self.condperiod]
        else:
            return alpha[self.condperiod].iloc[-1, :]

    # Alpha54  (-1 * RANK((STD(ABS(CLOSE - OPEN)) + (CLOSE - OPEN)) + CORR(CLOSE, OPEN,10)))
    def old_005(self, type=1):
        part1 = func_rolling_std((self.S_FWDS_ADJCLOSE-self.S_FWDS_ADJOPEN).abs(), 10) + (self.S_FWDS_ADJCLOSE-self.S_FWDS_ADJOPEN)
        part2 = func_rolling_corr(self.S_FWDS_ADJCLOSE, self.S_FWDS_ADJOPEN, 10)
        alpha = -(part1 + part2)
        return alpha
        alpha[alpha >= 1.0E+38] = 1.0E+38
        alpha[alpha <= -1.0E+38] = -1.0E+38
        if type:
            return alpha[self.condperiod]
        else:
            return alpha[self.condperiod].iloc[-1, :]

    # Alpha56  (RANK((OPEN  -  TSMIN(OPEN,  12)))  <  RANK((RANK(CORR(SUM(((HIGH  +  LOW)  /  2),  19),SUM(MEAN(VOLUME,40), 19), 13))^5)))
    def old_006(self, type=1):
        part1 = (self.S_FWDS_ADJOPEN-func_rolling_min(self.S_FWDS_ADJOPEN, 6)).rank(pct=True)
        temp1 = func_rolling_sum((self.S_FWDS_ADJHIGH+self.S_FWDS_ADJLOW)/2, 10)
        temp2 = func_rolling_sum(func_rolling_mean(self.S_DQ_VOLUME, 20), 10)
        part2 = func_rolling_corr(temp1, temp2, 7).rank(pct=True)
        alpha = part1 < part2
        return alpha
        alpha[alpha >= 1.0E+38] = 1.0E+38
        alpha[alpha <= -1.0E+38] = -1.0E+38
        if type:
            return alpha[self.condperiod]
        else:
            return alpha[self.condperiod].iloc[-1, :]

    # Alpha57  SMA((CLOSE-TSMIN(LOW,9))/(TSMAX(HIGH,9)-TSMIN(LOW,9))*100,3,1)
    def old_007(self, type=1):
        part1 = self.S_FWDS_ADJCLOSE-func_rolling_min(self.S_FWDS_ADJLOW, 9)
        part2 = func_rolling_max(self.S_FWDS_ADJHIGH, 9)-func_rolling_min(self.S_FWDS_ADJLOW, 9)
        alpha = func_ewma(100*part1/part2, alpha=1.0/3)
        return alpha
        alpha[alpha >= 1.0E+38] = 1.0E+38
        alpha[alpha <= -1.0E+38] = -1.0E+38
        if type:
            return alpha[self.condperiod]
        else:
            return alpha[self.condperiod].iloc[-1, :]

    # Alpha58  COUNT(CLOSE>DELAY(CLOSE,1),20)/20*100
    # Alpha53  COUNT(CLOSE>DELAY(CLOSE,1),12)/12*100
    def old_008(self, type=1):
        delay = self.S_FWDS_ADJCLOSE.shift()
        condition = self.S_FWDS_ADJCLOSE > delay
        alpha = func_rolling_sum(condition, 20)/20.0
        return alpha
        alpha[alpha >= 1.0E+38] = 1.0E+38
        alpha[alpha <= -1.0E+38] = -1.0E+38
        if type:
            return alpha[self.condperiod]
        else:
            return alpha[self.condperiod].iloc[-1, :]

    # Alpha59  SUM((CLOSE=DELAY(CLOSE,1)?0:CLOSE-(CLOSE>DELAY(CLOSE,1)?MIN(LOW,DELAY(CLOSE,1)):MAX(HIGH,DELAY(CLOSE,1)))),20)
    def old_009(self, type=1):
        delay = self.S_FWDS_ADJCLOSE.shift()
        condition1 = (self.S_FWDS_ADJCLOSE > delay)
        condition2 = (self.S_FWDS_ADJCLOSE < delay)
        part1 = self.S_FWDS_ADJCLOSE - np.minimum(self.S_FWDS_ADJLOW, delay)
        part1[self.S_FWDS_ADJCLOSE <= delay] = 0
        part2 = self.S_FWDS_ADJCLOSE - np.maximum(self.S_FWDS_ADJHIGH, delay)
        part2[self.S_FWDS_ADJCLOSE >= delay] = 0
        alpha = func_rolling_sum(part1 + part2, 20)
        return alpha
        alpha[alpha >= 1.0E+38] = 1.0E+38
        alpha[alpha <= -1.0E+38] = -1.0E+38
        if type:
            return alpha[self.condperiod]
        else:
            return alpha[self.condperiod].iloc[-1, :]

    # Alpha62  (-1 * CORR(HIGH, RANK(VOLUME), 5))

    def old_010(self, type=1):
        volume_rank = self.S_DQ_VOLUME.rank(pct=True)
        alpha = -func_rolling_corr(self.S_FWDS_ADJHIGH, volume_rank, 5)
        return alpha
        alpha[alpha >= 1.0E+38] = 1.0E+38
        alpha[alpha <= -1.0E+38] = -1.0E+38
        if type:
            return alpha[self.condperiod]
        else:
            return alpha[self.condperiod].iloc[-1, :]

    # Alpha63  SMA(MAX(CLOSE-DELAY(CLOSE,1),0),6,1)/SMA(ABS(CLOSE-DELAY(CLOSE,1)),6,1)*100
    def old_011(self, type=1):
        part1 = np.maximum(self.S_FWDS_ADJCLOSE-self.S_FWDS_ADJCLOSE.shift(), 0)
        part1 = func_ewma(part1, alpha=1.0/6)
        part2 = (self.S_FWDS_ADJCLOSE-self.S_FWDS_ADJCLOSE.shift()).abs()
        part2 = func_ewma(part2, alpha=1.0/6)
        alpha = part1*100/part2
        return alpha
        alpha[alpha >= 1.0E+38] = 1.0E+38
        alpha[alpha <= -1.0E+38] = -1.0E+38
        if type:
            return alpha[self.condperiod]
        else:
            return alpha[self.condperiod].iloc[-1, :]

    # Alpha67  SMA(MAX(CLOSE-DELAY(CLOSE,1),0),24,1)/SMA(ABS(CLOSE-DELAY(CLOSE,1)),24,1)*100
    def old_012(self, type=1):
        temp1 = self.S_FWDS_ADJCLOSE-self.S_FWDS_ADJCLOSE.shift()
        part1 = np.maximum(temp1, 0)
        part1 = func_ewma(part1, alpha=1.0/24)
        temp2 = temp1.abs()
        part2 = func_ewma(temp2, alpha=1.0/24)
        alpha = part1*100/part2
        return alpha
        alpha[alpha >= 1.0E+38] = 1.0E+38
        alpha[alpha <= -1.0E+38] = -1.0E+38
        if type:
            return alpha[self.condperiod]
        else:
            return alpha[self.condperiod].iloc[-1, :]

    # Alpha68  SMA(((HIGH+LOW)/2-(DELAY(HIGH,1)+DELAY(LOW,1))/2)*(HIGH-LOW)/VOLUME,15,2)
    def old_013(self, type=1):
        result = (self.S_FWDS_ADJHIGH.diff()+self.S_FWDS_ADJLOW.diff())/2 * (self.S_FWDS_ADJHIGH-self.S_FWDS_ADJLOW)/self.S_DQ_VOLUME
        alpha = func_ewma(result, alpha=2.0/15)
        return alpha
        alpha[alpha >= 1.0E+38] = 1.0E+38
        alpha[alpha <= -1.0E+38] = -1.0E+38
        if type:
            return alpha[self.condperiod]
        else:
            return alpha[self.condperiod].iloc[-1, :]

    # Alpha71  (CLOSE-MEAN(CLOSE,24))/MEAN(CLOSE,24)*100
    def old_014(self, type=1):
        alpha = self.S_FWDS_ADJCLOSE/func_rolling_mean(self.S_FWDS_ADJCLOSE, 24)
        return alpha
        alpha[alpha >= 1.0E+38] = 1.0E+38
        alpha[alpha <= -1.0E+38] = -1.0E+38
        if type:
            return alpha[self.condperiod]
        else:
            return alpha[self.condperiod].iloc[-1, :]

    # Alpha72  SMA((TSMAX(HIGH,6)-CLOSE)/(TSMAX(HIGH,6)-TSMIN(LOW,6))*100,15,1)
    def old_015(self, type=1):
        data1 = func_rolling_max(self.S_FWDS_ADJHIGH, 6) - self.S_FWDS_ADJCLOSE
        data2 = func_rolling_max(self.S_FWDS_ADJHIGH, 6) - func_rolling_min(self.S_FWDS_ADJLOW, 6)
        alpha = func_ewma(data1 / data2 * 100, alpha=1/15.0)
        return alpha
        alpha[alpha >= 1.0E+38] = 1.0E+38
        alpha[alpha <= -1.0E+38] = -1.0E+38
        if type:
            return alpha[self.condperiod]
        else:
            return alpha[self.condperiod].iloc[-1, :]

    # Alpha74  (RANK(CORR(SUM(((LOW  *  0.35)  +  (VWAP  *  0.65)),  20),  SUM(MEAN(VOLUME,40),  20),  7))  +RANK(CORR(RANK(VWAP), RANK(VOLUME), 6)))
    def old_016(self, type=1):
        data1 = func_rolling_sum((self.S_FWDS_ADJLOW * 0.35 + self.S_DQ_AVGPRICE * 0.65), 10)
        data2 = func_rolling_sum(func_rolling_mean(self.S_DQ_VOLUME, 20), 10)
        rank1 = func_rolling_corr(data1, data2, 7).rank(pct=True)
        rank2 = func_rolling_corr(self.S_DQ_AVGPRICE, self.S_DQ_VOLUME, 6).rank(pct=True)
        alpha = (rank1 + rank2)
        return alpha
        alpha[alpha >= 1.0E+38] = 1.0E+38
        alpha[alpha <= -1.0E+38] = -1.0E+38
        if type:
            return alpha[self.condperiod]
        else:
            return alpha[self.condperiod].iloc[-1, :]

    # Alpha76  STD(ABS((CLOSE/DELAY(CLOSE,1)-1))/VOLUME,20)/MEAN(ABS((CLOSE/DELAY(CLOSE,1)-1))/VOLUME,20)
    def old_017(self, type=1):
        data1 = func_rolling_std((self.S_FWDS_ADJCLOSE/self.S_FWDS_ADJCLOSE.shift()-1).abs()/self.S_DQ_VOLUME, 20)
        data2 = func_rolling_mean((self.S_FWDS_ADJCLOSE/self.S_FWDS_ADJCLOSE.shift()-1).abs()/self.S_DQ_VOLUME, 20)
        alpha = (data1 / data2)
        return alpha
        alpha[alpha >= 1.0E+38] = 1.0E+38
        alpha[alpha <= -1.0E+38] = -1.0E+38
        if type:
            return alpha[self.condperiod]
        else:
            return alpha[self.condperiod].iloc[-1, :]

    # Alpha78  ((HIGH+LOW+CLOSE)/3-MA((HIGH+LOW+CLOSE)/3,12))/(0.015*MEAN(ABS(CLOSE-MEAN((HIGH+LOW+CLOSE)/3,12)),12))

    def old_018(self, type=1):
        data1 = (self.S_FWDS_ADJHIGH+self.S_FWDS_ADJLOW+self.S_FWDS_ADJCLOSE)/3-func_rolling_mean((self.S_FWDS_ADJHIGH+self.S_FWDS_ADJLOW+self.S_FWDS_ADJCLOSE)/3, 12)
        data2 = (self.S_FWDS_ADJCLOSE - func_rolling_mean((self.S_FWDS_ADJHIGH+self.S_FWDS_ADJLOW+self.S_FWDS_ADJCLOSE)/3, 12)).abs()
        data3 = func_rolling_mean(data2, 12) * 0.015
        alpha = data1 / data3
        return alpha
        alpha[alpha >= 1.0E+38] = 1.0E+38
        alpha[alpha <= -1.0E+38] = -1.0E+38
        if type:
            return alpha[self.condperiod]
        else:
            return alpha[self.condperiod].iloc[-1, :]

    # Alpha79  SMA(MAX(CLOSE-DELAY(CLOSE,1),0),12,1)/SMA(ABS(CLOSE-DELAY(CLOSE,1)),12,1)*100
    def old_019(self, type=1):
        data1 = func_ewma(np.maximum(self.S_FWDS_ADJCLOSE - self.S_FWDS_ADJCLOSE.shift(), 0), alpha=1/12)
        data2 = func_ewma((self.S_FWDS_ADJCLOSE - self.S_FWDS_ADJCLOSE.shift()).abs(), alpha=1/12)
        alpha = data1 / data2
        return alpha
        alpha[alpha >= 1.0E+38] = 1.0E+38
        alpha[alpha <= -1.0E+38] = -1.0E+38
        if type:
            return alpha[self.condperiod]
        else:
            return alpha[self.condperiod].iloc[-1, :]

    # Alpha80  (VOLUME-DELAY(VOLUME,5))/DELAY(VOLUME,5)*100
    def old_020(self, type=1):
        alpha = self.S_DQ_VOLUME/self.S_DQ_VOLUME.shift(5)
        return alpha
        alpha[alpha >= 1.0E+38] = 1.0E+38
        alpha[alpha <= -1.0E+38] = -1.0E+38
        if type:
            return alpha[self.condperiod]
        else:
            return alpha[self.condperiod].iloc[-1, :]

    # Alpha81  SMA(VOLUME,21,2)
    def old_021(self, type=1):
        alpha = func_ewma(self.S_DQ_VOLUME.diff(), alpha=2.0/21)
        return alpha
        alpha[alpha >= 1.0E+38] = 1.0E+38
        alpha[alpha <= -1.0E+38] = -1.0E+38
        if type:
            return alpha[self.condperiod]
        else:
            return alpha[self.condperiod].iloc[-1, :]

    # Alpha82  SMA((TSMAX(HIGH,6)-CLOSE)/(TSMAX(HIGH,6)-TSMIN(LOW,6))*100,20,1)
    def old_022(self, type=1):
        part1 = func_rolling_max(self.S_FWDS_ADJHIGH, 6)-self.S_FWDS_ADJCLOSE
        part2 = func_rolling_max(self.S_FWDS_ADJHIGH, 6)-func_rolling_min(self.S_FWDS_ADJLOW, 6)
        alpha = func_ewma(100*part1/part2, alpha=1.0/20)
        return alpha
        alpha[alpha >= 1.0E+38] = 1.0E+38
        alpha[alpha <= -1.0E+38] = -1.0E+38
        if type:
            return alpha[self.condperiod]
        else:
            return alpha[self.condperiod].iloc[-1, :]

    # Alpha85  (TSRANK((VOLUME / MEAN(VOLUME,20)), 20) * TSRANK((-1 * DELTA(CLOSE, 7)), 8))
    def old_023(self, type=1):
        temp1 = self.S_DQ_VOLUME/func_rolling_mean(self.S_DQ_VOLUME, 20)
        part1 = temp1.rolling(20).apply(ts_rank)
        delta = -self.S_FWDS_ADJCLOSE.diff(7)
        part2 = delta.rolling(8).apply(ts_rank)
        alpha = part1*part2
        return alpha
        alpha[alpha >= 1.0E+38] = 1.0E+38
        alpha[alpha <= -1.0E+38] = -1.0E+38
        if type:
            return alpha[self.condperiod]
        else:
            return alpha[self.condperiod].iloc[-1, :]

    # Alpha89  2*(SMA(CLOSE,13,2)-SMA(CLOSE,27,2)-SMA(SMA(CLOSE,13,2)-SMA(CLOSE,27,2),10,2))
    def old_024(self, type=1):
        data1 = func_ewma(self.S_FWDS_ADJCLOSE, alpha=2.0/13)
        data2 = func_ewma(self.S_FWDS_ADJCLOSE, alpha=2.0/27)
        data3 = func_ewma(data1-data2, alpha=2.0/10)
        alpha = data1-data2-data3
        return alpha
        alpha[alpha >= 1.0E+38] = 1.0E+38
        alpha[alpha <= -1.0E+38] = -1.0E+38
        if type:
            return alpha[self.condperiod]
        else:
            return alpha[self.condperiod].iloc[-1, :]

    # Alpha90  ( RANK(CORR(RANK(VWAP), RANK(VOLUME), 5)) * -1)
    def old_025(self, type=1):
        alpha = -func_rolling_corr(self.S_DQ_AVGPRICE, self.S_DQ_VOLUME, 5)
        return alpha
        alpha[alpha >= 1.0E+38] = 1.0E+38
        alpha[alpha <= -1.0E+38] = -1.0E+38
        if type:
            return alpha[self.condperiod]
        else:
            return alpha[self.condperiod].iloc[-1, :]

    # Alpha91  ((RANK((CLOSE - MAX(CLOSE, 5)))*RANK(CORR((MEAN(VOLUME,40)), LOW, 5))) * -1)
    def old_026(self, type=1):
        data1 = (self.S_FWDS_ADJCLOSE - func_rolling_max(self.S_FWDS_ADJCLOSE, 5))
        data2 = func_rolling_corr(func_rolling_mean(self.S_DQ_VOLUME, 20), self.S_FWDS_ADJLOW, 5)
        alpha = -data1*data2
        return alpha
        alpha[alpha >= 1.0E+38] = 1.0E+38
        alpha[alpha <= -1.0E+38] = -1.0E+38
        if type:
            return alpha[self.condperiod]
        else:
            return alpha[self.condperiod].iloc[-1, :]

# old features again
    # Alpha93  SUM((OPEN>=DELAY(OPEN,1)?0:MAX((OPEN-LOW),(OPEN-DELAY(OPEN,1)))),20)
    def old_027(self, type=1):
        condition = (self.S_FWDS_ADJOPEN >= self.S_FWDS_ADJOPEN.shift())
        temp = np.maximum(self.S_FWDS_ADJOPEN-self.S_FWDS_ADJLOW, self.S_FWDS_ADJOPEN-self.S_FWDS_ADJOPEN.shift())
        temp[condition] = 0
        alpha = func_rolling_sum(temp, 20)
        return alpha
        alpha[alpha >= 1.0E+38] = 1.0E+38
        alpha[alpha <= -1.0E+38] = -1.0E+38
        if type:
            return alpha[self.condperiod]
        else:
            return alpha[self.condperiod].iloc[-1, :]

    # Alpha94  SUM((CLOSE>DELAY(CLOSE,1)?VOLUME:(CLOSE<DELAY(CLOSE,1)?-VOLUME:0)),30)
    def old_028(self, type=1):
        delay1 = self.S_FWDS_ADJCLOSE.shift()
        part1 = self.S_DQ_VOLUME.copy()
        part1[self.S_FWDS_ADJCLOSE <= delay1] = 0
        part2 = -self.S_DQ_VOLUME.copy()
        part2[self.S_FWDS_ADJCLOSE >= delay1] = 0
        alpha = func_rolling_sum(part1 + part2, 20)
        return alpha
        alpha[alpha >= 1.0E+38] = 1.0E+38
        alpha[alpha <= -1.0E+38] = -1.0E+38
        if type:
            return alpha[self.condperiod]
        else:
            return alpha[self.condperiod].iloc[-1, :]

    # Alpha95  STD(AMOUNT,20)
    def old_029(self, type=1):
        alpha = func_rolling_std(self.S_DQ_AMOUNT, 20)
        return alpha
        alpha[alpha >= 1.0E+38] = 1.0E+38
        alpha[alpha <= -1.0E+38] = -1.0E+38
        if type:
            return alpha[self.condperiod]
        else:
            return alpha[self.condperiod].iloc[-1, :]

    # Alpha96  SMA(SMA((CLOSE-TSMIN(LOW,9))/(TSMAX(HIGH,9)-TSMIN(LOW,9))*100,3,1),3,1)
    def old_030(self, type=1):
        sma1 = func_ewma(100*(self.S_FWDS_ADJCLOSE-func_rolling_min(self.S_FWDS_ADJLOW, 9))/(func_rolling_max(self.S_FWDS_ADJHIGH, 9)-func_rolling_min(self.S_FWDS_ADJLOW, 9)), alpha=1.0/3)
        alpha = func_ewma(sma1, alpha=1.0/3)
        return alpha
        alpha[alpha >= 1.0E+38] = 1.0E+38
        alpha[alpha <= -1.0E+38] = -1.0E+38
        if type:
            return alpha[self.condperiod]
        else:
            return alpha[self.condperiod].iloc[-1, :]

    # Alpha97  STD(VOLUME,10)
    def old_031(self, type=1):
        alpha = func_rolling_std(self.S_DQ_VOLUME, 10)
        return alpha
        alpha[alpha >= 1.0E+38] = 1.0E+38
        alpha[alpha <= -1.0E+38] = -1.0E+38
        if type:
            return alpha[self.condperiod]
        else:
            return alpha[self.condperiod].iloc[-1, :]

    # Alpha99  (-1 * RANK(COVIANCE(RANK(CLOSE), RANK(VOLUME), 5)))
    def old_032(self, type=1):
        #alpha = -pd.rolling_cov(self.S_FWDS_ADJCLOSE, self.S_DQ_VOLUME, 5)
        #d = pd.DataFrame({'S_FWDS_ADJCLOS':self.S_FWDS_ADJCLOSE, 'self.S_DQ_VOLUME':self.S_DQ_VOLUME})
        alpha = self.S_FWDS_ADJCLOSE.rolling(5).cov(self.S_DQ_VOLUME)
        return alpha
        alpha[alpha >= 1.0E+38] = 1.0E+38
        alpha[alpha <= -1.0E+38] = -1.0E+38
        if type:
            return alpha[self.condperiod]
        else:
            return alpha[self.condperiod].iloc[-1, :]

    # Alpha100  STD(VOLUME,20)
    def old_033(self, type=1):
        alpha = func_rolling_std(self.S_DQ_VOLUME, 20)
        return alpha
        alpha[alpha >= 1.0E+38] = 1.0E+38
        alpha[alpha <= -1.0E+38] = -1.0E+38
        if type:
            return alpha[self.condperiod]
        else:
            return alpha[self.condperiod].iloc[-1, :]

    # Alpha101  ((RANK(CORR(CLOSE, SUM(MEAN(VOLUME,30), 37), 15)) < RANK(CORR(RANK(((HIGH * 0.1) + (VWAP * 0.9))),RANK(VOLUME), 11))) * -1)
    def old_034(self, type=1):
        rank1 = func_rolling_corr(self.S_FWDS_ADJCLOSE, func_rolling_sum(func_rolling_mean(self.S_DQ_VOLUME, 10), 15), 7).rank(pct=True)
        rank2 = (self.S_FWDS_ADJHIGH*0.1+self.S_DQ_AVGPRICE*0.9).rank(pct=True)
        rank3 = self.S_DQ_VOLUME.rank(pct=True)
        rank4 = func_rolling_corr(rank2, rank3, 11).rank(pct=True)
        alpha = -(rank1 < rank4)
        return alpha
        alpha[alpha >= 1.0E+38] = 1.0E+38
        alpha[alpha <= -1.0E+38] = -1.0E+38
        if type:
            return alpha[self.condperiod]
        else:
            return alpha[self.condperiod].iloc[-1, :]

    # Alpha103  ((20-LOWDAY(LOW,20))/20)*100
    def old_035(self, type=1):
        alpha = 1 - self.S_FWDS_ADJLOW.rolling(20).apply(func_lowday)/20
        return alpha
        alpha[alpha >= 1.0E+38] = 1.0E+38
        alpha[alpha <= -1.0E+38] = -1.0E+38
        if type:
            return alpha[self.condperiod]
        else:
            return alpha[self.condperiod].iloc[-1, :]

    # Alpha104  (-1 * (DELTA(CORR(HIGH, VOLUME, 5), 5) * RANK(STD(CLOSE, 20))))
    def old_036(self, type=1):
        corr = func_rolling_corr(self.S_FWDS_ADJHIGH, self.S_DQ_VOLUME, 5)
        alpha = (-corr.diff(5) * func_rolling_std(self.S_FWDS_ADJCLOSE, 20))
        return alpha
        alpha[alpha >= 1.0E+38] = 1.0E+38
        alpha[alpha <= -1.0E+38] = -1.0E+38
        if type:
            return alpha[self.condperiod]
        else:
            return alpha[self.condperiod].iloc[-1, :]

    # Alpha105  (-1 * CORR(RANK(OPEN), RANK(VOLUME), 10))
    def old_037(self, type=1):
        alpha = -func_rolling_corr(self.S_FWDS_ADJOPEN.rank(pct=True), self.S_DQ_VOLUME.rank(pct=True), 10)
        return alpha
        alpha[alpha >= 1.0E+38] = 1.0E+38
        alpha[alpha <= -1.0E+38] = -1.0E+38
        if type:
            return alpha[self.condperiod]
        else:
            return alpha[self.condperiod].iloc[-1, :]

    # Alpha107  (((-1 * RANK((OPEN - DELAY(HIGH, 1)))) * RANK((OPEN - DELAY(CLOSE, 1)))) * RANK((OPEN - DELAY(LOW, 1))))
    def old_038(self, type=1):
        rank1 = -(self.S_FWDS_ADJOPEN-self.S_FWDS_ADJHIGH.shift()).rank(pct=True)
        rank2 = (self.S_FWDS_ADJOPEN-self.S_FWDS_ADJCLOSE.shift()).rank(pct=True)
        rank3 = (self.S_FWDS_ADJOPEN-self.S_FWDS_ADJLOW.shift()).rank(pct=True)
        alpha = (rank1*rank2*rank3)
        return alpha
        alpha[alpha >= 1.0E+38] = 1.0E+38
        alpha[alpha <= -1.0E+38] = -1.0E+38
        if type:
            return alpha[self.condperiod]
        else:
            return alpha[self.condperiod].iloc[-1, :]

    # Alpha109  SMA(HIGH-LOW,10,2)/SMA(SMA(HIGH-LOW,10,2),10,2)
    def old_039(self, type=1):
        data = self.S_FWDS_ADJHIGH-self.S_FWDS_ADJLOW
        sma1 = func_ewma(data, alpha=1.0/10)
        sma2 = func_ewma(sma1, alpha=1.0/10)
        alpha = sma1/sma2
        return alpha
        alpha[alpha >= 1.0E+38] = 1.0E+38
        alpha[alpha <= -1.0E+38] = -1.0E+38
        if type:
            return alpha[self.condperiod]
        else:
            return alpha[self.condperiod].iloc[-1, :]

    # Alpha112  (SUM((CLOSE-DELAY(CLOSE,1)>0?CLOSE-DELAY(CLOSE,1):0),12)-SUM((CLOSE-DELAY(CLOSE,1)<0?ABS(CLOSE-DELAY(CLOSE,1)):0),12))/(SUM((CLOSE-DELAY(CLOSE,1)>0?CLOSE-DELAY(CLOSE,1):0),12)+SUM((CLOSE-DELAY(CLOSE,1)<0?ABS(CLOSE-DELAY(CLOSE,1)):0),12))*100
    def old_040(self, type=1):
        cond1 = self.S_FWDS_ADJCLOSE > self.S_FWDS_ADJCLOSE.shift()
        cond2 = self.S_FWDS_ADJCLOSE < self.S_FWDS_ADJCLOSE.shift()
        data1 = self.S_FWDS_ADJCLOSE-self.S_FWDS_ADJCLOSE.shift()
        data2 = self.S_FWDS_ADJCLOSE-self.S_FWDS_ADJCLOSE.shift()
        data1[~cond1] = 0
        data2[~cond2] = 0
        data2 = data2.abs()
        sum1 = func_rolling_sum(data1, 12)
        sum2 = func_rolling_sum(data2, 12)
        alpha = ((sum1-sum2)/(sum1+sum2)*100)
        return alpha
        alpha[alpha >= 1.0E+38] = 1.0E+38
        alpha[alpha <= -1.0E+38] = -1.0E+38
        if type:
            return alpha[self.condperiod]
        else:
            return alpha[self.condperiod].iloc[-1, :]

    # Alpha113  (-1 * ((RANK((SUM(DELAY(CLOSE, 5), 20) / 20)) * CORR(CLOSE, VOLUME, 2)) * RANK(CORR(SUM(CLOSE, 5),SUM(CLOSE, 20), 2))))
    def old_041(self, type=1):
        part1 = func_rolling_mean(self.S_FWDS_ADJCLOSE.shift(5), 20).rank(pct=True)
        part2 = func_rolling_corr(self.S_FWDS_ADJCLOSE, self.S_DQ_VOLUME, 5)
        part3 = func_rolling_corr(func_rolling_sum(self.S_FWDS_ADJCLOSE, 5), func_rolling_sum(self.S_FWDS_ADJCLOSE, 20), 5)
        alpha = (-part1*part2*part3)
        return alpha
        alpha[alpha >= 1.0E+38] = 1.0E+38
        alpha[alpha <= -1.0E+38] = -1.0E+38
        if type:
            return alpha[self.condperiod]
        else:
            return alpha[self.condperiod].iloc[-1, :]

    # Alpha118  SUM(HIGH-OPEN,20)/SUM(OPEN-LOW,20)*100
    def old_042(self, type=1):
        data1 = self.S_FWDS_ADJHIGH-self.S_FWDS_ADJOPEN
        data2 = self.S_FWDS_ADJOPEN-self.S_FWDS_ADJLOW
        data3 = func_rolling_sum(data1, 20)
        data4 = func_rolling_sum(data2, 20)
        alpha = data3/data4
        return alpha
        alpha[alpha >= 1.0E+38] = 1.0E+38
        alpha[alpha <= -1.0E+38] = -1.0E+38
        if type:
            return alpha[self.condperiod]
        else:
            return alpha[self.condperiod].iloc[-1, :]

    # Alpha120  (RANK((VWAP - CLOSE)) / RANK((VWAP + CLOSE)))
    def old_043(self, type=1):
        data1 = (self.S_DQ_AVGPRICE-self.S_FWDS_ADJCLOSE).rank(pct=True)
        data2 = (self.S_DQ_AVGPRICE+self.S_FWDS_ADJCLOSE).rank(pct=True)
        alpha = (data1/data2)
        return alpha
        alpha[alpha >= 1.0E+38] = 1.0E+38
        alpha[alpha <= -1.0E+38] = -1.0E+38
        if type:
            return alpha[self.condperiod]
        else:
            return alpha[self.condperiod].iloc[-1, :]

    # Alpha122  (SMA(SMA(SMA(LOG(CLOSE),13,2),13,2),13,2)-DELAY(SMA(SMA(SMA(LOG(CLOSE),13,2),13,2),13,2),1))/DELAY(SMA(SMA(SMA(LOG(CLOSE),13,2),13,2),13,2),1)
    def old_044(self, type=1):
        log_close = np.log(self.S_FWDS_ADJCLOSE + 1)
        data = func_ewma(func_ewma(func_ewma(log_close, alpha=2/13), alpha=2/13), alpha=2/13)
        alpha = data/data.shift() - 1
        return alpha
        alpha[alpha >= 1.0E+38] = 1.0E+38
        alpha[alpha <= -1.0E+38] = -1.0E+38
        if type:
            return alpha[self.condperiod]
        else:
            return alpha[self.condperiod].iloc[-1, :]

    # Alpha123  ((RANK(CORR(SUM(((HIGH + LOW) / 2), 20), SUM(MEAN(VOLUME,60), 20), 9)) < RANK(CORR(LOW, VOLUME,6))) * -1)
    def old_045(self, type=1):
        data1 = func_rolling_sum((self.S_FWDS_ADJHIGH + self.S_FWDS_ADJLOW)/2, 5)
        data2 = func_rolling_sum(func_rolling_mean(self.S_DQ_VOLUME, 10), 5)
        rank1 = func_rolling_corr(data1, data2, 9).rank(pct=True)
        rank2 = func_rolling_corr(self.S_FWDS_ADJLOW, self.S_DQ_VOLUME, 6).rank(pct=True)
        alpha = rank1 < rank2
        return alpha
        alpha[alpha >= 1.0E+38] = 1.0E+38
        alpha[alpha <= -1.0E+38] = -1.0E+38
        if type:
            return alpha[self.condperiod]
        else:
            return alpha[self.condperiod].iloc[-1, :]

    # add old features again
    # Alpha129  SUM((CLOSE-DELAY(CLOSE,1)<0?ABS(CLOSE-DELAY(CLOSE,1)):0),12)
    def old_046(self, type=1):
        part1 = self.S_FWDS_ADJCLOSE.diff().abs()
        part1[(self.S_FWDS_ADJCLOSE >= self.S_FWDS_ADJCLOSE.shift())] = 0
        alpha = func_rolling_sum(part1, 12)
        return alpha
        alpha[alpha >= 1.0E+38] = 1.0E+38
        alpha[alpha <= -1.0E+38] = -1.0E+38
        if type:
            return alpha[self.condperiod]
        else:
            return alpha[self.condperiod].iloc[-1, :]

    # Alpha133  ((20-HIGHDAY(HIGH,20))/20)*100-((20-LOWDAY(LOW,20))/20)*100
    def old_047(self, type=1):
        alpha = self.S_FWDS_ADJHIGH.rolling(20).apply(func_highday) - self.S_FWDS_ADJLOW.rolling(20).apply(func_lowday)
        return alpha
        alpha[alpha >= 1.0E+38] = 1.0E+38
        alpha[alpha <= -1.0E+38] = -1.0E+38
        if type:
            return alpha[self.condperiod]
        else:
            return alpha[self.condperiod].iloc[-1, :]

    # Alpha133  ((20-HIGHDAY(HIGH,20))/20)*100-((20-LOWDAY(LOW,20))/20)*100
    def old_048(self, type=1):
        alpha = self.S_FWDS_ADJHIGH.rolling(12).apply(func_highday) - self.S_FWDS_ADJLOW.rolling(12).apply(func_lowday)
        return alpha
        alpha[alpha >= 1.0E+38] = 1.0E+38
        alpha[alpha <= -1.0E+38] = -1.0E+38
        if type:
            return alpha[self.condperiod]
        else:
            return alpha[self.condperiod].iloc[-1, :]

    # Alpha134  (CLOSE-DELAY(CLOSE,12))/DELAY(CLOSE,12)*VOLUME
    def old_049(self, type=1):
        part = (self.S_FWDS_ADJCLOSE - self.S_FWDS_ADJCLOSE.shift())/self.S_FWDS_ADJCLOSE.shift() * self.S_DQ_VOLUME
        alpha = func_rolling_sum(part, 12)
        return alpha
        alpha[alpha >= 1.0E+38] = 1.0E+38
        alpha[alpha <= -1.0E+38] = -1.0E+38
        if type:
            return alpha[self.condperiod]
        else:
            return alpha[self.condperiod].iloc[-1, :]

    # Alpha135  SMA(DELAY(CLOSE/DELAY(CLOSE,20),1),20,1)
    def old_050(self, type=1):
        part = self.S_FWDS_ADJCLOSE.shift()/self.S_FWDS_ADJCLOSE.shift(21)
        alpha = func_ewma(part, alpha=1/20)
        return alpha
        alpha[alpha >= 1.0E+38] = 1.0E+38
        alpha[alpha <= -1.0E+38] = -1.0E+38
        if type:
            return alpha[self.condperiod]
        else:
            return alpha[self.condperiod].iloc[-1, :]

    # Alpha136  ((-1 * RANK(DELTA(RET, 3))) * CORR(OPEN, VOLUME, 10))
    def old_051(self, type=1):
        data1 = - self.S_FWDS_ADJCLOSE.pct_change().diff(3).rank(pct=True)
        data2 = func_rolling_corr(self.S_FWDS_ADJOPEN, self.S_DQ_VOLUME, 10)
        alpha = (data1 * data2)
        return alpha
        alpha[alpha >= 1.0E+38] = 1.0E+38
        alpha[alpha <= -1.0E+38] = -1.0E+38
        if type:
            return alpha[self.condperiod]
        else:
            return alpha[self.condperiod].iloc[-1, :]

    # Alpha139  (-1 * CORR(OPEN, VOLUME, 10))
    def old_052(self, type=1):
        alpha = - func_rolling_corr(self.S_FWDS_ADJOPEN, self.S_DQ_VOLUME, 10)
        return alpha
        alpha[alpha >= 1.0E+38] = 1.0E+38
        alpha[alpha <= -1.0E+38] = -1.0E+38
        if type:
            return alpha[self.condperiod]
        else:
            return alpha[self.condperiod].iloc[-1, :]

    # Alpha141  (RANK(CORR(RANK(HIGH), RANK(MEAN(VOLUME,15)), 9))* -1)
    def old_053(self, type=1):
        df1 = self.S_FWDS_ADJHIGH.rank(pct=True)
        df2 = func_rolling_mean(self.S_DQ_VOLUME, 15).rank(pct=True)
        alpha = -func_rolling_corr(df1, df2, 9).rank(pct=True)
        return alpha
        alpha[alpha >= 1.0E+38] = 1.0E+38
        alpha[alpha <= -1.0E+38] = -1.0E+38
        if type:
            return alpha[self.condperiod]
        else:
            return alpha[self.condperiod].iloc[-1, :]

    # Alpha142  (((-1 * RANK(TSRANK(CLOSE, 10))) * RANK(DELTA(DELTA(CLOSE, 1), 1))) * RANK(TSRANK((VOLUME/MEAN(VOLUME,20)), 5)))
    def old_054(self, type=1):
        rank1 = -self.S_FWDS_ADJCLOSE.rolling(10).apply(ts_rank).rank(pct=True)
        rank2 = self.S_FWDS_ADJCLOSE.diff(2).rank(pct=True)
        rank3 = (self.S_DQ_VOLUME / func_rolling_mean(self.S_DQ_VOLUME, 20)).rolling(5).apply(ts_rank).rank(pct=True)
        alpha = (rank1 * rank2 * rank3)
        return alpha
        alpha[alpha >= 1.0E+38] = 1.0E+38
        alpha[alpha <= -1.0E+38] = -1.0E+38
        if type:
            return alpha[self.condperiod]
        else:
            return alpha[self.condperiod].iloc[-1, :]

    # Alpha145  (MEAN(VOLUME,9)-MEAN(VOLUME,26))/MEAN(VOLUME,12)*100
    def old_055(self, type=1):
        alpha = (func_rolling_mean(self.S_DQ_VOLUME, 9) - func_rolling_mean(self.S_DQ_VOLUME, 26))/func_rolling_mean(self.S_DQ_VOLUME, 12)
        return alpha
        alpha[alpha >= 1.0E+38] = 1.0E+38
        alpha[alpha <= -1.0E+38] = -1.0E+38
        if type:
            return alpha[self.condperiod]
        else:
            return alpha[self.condperiod].iloc[-1, :]

    # Alpha148  ((RANK(CORR((OPEN), SUM(MEAN(VOLUME,60), 9), 6)) < RANK((OPEN - TSMIN(OPEN, 14)))) * -1)
    def old_056(self, type=1):
        rank1 = func_rolling_corr(self.S_FWDS_ADJOPEN, func_rolling_sum(func_rolling_mean(self.S_DQ_VOLUME, 20), 9), 6).rank(pct=True)
        rank2 = (self.S_FWDS_ADJOPEN - func_rolling_min(self.S_FWDS_ADJOPEN, 14)).rank(pct=True)
        alpha = rank1 - rank2
        return alpha
        alpha[alpha >= 1.0E+38] = 1.0E+38
        alpha[alpha <= -1.0E+38] = -1.0E+38
        if type:
            return alpha[self.condperiod]
        else:
            return alpha[self.condperiod].iloc[-1, :]

    # Alpha151  SMA(CLOSE-DELAY(CLOSE,20),20,1)
    def old_057(self, type=1):
        part = self.S_FWDS_ADJCLOSE - self.S_FWDS_ADJCLOSE.shift(12)
        alpha = func_ewma(part, alpha=1/12)
        return alpha
        alpha[alpha >= 1.0E+38] = 1.0E+38
        alpha[alpha <= -1.0E+38] = -1.0E+38
        if type:
            return alpha[self.condperiod]
        else:
            return alpha[self.condperiod].iloc[-1, :]

    # Alpha152  SMA(MEAN(DELAY(SMA(DELAY(CLOSE/DELAY(CLOSE,9),1),9,1),1),12)-MEAN(DELAY(SMA(DELAY(CLOSE/DELAY(CLOSE,9),1),9,1),1),26),9,1)
    def old_058(self, type=1):
        data1 = func_rolling_mean((func_ewma(((self.S_FWDS_ADJCLOSE/self.S_FWDS_ADJCLOSE.shift(9)).shift()), alpha=1/9)).shift(), 12)
        data2 = func_rolling_mean((func_ewma(((self.S_FWDS_ADJCLOSE/self.S_FWDS_ADJCLOSE.shift(9)).shift()), alpha=1/9)).shift(), 26)
        alpha = func_ewma(data1-data2, alpha=1/9)
        return alpha
        alpha[alpha >= 1.0E+38] = 1.0E+38
        alpha[alpha <= -1.0E+38] = -1.0E+38
        if type:
            return alpha[self.condperiod]
        else:
            return alpha[self.condperiod].iloc[-1, :]

    # Alpha155  SMA(VOLUME,13,2)-SMA(VOLUME,27,2)-SMA(SMA(VOLUME,13,2)-SMA(VOLUME,27,2),10,2)
    def old_059(self, type=1):
        sma1 = func_ewma(self.S_DQ_VOLUME, alpha=2/13)
        sma2 = func_ewma(self.S_DQ_VOLUME, alpha=2/27)
        alpha = func_ewma(sma1-sma2, alpha=1/5)
        return alpha
        alpha[alpha >= 1.0E+38] = 1.0E+38
        alpha[alpha <= -1.0E+38] = -1.0E+38
        if type:
            return alpha[self.condperiod]
        else:
            return alpha[self.condperiod].iloc[-1, :]

    # Alpha158  ((HIGH-SMA(CLOSE,15,2))-(LOW-SMA(CLOSE,15,2)))/CLOSE
    def old_060(self, type=1):
        alpha = (2*func_ewma(self.S_FWDS_ADJCLOSE, alpha=2/15)-self.S_FWDS_ADJHIGH-self.S_FWDS_ADJLOW)/self.S_FWDS_ADJCLOSE
        return alpha
        alpha[alpha >= 1.0E+38] = 1.0E+38
        alpha[alpha <= -1.0E+38] = -1.0E+38
        if type:
            return alpha[self.condperiod]
        else:
            return alpha[self.condperiod].iloc[-1, :]

    # Alpha161  MEAN(MAX(MAX((HIGH-LOW),ABS(DELAY(CLOSE,1)-HIGH)),ABS(DELAY(CLOSE,1)-LOW)),12)
    def old_061(self, type=1):
        data1 = (self.S_FWDS_ADJHIGH-self.S_FWDS_ADJLOW)
        data2 = abs(self.S_FWDS_ADJCLOSE.shift()-self.S_FWDS_ADJHIGH)
        data3 = abs(self.S_FWDS_ADJCLOSE.shift()-self.S_FWDS_ADJLOW)
        result = np.maximum(np.maximum(data1, data2), data3)
        alpha = func_rolling_mean(result, 12)
        return alpha
        alpha[alpha >= 1.0E+38] = 1.0E+38
        alpha[alpha <= -1.0E+38] = -1.0E+38
        if type:
            return alpha[self.condperiod]
        else:
            return alpha[self.condperiod].iloc[-1, :]

    # Alpha162  (SMA(MAX(CLOSE-DELAY(CLOSE,1),0),12,1)/SMA(ABS(CLOSE-DELAY(CLOSE,1)),12,1)*100-MIN(SMA(MAX(CLOSE-DELAY(CLOSE,1),0),12,1)/SMA(ABS(CLOSE-DELAY(CLOSE,1)),12,1)*100,12))/(MAX(SMA(MAX(CLOSE-DELAY(CLOSE,1),0),12,1)/SMA(ABS(CLOSE-DELAY(CLOSE,1)),12,1)*100,12)-MIN(SMA(MAX(CLOSE-DELAY(CLOSE,1),0),12,1)/SMA(ABS(CLOSE-DELAY(CLOSE,1)),12,1)*100,12))
    def old_062(self, type=1):
        part1 = func_ewma(np.maximum(self.S_FWDS_ADJCLOSE-self.S_FWDS_ADJCLOSE.shift(), 0), alpha=1/12)
        part2 = func_ewma(abs(self.S_FWDS_ADJCLOSE-self.S_FWDS_ADJCLOSE.shift()), alpha=1/12)
        alpha = (part1/part2*100 - func_rolling_min(part1/part2*100, 12))/(func_rolling_max(part1/part2*100, 12) - func_rolling_min(part1/part2*100, 12))
        return alpha
        alpha[alpha >= 1.0E+38] = 1.0E+38
        alpha[alpha <= -1.0E+38] = -1.0E+38
        if type:
            return alpha[self.condperiod]
        else:
            return alpha[self.condperiod].iloc[-1, :]

    # Alpha163  RANK(((((-1 * RET) * MEAN(VOLUME,20)) * VWAP) * (HIGH - CLOSE)))
    def old_063(self, type=1):
        alpha = (-self.S_FWDS_ADJCLOSE.pct_change())*func_rolling_mean(self.S_DQ_VOLUME, 20)*self.S_DQ_AVGPRICE*(self.S_FWDS_ADJHIGH-self.S_FWDS_ADJCLOSE)
        return alpha
        alpha[alpha >= 1.0E+38] = 1.0E+38
        alpha[alpha <= -1.0E+38] = -1.0E+38
        if type:
            return alpha[self.condperiod]
        else:
            return alpha[self.condperiod].iloc[-1, :]

    # Alpha164  SMA((((CLOSE>DELAY(CLOSE,1))?1/(CLOSE-DELAY(CLOSE,1)):1)-MIN(((CLOSE>DELAY(CLOSE,1))?1/(CLOSE-DELAY(CLOSE,1)):1),12))/(HIGH-LOW)*100,13,2)
    def old_064(self, type=1):
        cond = self.S_FWDS_ADJCLOSE > self.S_FWDS_ADJCLOSE.shift()
        data1 = 1/(self.S_FWDS_ADJCLOSE-self.S_FWDS_ADJCLOSE.shift())
        data1[~cond] = 1
        data2 = func_rolling_min(data1, 12)
        data3 = 100*(data1-data2)/(self.S_FWDS_ADJHIGH-self.S_FWDS_ADJLOW)
        alpha = func_ewma(data3, alpha=2/13)
        return alpha
        alpha[alpha >= 1.0E+38] = 1.0E+38
        alpha[alpha <= -1.0E+38] = -1.0E+38
        if type:
            return alpha[self.condperiod]
        else:
            return alpha[self.condperiod].iloc[-1, :]

    # Alpha167  SUM((CLOSE-DELAY(CLOSE,1)>0?CLOSE-DELAY(CLOSE,1):0),12)
    def old_065(self, type=1):
        data1 = self.S_FWDS_ADJCLOSE-self.S_FWDS_ADJCLOSE.shift()
        cond = (data1 <= 0)
        data1[cond] = 0
        alpha = func_rolling_sum(data1, 12)
        return alpha
        alpha[alpha >= 1.0E+38] = 1.0E+38
        alpha[alpha <= -1.0E+38] = -1.0E+38
        if type:
            return alpha[self.condperiod]
        else:
            return alpha[self.condperiod].iloc[-1, :]

    # Alpha169  SMA(MEAN(DELAY(SMA(CLOSE-DELAY(CLOSE,1),9,1),1),12)-MEAN(DELAY(SMA(CLOSE-DELAY(CLOSE,1),9,1),1),26),10,1)
    def old_066(self, type=1):
        data1 = func_ewma(self.S_FWDS_ADJCLOSE-self.S_FWDS_ADJCLOSE.shift(), alpha=1/9).shift()
        data2 = func_rolling_mean(data1, 12)
        data3 = func_rolling_mean(data1, 26)
        alpha = func_ewma(data2-data3, alpha=1/10)
        return alpha
        alpha[alpha >= 1.0E+38] = 1.0E+38
        alpha[alpha <= -1.0E+38] = -1.0E+38
        if type:
            return alpha[self.condperiod]
        else:
            return alpha[self.condperiod].iloc[-1, :]

    # Alpha174  SMA((CLOSE>DELAY(CLOSE,1)?STD(CLOSE,20):0),20,1)
    def old_067(self, type=1):
        cond = self.S_FWDS_ADJCLOSE > self.S_FWDS_ADJCLOSE.shift()
        data2 = func_rolling_std(self.S_FWDS_ADJCLOSE, 20)
        data2[~cond] = 0
        alpha = func_ewma(data2, alpha=1/20)
        return alpha
        alpha[alpha >= 1.0E+38] = 1.0E+38
        alpha[alpha <= -1.0E+38] = -1.0E+38
        if type:
            return alpha[self.condperiod]
        else:
            return alpha[self.condperiod].iloc[-1, :]

    # Alpha175  MEAN(MAX(MAX((HIGH-LOW),ABS(DELAY(CLOSE,1)-HIGH)),ABS(DELAY(CLOSE,1)-LOW)),6)
    def old_068(self, type=1):
        data1 = self.S_FWDS_ADJHIGH-self.S_FWDS_ADJLOW
        data2 = abs(self.S_FWDS_ADJCLOSE.shift()-self.S_FWDS_ADJHIGH)
        data3 = abs(self.S_FWDS_ADJCLOSE.shift()-self.S_FWDS_ADJLOW)
        alpha = func_rolling_mean(np.maximum(np.maximum(data1, data2), data3), 6)
        return alpha
        alpha[alpha >= 1.0E+38] = 1.0E+38
        alpha[alpha <= -1.0E+38] = -1.0E+38
        if type:
            return alpha[self.condperiod]
        else:
            return alpha[self.condperiod].iloc[-1, :]

    # Alpha176  CORR(RANK(((CLOSE - TSMIN(LOW, 12)) / (TSMAX(HIGH, 12) - TSMIN(LOW,12)))), RANK(VOLUME), 6)
    def old_069(self, type=1):
        data1 = (self.S_FWDS_ADJCLOSE-func_rolling_min(self.S_FWDS_ADJLOW, 12))/(func_rolling_max(self.S_FWDS_ADJHIGH, 12)-func_rolling_min(self.S_FWDS_ADJLOW, 12))
        data2 = data1.rank(pct=True)
        data3 = self.S_DQ_VOLUME.rank(pct=True)
        alpha = func_rolling_corr(data2, data3, 6)
        return alpha
        alpha[alpha >= 1.0E+38] = 1.0E+38
        alpha[alpha <= -1.0E+38] = -1.0E+38
        if type:
            return alpha[self.condperiod]
        else:
            return alpha[self.condperiod].iloc[-1, :]

    # Alpha177  ((20-HIGHDAY(HIGH,20))/20)*100
    def old_070(self, type=1):
        alpha = 20 - self.S_FWDS_ADJHIGH.rolling(20).apply(func_highday)
        return alpha
        alpha[alpha >= 1.0E+38] = 1.0E+38
        alpha[alpha <= -1.0E+38] = -1.0E+38
        if type:
            return alpha[self.condperiod]
        else:
            return alpha[self.condperiod].iloc[-1, :]

    # Alpha179  (RANK(CORR(VWAP, VOLUME, 4)) *RANK(CORR(RANK(LOW), RANK(MEAN(VOLUME,50)), 12)))
    def old_071(self, type=1):
        rank1 = func_rolling_corr(self.S_DQ_AVGPRICE, self.S_DQ_VOLUME, 4).rank(pct=True)
        data2 = self.S_FWDS_ADJLOW.rank(pct=True)
        data3 = func_rolling_mean(self.S_DQ_VOLUME, 20).rank(pct=True)
        rank2 = func_rolling_corr(data2, data3, 12).rank(pct=True)
        alpha = rank1*rank2
        return alpha
        alpha[alpha >= 1.0E+38] = 1.0E+38
        alpha[alpha <= -1.0E+38] = -1.0E+38
        if type:
            return alpha[self.condperiod]
        else:
            return alpha[self.condperiod].iloc[-1, :]

    # Alpha184  (RANK(CORR(DELAY((OPEN - CLOSE), 1), CLOSE, 200)) + RANK((OPEN - CLOSE)))
    def old_072(self, type=1):
        data1 = self.S_FWDS_ADJOPEN.shift()-self.S_FWDS_ADJCLOSE.shift()
        data2 = self.S_FWDS_ADJOPEN - self.S_FWDS_ADJCLOSE
        corr = func_rolling_corr(data1, self.S_FWDS_ADJCLOSE, 20)
        alpha = data2.rank(pct=True)+corr.rank(pct=True)
        return alpha
        alpha[alpha >= 1.0E+38] = 1.0E+38
        alpha[alpha <= -1.0E+38] = -1.0E+38
        if type:
            return alpha[self.condperiod]
        else:
            return alpha[self.condperiod].iloc[-1, :]

    # Alpha187  SUM((OPEN<=DELAY(OPEN,1)?0:MAX((HIGH-OPEN),(OPEN-DELAY(OPEN,1)))),20)
    def old_073(self, type=1):
        cond = (self.S_FWDS_ADJOPEN <= self.S_FWDS_ADJOPEN.shift())
        part = np.maximum(self.S_FWDS_ADJHIGH - self.S_FWDS_ADJOPEN, self.S_FWDS_ADJOPEN - self.S_FWDS_ADJOPEN.shift())
        part[cond] = 0
        alpha = func_rolling_sum(part, 20)
        return alpha
        alpha[alpha >= 1.0E+38] = 1.0E+38
        alpha[alpha <= -1.0E+38] = -1.0E+38
        if type:
            return alpha[self.condperiod]
        else:
            return alpha[self.condperiod].iloc[-1, :]

    # Alpha189  MEAN(ABS(CLOSE-MEAN(CLOSE,6)),6)
    def old_074(self, type=1):
        ma6 = func_rolling_mean(self.S_FWDS_ADJCLOSE, 6)
        alpha = func_rolling_mean(((self.S_FWDS_ADJCLOSE - ma6)/ma6).abs(), 6)
        return alpha
        alpha[alpha >= 1.0E+38] = 1.0E+38
        alpha[alpha <= -1.0E+38] = -1.0E+38
        if type:
            return alpha[self.condperiod]
        else:
            return alpha[self.condperiod].iloc[-1, :]

    # Alpha174  SMA((CLOSE>DELAY(CLOSE,1)?STD(CLOSE,20):0),20,1)
    def old_075(self, type=1):
        cond = self.S_FWDS_ADJCLOSE > self.S_FWDS_ADJCLOSE.shift()
        data2 = func_rolling_std(self.S_FWDS_ADJCLOSE, 6)
        data2[~cond] = 0
        alpha = func_ewma(data2, alpha=1/6)
        return alpha
        alpha[alpha >= 1.0E+38] = 1.0E+38
        alpha[alpha <= -1.0E+38] = -1.0E+38
        if type:
            return alpha[self.condperiod]
        else:
            return alpha[self.condperiod].iloc[-1, :]

    # Alpha174  SMA((CLOSE>DELAY(CLOSE,1)?STD(CLOSE,20):0),20,1)
    def old_076(self, type=1):
        cond = self.S_FWDS_ADJCLOSE > self.S_FWDS_ADJCLOSE.shift()
        data2 = func_rolling_std(self.S_FWDS_ADJCLOSE, 12)
        data2[~cond] = 0
        alpha = func_ewma(data2, alpha=1/12)
        return alpha
        alpha[alpha >= 1.0E+38] = 1.0E+38
        alpha[alpha <= -1.0E+38] = -1.0E+38
        if type:
            return alpha[self.condperiod]
        else:
            return alpha[self.condperiod].iloc[-1, :]

    # better features
    # Alpha15  OPEN/DELAY(CLOSE,1)-1
    def better_001(self, type=1):
        alpha = (self.S_FWDS_ADJOPEN/self.S_FWDS_ADJCLOSE.shift()-1) * np.log2(self.S_DQ_VOLUME + 1)
        return alpha
        alpha[alpha >= 1.0E+38] = 1.0E+38
        alpha[alpha <= -1.0E+38] = -1.0E+38
        #result[result > 0.08] = np.nan
        if type:
            return alpha[self.condperiod]
        else:
            return alpha[self.condperiod].iloc[-1, :]

    # Alpha15  OPEN/DELAY(CLOSE,1)-1
    def better_002(self, type=1):
        #alpha=(self.S_FWDS_ADJOPEN/self.S_FWDS_ADJCLOSE.shift()-1) * self.S_DQ_VOLUME/func_rolling_mean(self.S_DQ_VOLUME, 5)
        alpha = (self.S_FWDS_ADJOPEN/self.S_FWDS_ADJCLOSE.shift()-1) * self.S_DQ_VOLUME/self.S_DQ_VOLUME.rolling(5).mean()
        return alpha
        alpha[alpha >= 1.0E+38] = 1.0E+38
        alpha[alpha <= -1.0E+38] = -1.0E+38
        #result[result > 0.08] = np.nan
        if type:
            return alpha[self.condperiod]
        else:
            return alpha[self.condperiod].iloc[-1, :]

    # Alpha15  OPEN/DELAY(CLOSE,1)-1
    def better_003(self, type=1):
        alpha = (self.S_FWDS_ADJOPEN/self.S_FWDS_ADJCLOSE.shift()-1) * (self.S_DQ_AVGPRICE - self.S_FWDS_ADJCLOSE) * (self.S_FWDS_ADJHIGH - self.S_FWDS_ADJLOW) / self.S_FWDS_ADJCLOSE
        return alpha
        alpha[alpha >= 1.0E+38] = 1.0E+38
        alpha[alpha <= -1.0E+38] = -1.0E+38
        #result[result > 0.08] = np.nan
        if type:
            return alpha[self.condperiod]
        else:
            return alpha[self.condperiod].iloc[-1, :]

    # Alpha2  (-1 * DELTA((((CLOSE - LOW) - (HIGH - CLOSE)) / (HIGH - LOW)), 1))
    def better_004(self, type=1):
        alpha = -((2*self.S_DQ_AVGPRICE-self.S_FWDS_ADJLOW-self.S_FWDS_ADJHIGH)/(self.S_FWDS_ADJHIGH-self.S_FWDS_ADJLOW)).diff()
        return alpha
        alpha[alpha >= 1.0E+38] = 1.0E+38
        alpha[alpha <= -1.0E+38] = -1.0E+38
        if type:
            return alpha[self.condperiod]
        else:
            return alpha[self.condperiod].iloc[-1, :]

    # Alpha2  (-1 * DELTA((((CLOSE - LOW) - (HIGH - CLOSE)) / (HIGH - LOW)), 1))
    def better_005(self, type=1):
        alpha = -((2*self.S_DQ_AVGPRICE-self.S_FWDS_ADJLOW-self.S_FWDS_ADJHIGH)/(self.S_FWDS_ADJHIGH-self.S_FWDS_ADJLOW)).diff(5)
        return alpha
        alpha[alpha >= 1.0E+38] = 1.0E+38
        alpha[alpha <= -1.0E+38] = -1.0E+38
        if type:
            return alpha[self.condperiod]
        else:
            return alpha[self.condperiod].iloc[-1, :]

    # Alpha2  (-1 * DELTA((((CLOSE - LOW) - (HIGH - CLOSE)) / (HIGH - LOW)), 1))
    def better_006(self, type=1):
        alpha = func_rolling_mean((2*self.S_DQ_AVGPRICE-self.S_FWDS_ADJLOW-self.S_FWDS_ADJHIGH)/(self.S_FWDS_ADJHIGH-self.S_FWDS_ADJLOW), 5)
        return alpha
        alpha[alpha >= 1.0E+38] = 1.0E+38
        alpha[alpha <= -1.0E+38] = -1.0E+38
        if type:
            return alpha[self.condperiod]
        else:
            return alpha[self.condperiod].iloc[-1, :]

    def better_007(self, type=1):
        alpha = (self.S_FWDS_ADJHIGH - self.S_FWDS_ADJLOW)/self.S_FWDS_ADJCLOSE * self.S_DQ_VOLUME.diff()
        return alpha
        alpha[alpha >= 1.0E+38] = 1.0E+38
        alpha[alpha <= -1.0E+38] = -1.0E+38
        if type:
            return alpha[self.condperiod]
        else:
            return alpha[self.condperiod].iloc[-1, :]

    def better_008(self, type=1):
        alpha = (self.S_FWDS_ADJHIGH - self.S_FWDS_ADJLOW)/self.S_FWDS_ADJCLOSE / np.log2(self.S_DQ_VOLUME + 1)
        return alpha
        alpha[alpha >= 1.0E+38] = 1.0E+38
        alpha[alpha <= -1.0E+38] = -1.0E+38
        if type:
            return alpha[self.condperiod]
        else:
            return alpha[self.condperiod].iloc[-1, :]

    # Alpha144  SUMIF(ABS(CLOSE/DELAY(CLOSE,1)-1)/AMOUNT,20,CLOSE<DELAY(CLOSE,1))/COUNT(CLOSE<DELAY(CLOSE,1),20)
    def better_009(self, type=1):
        df1 = self.S_DQ_AVGPRICE < self.S_DQ_AVGPRICE.shift()
        sumif = func_rolling_sum(((abs(self.S_DQ_AVGPRICE / self.S_DQ_AVGPRICE.shift() - 1)/np.log(self.S_DQ_AMOUNT + 1))[df1].fillna(0)), 20)
        count = func_rolling_sum(df1, 20)
        alpha = (sumif / count)
        return alpha
        alpha[alpha >= 1.0E+38] = 1.0E+38
        alpha[alpha <= -1.0E+38] = -1.0E+38
        if type:
            return alpha[self.condperiod]
        else:
            return alpha[self.condperiod].iloc[-1, :]

    # Alpha144  SUMIF(ABS(CLOSE/DELAY(CLOSE,1)-1)/AMOUNT,20,CLOSE<DELAY(CLOSE,1))/COUNT(CLOSE<DELAY(CLOSE,1),20)
    def better_010(self, type=1):
        df1 = self.S_DQ_AVGPRICE < self.S_DQ_AVGPRICE.shift()
        sumif = func_rolling_sum(((abs(self.S_DQ_AVGPRICE / self.S_DQ_AVGPRICE.shift() - 1)/np.log(self.S_DQ_AMOUNT + 1))[df1].fillna(0)), 10)
        count = func_rolling_sum(df1, 10)
        alpha = (sumif / count)
        return alpha
        alpha[alpha >= 1.0E+38] = 1.0E+38
        alpha[alpha <= -1.0E+38] = -1.0E+38
        if type:
            return alpha[self.condperiod]
        else:
            return alpha[self.condperiod].iloc[-1, :]

    # Alpha159  ((CLOSE-SUM(MIN(LOW,DELAY(CLOSE,1)),6))/SUM(MAX(HGIH,DELAY(CLOSE,1))-MIN(LOW,DELAY(CLOSE,1)),6)*12*24+(CLOSE-SUM(MIN(LOW,DELAY(CLOSE,1)),12))/SUM(MAX(HGIH,DELAY(CLOSE,1))-MIN(LOW,DELAY(CLOSE,1)),12)*6*24+(CLOSE-SUM(MIN(LOW,DELAY(CLOSE,1)),24))/SUM(MAX(HGIH,DELAY(CLOSE,1))-MIN(LOW,DELAY(CLOSE,1)),24)*6*24)*100/(6*12+6*24+12*24)
    def better_011(self, type=1):

        # ((CLOSE-SUM(mitt,6))/SUM(matt-mitt,6)*12*24+(CLOSE-SUM(mitt,12))/SUM(matt-mitt,12)*6*24+(CLOSE-SUM(mitt,24))/SUM(matt-mitt,24)*6*24)*100/(6*12+6*24+12*24)
        data1 = np.minimum(self.S_FWDS_ADJLOW, self.S_FWDS_ADJLOW.shift(1))
        data2 = np.maximum(self.S_FWDS_ADJHIGH, self.S_FWDS_ADJHIGH.shift(1))
        part1 = (self.S_FWDS_ADJCLOSE - func_rolling_sum(data1, 6))/func_rolling_sum(data2-data1, 6)*12*24
        part2 = (self.S_FWDS_ADJCLOSE - func_rolling_sum(data1, 12))/func_rolling_sum(data2-data1, 12)*6*24
        part3 = (self.S_FWDS_ADJCLOSE - func_rolling_sum(data1, 24))/func_rolling_sum(data2-data1, 24)*6*24
        alpha = (part1+part2+part3)*100/(6*12+6*24+12*24)
        return alpha
        alpha[alpha >= 1.0E+38] = 1.0E+38
        alpha[alpha <= -1.0E+38] = -1.0E+38
        if type:
            return alpha[self.condperiod]
        else:
            return alpha[self.condperiod].iloc[-1, :]

    # Alpha159  ((CLOSE-SUM(MIN(LOW,DELAY(CLOSE,1)),6))/SUM(MAX(HGIH,DELAY(CLOSE,1))-MIN(LOW,DELAY(CLOSE,1)),6)*12*24+(CLOSE-SUM(MIN(LOW,DELAY(CLOSE,1)),12))/SUM(MAX(HGIH,DELAY(CLOSE,1))-MIN(LOW,DELAY(CLOSE,1)),12)*6*24+(CLOSE-SUM(MIN(LOW,DELAY(CLOSE,1)),24))/SUM(MAX(HGIH,DELAY(CLOSE,1))-MIN(LOW,DELAY(CLOSE,1)),24)*6*24)*100/(6*12+6*24+12*24)
    def better_012(self, type=1):

        # ((CLOSE-SUM(mitt,6))/SUM(matt-mitt,6)*12*24+(CLOSE-SUM(mitt,12))/SUM(matt-mitt,12)*6*24+(CLOSE-SUM(mitt,24))/SUM(matt-mitt,24)*6*24)*100/(6*12+6*24+12*24)
        data1 = np.minimum(self.S_FWDS_ADJLOW, self.S_DQ_AVGPRICE.shift(1))
        data2 = np.maximum(self.S_FWDS_ADJHIGH, self.S_DQ_AVGPRICE.shift(1))
        part1 = (self.S_DQ_AVGPRICE - func_rolling_sum(data1, 6))/func_rolling_sum(data2-data1, 6)*12*24
        part2 = (self.S_DQ_AVGPRICE - func_rolling_sum(data1, 12))/func_rolling_sum(data2-data1, 12)*6*24
        part3 = (self.S_DQ_AVGPRICE - func_rolling_sum(data1, 24))/func_rolling_sum(data2-data1, 24)*6*24
        alpha = (part1+part2+part3)*100/(6*12+6*24+12*24)
        return alpha
        alpha[alpha >= 1.0E+38] = 1.0E+38
        alpha[alpha <= -1.0E+38] = -1.0E+38
        if type:
            return alpha[self.condperiod]

    def better_013(self, type=1):
        alpha = (self.S_DQ_AVGPRICE / self.S_FWDS_ADJCLOSE - 1) * np.log2(self.S_DQ_AMOUNT + 1)
        return alpha
        alpha[alpha >= 1.0E+38] = 1.0E+38
        alpha[alpha <= -1.0E+38] = -1.0E+38
        if type:
            return alpha[self.condperiod]
        else:
            return alpha[self.condperiod].iloc[-1, :]

    def better_014(self, type=1):
        alpha = func_rolling_mean((self.S_DQ_AVGPRICE / self.S_FWDS_ADJCLOSE - 1) * np.log2(self.S_DQ_AMOUNT + 1), 4)
        return alpha
        alpha[alpha >= 1.0E+38] = 1.0E+38
        alpha[alpha <= -1.0E+38] = -1.0E+38
        if type:
            return alpha[self.condperiod]
        else:
            return alpha[self.condperiod].iloc[-1, :]

    # Alpha22  SMEAN(((CLOSE-MEAN(CLOSE,6))/MEAN(CLOSE,6)-DELAY((CLOSE-MEAN(CLOSE,6))/MEAN(CLOSE,6),3)),12,1)
    def better_015(self, type=1):
        part0 = func_rolling_mean(self.S_DQ_AVGPRICE, 10)
        part1 = (self.S_DQ_AVGPRICE - part0)/part0
        alpha = part1-part1.shift(5)
        alpha = func_ewma(alpha, alpha=1.0/20)
        return alpha
        alpha[alpha >= 1.0E+38] = 1.0E+38
        alpha[alpha <= -1.0E+38] = -1.0E+38
        if type:
            return alpha[self.condperiod]
        else:
            return alpha[self.condperiod].iloc[-1, :]

    # Alpha22  SMEAN(((CLOSE-MEAN(CLOSE,6))/MEAN(CLOSE,6)-DELAY((CLOSE-MEAN(CLOSE,6))/MEAN(CLOSE,6),3)),12,1)
    def better_016(self, type=1):
        part0 = func_rolling_mean(self.S_DQ_AVGPRICE, 6)
        part1 = (self.S_DQ_AVGPRICE - part0)/part0
        alpha = part1-part1.shift(3)
        alpha = func_ewma(alpha, alpha=1.0/12)
        return alpha
        alpha[alpha >= 1.0E+38] = 1.0E+38
        alpha[alpha <= -1.0E+38] = -1.0E+38
        if type:
            return alpha[self.condperiod]
        else:
            return alpha[self.condperiod].iloc[-1, :]

    # Alpha22  SMEAN(((CLOSE-MEAN(CLOSE,6))/MEAN(CLOSE,6)-DELAY((CLOSE-MEAN(CLOSE,6))/MEAN(CLOSE,6),3)),12,1)
    def better_017(self, type=1):
        part0 = func_rolling_mean(self.S_DQ_AMOUNT, 6)
        part1 = (self.S_DQ_AMOUNT - part0)/part0
        alpha = part1-part1.shift(3)
        alpha = func_ewma(alpha, alpha=1.0/12)
        return alpha
        alpha[alpha >= 1.0E+38] = 1.0E+38
        alpha[alpha <= -1.0E+38] = -1.0E+38
        if type:
            return alpha[self.condperiod]
        else:
            return alpha[self.condperiod].iloc[-1, :]

    # Alpha22  SMEAN(((CLOSE-MEAN(CLOSE,6))/MEAN(CLOSE,6)-DELAY((CLOSE-MEAN(CLOSE,6))/MEAN(CLOSE,6),3)),12,1)
    def better_018(self, type=1):
        part0 = func_rolling_mean(self.S_DQ_AMOUNT, 10)
        part1 = (self.S_DQ_AMOUNT - part0)/part0
        alpha = part1-part1.shift(5)
        alpha = func_ewma(alpha, alpha=1.0/20)
        return alpha
        alpha[alpha >= 1.0E+38] = 1.0E+38
        alpha[alpha <= -1.0E+38] = -1.0E+38
        if type:
            return alpha[self.condperiod]
        else:
            return alpha[self.condperiod].iloc[-1, :]

    # Alpha122  (SMA(SMA(SMA(LOG(CLOSE),13,2),13,2),13,2)-DELAY(SMA(SMA(SMA(LOG(CLOSE),13,2),13,2),13,2),1))/DELAY(SMA(SMA(SMA(LOG(CLOSE),13,2),13,2),13,2),1)
    def better_019(self, type=1):
        log_close = np.log(self.S_DQ_VOLUME + 1)
        data = func_ewma(func_ewma(func_ewma(log_close, alpha=2/13), alpha=2/13), alpha=2/13)
        alpha = data/data.shift() - 1
        return alpha
        alpha[alpha >= 1.0E+38] = 1.0E+38
        alpha[alpha <= -1.0E+38] = -1.0E+38
        if type:
            return alpha[self.condperiod]
        else:
            return alpha[self.condperiod].iloc[-1, :]

    # Alpha76  STD(ABS((CLOSE/DELAY(CLOSE,1)-1))/VOLUME,20)/MEAN(ABS((CLOSE/DELAY(CLOSE,1)-1))/VOLUME,20)
    def better_020(self, type=1):
        data1 = func_rolling_std((self.S_FWDS_ADJCLOSE/self.S_FWDS_ADJCLOSE.shift()-1).abs()/self.S_DQ_VOLUME, 10)
        data2 = func_rolling_mean((self.S_FWDS_ADJCLOSE/self.S_FWDS_ADJCLOSE.shift()-1).abs()/self.S_DQ_VOLUME, 10)
        alpha = (data1 / data2)
        return alpha
        alpha[alpha >= 1.0E+38] = 1.0E+38
        alpha[alpha <= -1.0E+38] = -1.0E+38
        if type:
            return alpha[self.condperiod]
        else:
            return alpha[self.condperiod].iloc[-1, :]

    def better_021(self, type=1):
        alpha = self.S_DQ_AVGPRICE / self.S_FWDS_ADJCLOSE.shift()
        return alpha
        alpha[alpha >= 1.0E+38] = 1.0E+38
        alpha[alpha <= -1.0E+38] = -1.0E+38
        if type:
            return alpha[self.condperiod]
        else:
            return alpha[self.condperiod].iloc[-1, :]

    def better_022(self, type=1):
        alpha = (self.S_DQ_AVGPRICE / self.S_FWDS_ADJCLOSE.shift() - 1) * np.log2(self.S_DQ_AMOUNT + 1)
        return alpha
        alpha[alpha >= 1.0E+38] = 1.0E+38
        alpha[alpha <= -1.0E+38] = -1.0E+38
        if type:
            return alpha[self.condperiod]
        else:
            return alpha[self.condperiod].iloc[-1, :]

    def better_023(self, type=1):
        alpha = func_rolling_mean((self.S_DQ_AVGPRICE / self.S_FWDS_ADJCLOSE.shift() - 1) * np.log2(self.S_DQ_AMOUNT + 1), 4)
        return alpha
        alpha[alpha >= 1.0E+38] = 1.0E+38
        alpha[alpha <= -1.0E+38] = -1.0E+38
        if type:
            return alpha[self.condperiod]
        else:
            return alpha[self.condperiod].iloc[-1, :]

    # Alpha7  ((RANK(MAX((VWAP - CLOSE), 3)) + RANK(MIN((VWAP - CLOSE), 3))) * RANK(DELTA(VOLUME, 3)))
    def better_024(self, type=1):
        part1 = (func_rolling_max(self.S_DQ_AVGPRICE-self.S_FWDS_ADJCLOSE, 5))
        part2 = (func_rolling_min(self.S_DQ_AVGPRICE-self.S_FWDS_ADJCLOSE, 5))
        part3 = (self.S_DQ_VOLUME.diff(5))
        alpha = (part1+part2)*part3
        return alpha
        alpha[alpha >= 1.0E+38] = 1.0E+38
        alpha[alpha <= -1.0E+38] = -1.0E+38
        if type:
            return alpha[self.condperiod]
        else:
            return alpha[self.condperiod].iloc[-1, :]

    # Alpha7  ((RANK(MAX((VWAP - CLOSE), 3)) + RANK(MIN((VWAP - CLOSE), 3))) * RANK(DELTA(VOLUME, 3)))
    def better_025(self, type=1):
        part1 = (func_rolling_max(self.S_DQ_AVGPRICE-self.S_FWDS_ADJCLOSE, 5))
        part2 = (func_rolling_min(self.S_DQ_AVGPRICE-self.S_FWDS_ADJCLOSE, 5))
        part3 = self.S_DQ_VOLUME - func_rolling_mean(self.S_DQ_VOLUME, 5)
        alpha = (part1+part2)*part3
        return alpha
        alpha[alpha >= 1.0E+38] = 1.0E+38
        alpha[alpha <= -1.0E+38] = -1.0E+38
        if type:
            return alpha[self.condperiod]
        else:
            return alpha[self.condperiod].iloc[-1, :]

    # Alpha109  SMA(HIGH-LOW,5,2)/SMA(SMA(HIGH-LOW,10,2),20,2)
    def better_026(self, type=1):
        data = (self.S_FWDS_ADJHIGH-self.S_FWDS_ADJLOW)/self.S_FWDS_ADJCLOSE
        sma1 = func_ewma(data, alpha=2.0/5)
        sma2 = func_ewma(sma1, alpha=2.0/20)
        alpha = (sma1/sma2)
        return alpha
        alpha[alpha >= 1.0E+38] = 1.0E+38
        alpha[alpha <= -1.0E+38] = -1.0E+38
        if type:
            return alpha[self.condperiod]
        else:
            return alpha[self.condperiod].iloc[-1, :]

    # Alpha144  SUMIF(ABS(CLOSE/DELAY(CLOSE,1)-1)/AMOUNT,20,CLOSE<DELAY(CLOSE,1))/COUNT(CLOSE<DELAY(CLOSE,1),20)
    def better_027(self, type=1):
        df1 = self.S_DQ_AVGPRICE < self.S_FWDS_ADJCLOSE.shift()
        sumif = func_rolling_sum(((abs(self.S_DQ_AVGPRICE / self.S_FWDS_ADJCLOSE.shift() - 1)/np.log(self.S_DQ_AMOUNT + 1))[df1].fillna(0)), 20)
        count = func_rolling_sum(df1, 20)
        alpha = (sumif / count)
        return alpha
        alpha[alpha >= 1.0E+38] = 1.0E+38
        alpha[alpha <= -1.0E+38] = -1.0E+38
        if type:
            return alpha[self.condperiod]
        else:
            return alpha[self.condperiod].iloc[-1, :]

    # Alpha144  SUMIF(ABS(CLOSE/DELAY(CLOSE,1)-1)/AMOUNT,20,CLOSE<DELAY(CLOSE,1))/COUNT(CLOSE<DELAY(CLOSE,1),20)
    def better_028(self, type=1):
        df1 = self.S_DQ_AVGPRICE < self.S_FWDS_ADJCLOSE
        sumif = func_rolling_sum(((abs(self.S_DQ_AVGPRICE / self.S_FWDS_ADJCLOSE - 1)/np.log(self.S_DQ_AMOUNT + 1))[df1].fillna(0)), 20)
        count = func_rolling_sum(df1, 20)
        alpha = (sumif / count)
        return alpha
        if type:
            return alpha[self.condperiod]
        else:
            return alpha[self.condperiod].iloc[-1, :]

    '''
    #拉升斜率
    def old_046(self, type=1):
        alpha = self.S_FWDS_ADJCLOSE.rolling(6).apply(func_grad)
        alpha[alpha >= -1.4E-12][alpha <= 1.4E-12] =  0
        alpha[alpha >= 1.0E+38] =  1.0E+38
        alpha[alpha <= -1.0E+38] =  -1.0E+38
        if type:
            return alpha[self.condperiod]
        else:
            return alpha[self.condperiod].iloc[-1,:]  
            
    #拉升斜率
    def old_047(self, type=1):
        alpha = self.S_FWDS_ADJCLOSE.rolling(12).apply(func_grad)
        alpha[alpha >= -1.4E-12][alpha <= 1.4E-12] =  0
        alpha[alpha >= 1.0E+38] =  1.0E+38
        alpha[alpha <= -1.0E+38] =  -1.0E+38
        if type:
            return alpha[self.condperiod]
        else:
            return alpha[self.condperiod].iloc[-1,:]          
            
    #拉升斜率
    def old_048(self, type=1):
        alpha = self.S_FWDS_ADJCLOSE.rolling(20).apply(func_grad)
        alpha[alpha >= -1.4E-12][alpha <= 1.4E-12] =  0
        alpha[alpha >= 1.0E+38] =  1.0E+38
        alpha[alpha <= -1.0E+38] =  -1.0E+38
        if type:
            return alpha[self.condperiod]
        else:
            return alpha[self.condperiod].iloc[-1,:]          
    '''

    def target_01(self, type=1):
        alpha = (self.S_FWDS_ADJCLOSE.shift(-1) - self.S_FWDS_ADJCLOSE) / self.S_FWDS_ADJCLOSE
        return alpha
        alpha[alpha >= 1.0E+38] = 1.0E+38
        alpha[alpha <= -1.0E+38] = -1.0E+38
        if type:
            return alpha[self.condperiod]
        else:
            return alpha[self.condperiod].iloc[-1, :]

    def target_02(self, type=1):
        alpha = (self.S_FWDS_ADJCLOSE.shift(-2) - self.S_FWDS_ADJCLOSE) / self.S_FWDS_ADJCLOSE
        return alpha
        alpha[alpha >= 1.0E+38] = 1.0E+38
        alpha[alpha <= -1.0E+38] = -1.0E+38
        if type:
            return alpha[self.condperiod]
        else:
            return alpha[self.condperiod].iloc[-1, :]

    def target_03(self, type=1):
        alpha = (self.S_FWDS_ADJCLOSE.shift(-3) - self.S_FWDS_ADJCLOSE) / self.S_FWDS_ADJCLOSE
        return alpha
        alpha[alpha >= 1.0E+38] = 1.0E+38
        alpha[alpha <= -1.0E+38] = -1.0E+38
        if type:
            return alpha[self.condperiod]
        else:
            return alpha[self.condperiod].iloc[-1, :]

    def target_04(self, type=1):
        alpha = (self.S_FWDS_ADJCLOSE.shift(-4) - self.S_FWDS_ADJCLOSE) / self.S_FWDS_ADJCLOSE
        return alpha
        alpha[alpha >= 1.0E+38] = 1.0E+38
        alpha[alpha <= -1.0E+38] = -1.0E+38
        if type:
            return alpha[self.condperiod]
        else:
            return alpha[self.condperiod].iloc[-1, :]

    def target_05(self, type=1):
        alpha = (self.S_FWDS_ADJCLOSE.shift(-5) - self.S_FWDS_ADJCLOSE) / self.S_FWDS_ADJCLOSE
        return alpha
        alpha[alpha >= 1.0E+38] = 1.0E+38
        alpha[alpha <= -1.0E+38] = -1.0E+38
        if type:
            return alpha[self.condperiod]
        else:
            return alpha[self.condperiod].iloc[-1, :]

    def target_014(self, type=1):
        alpha = (self.S_FWDS_ADJCLOSE.shift(-14) - self.S_FWDS_ADJCLOSE) / self.S_FWDS_ADJCLOSE
        return alpha
        alpha[alpha >= 1.0E+38] = 1.0E+38
        alpha[alpha <= -1.0E+38] = -1.0E+38
        if type:
            return alpha[self.condperiod]
        else:
            return alpha[self.condperiod].iloc[-1, :]

    def target_028(self, type=1):
        alpha = (self.S_FWDS_ADJCLOSE.shift(-28) - self.S_FWDS_ADJCLOSE) / self.S_FWDS_ADJCLOSE
        return alpha
        alpha[alpha >= 1.0E+38] = 1.0E+38
        alpha[alpha <= -1.0E+38] = -1.0E+38
        if type:
            return alpha[self.condperiod]
        else:
            return alpha[self.condperiod].iloc[-1, :]

    def inc_01(self, type=1):
        alpha = (self.S_FWDS_ADJCLOSE - self.S_FWDS_ADJCLOSE.shift(1)) / self.S_FWDS_ADJCLOSE
        return alpha
        alpha[alpha >= 1.0E+38] = 1.0E+38
        alpha[alpha <= -1.0E+38] = -1.0E+38
        if type:
            return alpha[self.condperiod]
        else:
            return alpha[self.condperiod].iloc[-1, :]

    def inc_02(self, type=1):
        alpha = (self.S_FWDS_ADJCLOSE - self.S_FWDS_ADJCLOSE.shift(2)) / self.S_FWDS_ADJCLOSE
        return alpha
        alpha[alpha >= 1.0E+38] = 1.0E+38
        alpha[alpha <= -1.0E+38] = -1.0E+38
        if type:
            return alpha[self.condperiod]
        else:
            return alpha[self.condperiod].iloc[-1, :]

    def inc_03(self, type=1):
        alpha = (self.S_FWDS_ADJCLOSE - self.S_FWDS_ADJCLOSE.shift(3)) / self.S_FWDS_ADJCLOSE
        return alpha
        alpha[alpha >= 1.0E+38] = 1.0E+38
        alpha[alpha <= -1.0E+38] = -1.0E+38
        if type:
            return alpha[self.condperiod]
        else:
            return alpha[self.condperiod].iloc[-1, :]

    def inc_04(self, type=1):
        alpha = (self.S_FWDS_ADJCLOSE - self.S_FWDS_ADJCLOSE.shift(4)) / self.S_FWDS_ADJCLOSE
        return alpha
        alpha[alpha >= 1.0E+38] = 1.0E+38
        alpha[alpha <= -1.0E+38] = -1.0E+38
        if type:
            return alpha[self.condperiod]
        else:
            return alpha[self.condperiod].iloc[-1, :]

    def inc_05(self, type=1):
        alpha = (self.S_FWDS_ADJCLOSE - self.S_FWDS_ADJCLOSE.shift(5)) / self.S_FWDS_ADJCLOSE
        return alpha
        alpha[alpha >= 1.0E+38] = 1.0E+38
        alpha[alpha <= -1.0E+38] = -1.0E+38
        if type:
            return alpha[self.condperiod]
        else:
            return alpha[self.condperiod].iloc[-1, :]

    def inc_06(self, type=1):
        alpha = (self.S_FWDS_ADJCLOSE - self.S_FWDS_ADJCLOSE.shift(6)) / self.S_FWDS_ADJCLOSE
        return alpha
        alpha[alpha >= 1.0E+38] = 1.0E+38
        alpha[alpha <= -1.0E+38] = -1.0E+38
        if type:
            return alpha[self.condperiod]
        else:
            return alpha[self.condperiod].iloc[-1, :]

    def inc_07(self, type=1):
        alpha = (self.S_FWDS_ADJCLOSE - self.S_FWDS_ADJCLOSE.shift(7)) / self.S_FWDS_ADJCLOSE
        return alpha
        alpha[alpha >= 1.0E+38] = 1.0E+38
        alpha[alpha <= -1.0E+38] = -1.0E+38
        if type:
            return alpha[self.condperiod]
        else:
            return alpha[self.condperiod].iloc[-1, :]

    def inc_08(self, type=1):
        alpha = (self.S_FWDS_ADJCLOSE - self.S_FWDS_ADJCLOSE.shift(8)) / self.S_FWDS_ADJCLOSE
        return alpha
        alpha[alpha >= 1.0E+38] = 1.0E+38
        alpha[alpha <= -1.0E+38] = -1.0E+38
        if type:
            return alpha[self.condperiod]
        else:
            return alpha[self.condperiod].iloc[-1, :]

    def inc_09(self, type=1):
        alpha = (self.S_FWDS_ADJCLOSE - self.S_FWDS_ADJCLOSE.shift(9)) / self.S_FWDS_ADJCLOSE
        return alpha
        alpha[alpha >= 1.0E+38] = 1.0E+38
        alpha[alpha <= -1.0E+38] = -1.0E+38
        if type:
            return alpha[self.condperiod]
        else:
            return alpha[self.condperiod].iloc[-1, :]

    def inc_10(self, type=1):
        alpha = (self.S_FWDS_ADJCLOSE - self.S_FWDS_ADJCLOSE.shift(10)) / self.S_FWDS_ADJCLOSE
        return alpha
        alpha[alpha >= 1.0E+38] = 1.0E+38
        alpha[alpha <= -1.0E+38] = -1.0E+38
        if type:
            return alpha[self.condperiod]
        else:
            return alpha[self.condperiod].iloc[-1, :]

    def inc_11(self, type=1):
        alpha = (self.S_FWDS_ADJCLOSE - self.S_FWDS_ADJCLOSE.shift(11)) / self.S_FWDS_ADJCLOSE
        return alpha
        alpha[alpha >= 1.0E+38] = 1.0E+38
        alpha[alpha <= -1.0E+38] = -1.0E+38
        if type:
            return alpha[self.condperiod]
        else:
            return alpha[self.condperiod].iloc[-1, :]

    def inc_12(self, type=1):
        alpha = (self.S_FWDS_ADJCLOSE - self.S_FWDS_ADJCLOSE.shift(12)) / self.S_FWDS_ADJCLOSE
        return alpha
        alpha[alpha >= 1.0E+38] = 1.0E+38
        alpha[alpha <= -1.0E+38] = -1.0E+38
        if type:
            return alpha[self.condperiod]
        else:
            return alpha[self.condperiod].iloc[-1, :]

    def inc_13(self, type=1):
        alpha = (self.S_FWDS_ADJCLOSE - self.S_FWDS_ADJCLOSE.shift(13)) / self.S_FWDS_ADJCLOSE
        return alpha
        alpha[alpha >= 1.0E+38] = 1.0E+38
        alpha[alpha <= -1.0E+38] = -1.0E+38
        if type:
            return alpha[self.condperiod]
        else:
            return alpha[self.condperiod].iloc[-1, :]

    def inc_14(self, type=1):
        alpha = (self.S_FWDS_ADJCLOSE - self.S_FWDS_ADJCLOSE.shift(14)) / self.S_FWDS_ADJCLOSE
        return alpha
        alpha[alpha >= 1.0E+38] = 1.0E+38
        alpha[alpha <= -1.0E+38] = -1.0E+38
        if type:
            return alpha[self.condperiod]
        else:
            return alpha[self.condperiod].iloc[-1, :]

    def volume(self, type=1):
        alpha = self.S_DQ_VOLUME
        return alpha
        alpha[alpha >= 1.0E+38] = 1.0E+38
        alpha[alpha <= -1.0E+38] = -1.0E+38
        if type:
            return alpha[self.condperiod]
        else:
            return alpha[self.condperiod].iloc[-1, :]

    def amount(self, type=1):
        alpha = self.S_DQ_AMOUNT
        return alpha
        alpha[alpha >= 1.0E+38] = 1.0E+38
        alpha[alpha <= -1.0E+38] = -1.0E+38
        if type:
            return alpha[self.condperiod]
        else:
            return alpha[self.condperiod].iloc[-1, :]

    def open(self, type=1):
        alpha = self.S_FWDS_ADJOPEN
        return alpha
        alpha[alpha >= 1.0E+38] = 1.0E+38
        alpha[alpha <= -1.0E+38] = -1.0E+38
        if type:
            return alpha[self.condperiod]
        else:
            return alpha[self.condperiod].iloc[-1, :]

    def high(self, type=1):
        alpha = self.S_FWDS_ADJHIGH
        return alpha
        alpha[alpha >= 1.0E+38] = 1.0E+38
        alpha[alpha <= -1.0E+38] = -1.0E+38
        if type:
            return alpha[self.condperiod]
        else:
            return alpha[self.condperiod].iloc[-1, :]

    def low(self, type=1):
        alpha = self.S_FWDS_ADJLOW
        return alpha
        alpha[alpha >= 1.0E+38] = 1.0E+38
        alpha[alpha <= -1.0E+38] = -1.0E+38
        if type:
            return alpha[self.condperiod]
        else:
            return alpha[self.condperiod].iloc[-1, :]

    def close(self, type=1):
        alpha = self.S_FWDS_ADJCLOSE
        return alpha
        alpha[alpha >= 1.0E+38] = 1.0E+38
        alpha[alpha <= -1.0E+38] = -1.0E+38
        if type:
            return alpha[self.condperiod]
        else:
            return alpha[self.condperiod].iloc[-1, :]

    def avgprice(self, type=1):
        alpha = self.S_DQ_AVGPRICE
        return alpha
        alpha[alpha >= 1.0E+38] = 1.0E+38
        alpha[alpha <= -1.0E+38] = -1.0E+38
        if type:
            return alpha[self.condperiod]
        else:
            return alpha[self.condperiod].iloc[-1, :]
