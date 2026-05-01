#!/bin/bash
#for i in {39..165}
for i in {44..3000}
do 
    #cat ../lr0001/tushare_day.$i | grep " 11:29	" > new_pre
    #echo "######## "$i" ########"
    #python ../eval.py | grep "mean"
    
    python predict_regression_newest_tmp.py lr0001/model_force.pt.$i
    #cp tushare_day_newest new_pre
    echo $i"###########"
    #python ./eval.py | grep "mean"
    python halfday_trade.py
    awk '{print $6"\t"$7"\t"$2"\t"$21}' trade.txt | ./regr2.sh | grep -v '#' | ./calcSP.sh
done
