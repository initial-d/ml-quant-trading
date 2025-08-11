#!/bin/bash
#for i in {39..165}
for i in {113..2242}
do 
    #cat ../lr0001/tushare_day.$i | grep " 11:29	" > new_pre
    #echo "######## "$i" ########"
    #python ../eval.py | grep "mean"
    
    #python predict_regression_base.py lr0001_kl/model_force.pt.$i
    #python predict_regression_online_base.py lr0001_kl_concat/model_force.pt.$i
    python predict_3part.py lr0001_mse_3concat/model_force.pt.$i
    cp tushare_day new_pre
    echo $i
    python ./eval.py | grep "mean"
    #python filter_data_half_classify.py
    #echo $i" base"
    #awk '$1>=20211101&&$1<=20220125{print $1"\t"$2"\t"$3"\t"$21}' trade_half.txt | ./regr2.sh | grep -v '#' 
    #echo $i" test"
    #awk '$1>=20211101&&$1<=20220125{print $1"\t"$2"\t"$3"\t"$22}' trade_half.txt | ./regr2.sh | grep -v '#' 
done
