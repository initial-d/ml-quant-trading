#!/bin/bash
for i in {0..200}
do 
    for j in {7..7}
    do 
        cat ./lr0001_self/tushare_day.$i.$j > new_pre
        echo "######## "$i" ########"
        python eval.py | grep "mean"
        #python trade.py $j
    done
done
