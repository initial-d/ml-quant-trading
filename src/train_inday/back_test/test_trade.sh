#!/bin/bash
for i in {565..566}
do
    for j in {0..14}
    do 
        cat ./lr0001_self/tushare_day.$i.$j > new_pre
        echo "######## "$i" ######## "$j
        python eval.py | grep "mean"
        
        python trade.py ./lr0001_self/tushare_day.$i.$j
        echo $i" "$j" base"
        awk '$1>=20210110&&$1<20210830{print $1"\t"$2"\t"$3"\t"$4}' trade.txt | ./regr2.sh | grep -v "#" | ./calcSP.sh
        echo $i" "$j" test"
        awk '$1>=20210110&&$1<20210830{print $1"\t"$2"\t"$3"\t"$22}' trade.txt | ./regr2.sh | grep -v "#" | ./calcSP.sh
    done
done
