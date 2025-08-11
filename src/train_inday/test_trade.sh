#!/bin/bash
#for i in {39..165}
for i in {18..91}
do 
    cat ./lr0001/tushare_day.$i | grep " 11:30	" > new_pre
    echo "######## "$i" ########"
    python eval.py | grep "mean"
    
    python trade.py ./lr0001/tushare_day.$i
    echo $i" "$j" base"
    awk '$1>=20210110&&$1<20210830{print $1"\t"$2"\t"$3"\t"$4}' trade.txt | ./regr2.sh | grep -v "#" | ./calcSP.sh
    echo $i" "$j" test"
    awk '$1>=20210110&&$1<20210830{print $1"\t"$2"\t"$3"\t"$22}' trade.txt | ./regr2.sh | grep -v "#" | ./calcSP.sh
done
