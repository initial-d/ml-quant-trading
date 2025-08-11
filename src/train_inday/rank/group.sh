awk -F ' ' '$1>20220101{if(x[$1]==""){x[$1]=$2}} END {for(i in x) {print i, x[i]}}' trade.txt | sort -k1 | less
