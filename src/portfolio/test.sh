#!/bin/bash
array=(0.0 0.25 0.5 0.75 1.0 1.5 2.0  3.0  4.0   5  6  7  8 9 10 11 12 13 14 15 16 17 18 19 20 21 22 23 24 25 26 27 28 29 30 31 32 33 34 35 36 37 38 39 40 41 42 43 44 45 46 47 48 49 50 55 60 65 70 75 80 85 90 95 100 110 120 130 140 150 160 170 180 190 200 210 220 230 240 250 260 270 280 290 300)
#array=(310 320 330 340 350 360 370 380 390 400 410 420 430 440 450 460 470 480 490 500)
for element in ${array[@]}
do
    echo "alpha = "$element
    awk -v alpha="$element" '$8==alpha{print $0}' trade_with_risk.txt | python eval_stock.py  | ./calcSP.sh
    #awk -v alpha="$element" '{print alpha}' trade_with_risk.txt 
    echo ""
done
