#conda activate pytorch

for i in {0..10000}
do 
    echo "######## "$i" ########"
    python predict_day_neu_mse.py ./lr0001_day_neu_kl/model_force.pt.$i
    cp tushare_day new_pre

    python eval.py | grep 'mean'

    cat new_pre | awk '{print $2,$1,$4,$3}'  > tushare_day.v12
    #cat new_pre | awk '{print $2,$1,$4,$3}' | sort -k1,1 -k3,3gr > tushare_day.v12
    python filter_data_newest.py

    cat trade.txt | ./regr2.sh | grep  -v '#' | ./calcSP.sh

done

