#conda activate pytorch

for i in {12001..16000}
do 
    echo "######## "$i" ########"
    #python predict_day_neu_mse.py ./lr0001_day_neu_nozscore/model_force.pt.$i
    #python predict_day_neu_mse.py ./lr00001_day_neu/model_force.pt.$i
    #python predict_day_neu_mse.py ./lr0001_day_neu_v2/model_force.pt.$i
    #python predict_day_neu_rank_mse.py ./lr0001_day_neu_rank_relu_adj_v13/model_force.pt.$i
    python predict_day_neu_rank_mse.py ./lr0001_day_neu_rank_relu_shp_v13/model_force.pt.$i
    cp tushare_day new_pre
    #python eval.py | grep 'mean'

    #cat new_pre | awk '{print $2,$1,$4,$3}' | sort -k1,1 -k3,3gr > tushare_day.v12

    cat new_pre | awk '{print $2,$1,$4,$3}'  > tushare_day.v12
    #python filter_data.py
    python filter_data_newest.py

    cat trade.txt | ./regr2.sh | grep  -v '#' 
    cat trade.txt | ./regr2.sh | grep  -v '#' | ./calcSP.sh

done

