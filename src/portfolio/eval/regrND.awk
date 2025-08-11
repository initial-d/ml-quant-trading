#!/bin/sh
#
ndpred=$1
[ ! "$ndpred" ] && ndpred=1

awk -v ndpred=$ndpred '
BEGIN{
	trade_disp_mode = 0;
	sum_gain = 1;
	trade_cur_long_gain = 0;
	trade_cur_shrt_gain = 0;
	current_date=""; 
	wait = 1; 
	trade_long_fee=0.00125;
	trade_shrt_fee=0.00170;
	stock_dsum = 0; stock_sum = 0;
	stock_up_dsum=0; stock_up_sum = 0;
	trade_cur_wrate = 0;
	trade_avg_wrate = 0; 
	trade_wday_sum = 0;
	trade_day_wrate = 1.0;

	trade_cont_false = 0;
	trade_cont_lost  = 0;
	trade_cont_btrue  = 0;
	trade_cont_wtrue  = 0;

	trade_busy_sum = 0; 
	trade_wait_sum = 0; 

	trade_proba_postive  = 0.48;
	trade_proba_base  = 0.370;
	trade_max_num = 100;

	trade_pred_uper = 0;
	trade_pred_sum  = 0;
	trade_pred_rate = 0;

	busy_buy_signal = 0.0125;

	trade_wait_size = 0;
	trade_wait_seq[0] = 0;

	trade_sel_gain["123456"] = 0;
	trade_sel_pred["123456"] = 0;
	delete trade_sel_gain["123456"];
	delete trade_sel_pred["123456"];

	stock_stat_time = 60;
	stock_pred_right[0] = 0;
	stock_pred_sum[0] = 0;
	stock_pred_gain[0] = 0;
	delete stock_pred_right[0];
	delete stock_pred_sum[0];
	delete stock_pred_gain[0];


	gain_vec[0] = 1;
	for( i = 0; i < ndpred; i++ ) {
		gain_vec[i] = 1.0 / ndpred;
	}
	nTurn = 0;
}
{

	stock_idno  = $2;
	stock_proba = $3;
	stock_gain  = $4;
	stock_gain1 = $5;
#	if( ndpred == 1 ) stock_gain = stock_gain1;
	if( current_date == $1 ) {
		if( stock_dsum < trade_max_num && stock_proba >= trade_proba_postive ) {
			if( ( stock_pred_sum[stock_idno] < stock_stat_time ) ||
				( stock_pred_sum[stock_idno] >= stock_stat_time && stock_pred_right[stock_idno] / stock_pred_sum[stock_idno] >= 0.50 ) ) {
				trade_cur_long_gain += stock_gain;
				trade_cur_shrt_gain += 1 / ( 1 + stock_gain ) - 1;
				stock_dsum++;
				stock_sum++;
				trade_sel_gain[stock_idno] = stock_gain;
				trade_sel_pred[stock_idno] = stock_proba;
				if( stock_gain > 0 ) { stock_up_dsum++; stock_up_sum++; }
			}

			{
				if( stock_gain > 0 ) stock_pred_right[stock_idno]++;
				stock_pred_sum[stock_idno]++;
				if( stock_pred_sum[stock_idno] == 1 ) {
					stock_pred_gain[stock_idno] = 1;
				}
				stock_pred_gain[stock_idno] *= ( 1 + stock_gain );
			}

		}
		if( stock_proba >= trade_proba_base ) trade_pred_uper += 1;
		trade_pred_sum += 1;
		next;
	}

	if( current_date == ""  ) {
		current_date = $1;
		if( stock_dsum < trade_max_num && stock_proba >= trade_proba_postive ) {
			if( ( stock_pred_sum[stock_idno] < stock_stat_time ) ||
				( stock_pred_sum[stock_idno] >= stock_stat_time && stock_pred_right[stock_idno] / stock_pred_sum[stock_idno] >= 0.50 ) ) {
				trade_cur_long_gain = stock_gain;
				trade_cur_shrt_gain += 1 / ( 1 + stock_gain ) - 1;
				stock_dsum = 1;
				stock_sum++;
				trade_sel_gain[stock_idno] = stock_gain;
				trade_sel_pred[stock_idno] = stock_proba;
				if( stock_gain > 0 ) { stock_up_dsum++; stock_up_sum++; }
			}
			{
				if( stock_gain > 0 ) stock_pred_right[stock_idno]++;
				stock_pred_sum[stock_idno]++;
				if( stock_pred_sum[stock_idno] == 1 ) {
					stock_pred_gain[stock_idno] = 1;
				}
				stock_pred_gain[stock_idno] *= ( 1 + stock_gain );
			}
		}
		trade_pred_sum = 1;
		trade_pred_uper = 0;
		if( stock_proba >= trade_proba_base ) trade_pred_uper = 1;
		next;
	}

	if( stock_dsum != 0 ) {
		trade_cur_long_gain = trade_cur_long_gain/stock_dsum - trade_long_fee;
		trade_cur_shrt_gain = trade_cur_shrt_gain/stock_dsum - trade_shrt_fee;
		trade_cur_wrate = stock_up_dsum / stock_dsum;
		trade_avg_wrate = stock_up_sum /  stock_sum;
		trade_pred_rate = trade_pred_uper / trade_pred_sum;
	}

	##################################################################################
	#决策时间线，这之前不能使用trade_cur_long_gain,模型数据只能使用产生的数据量和预测分数#
	##################################################################################
	wait = 0;
	nTurn++;
	{
		x = nTurn % ndpred;
		gain_vec[x] *= ( 1 + trade_cur_long_gain );
		sum_gain = 0;
		for( i = 0; i < ndpred; i++ ) sum_gain += gain_vec[i];

		if( stock_dsum > 0 ) trade_busy_sum++;
		if( trade_cur_long_gain > 0 ) {
			trade_wday_sum++;
			trade_cont_false = 0;
			trade_cont_btrue++;
		}
		else if( trade_cur_long_gain < 0 ) {
			trade_cont_false++;
			trade_cont_btrue = 0;
		}
		if( trade_busy_sum > 0 )
		trade_day_wrate = trade_wday_sum / trade_busy_sum;
	}

	if( trade_disp_mode == 0 ){
		printf( "%s\t%d\t%g\t%g\t%g\t%g\t%g\tDAYWIN[%g%%]\t[[%s%s:%d]]\n",
				current_date, stock_dsum, sum_gain, trade_cur_long_gain, trade_cur_shrt_gain,
				trade_cur_wrate, trade_pred_rate, trade_day_wrate * 100,
				( wait > 0 ) ? "WAIT":"BUSY", ( trade_cur_long_gain > 0 ) ? "+":"-", 
				( wait > 0 ) ? trade_wait_size : (trade_cont_btrue ? trade_cont_btrue : trade_cont_false ));
	}

	for( stk in trade_sel_gain ){
		if( wait <= 0 && trade_disp_mode == 1) {
			printf("%s %s %g %g\n", current_date, stk, trade_sel_gain[stk], trade_sel_pred[stk]);
		}
		delete trade_sel_gain[stk];
		delete trade_sel_pred[stk];
	}

	#开始处理新一天的数据
	{
		current_date = $1;
		stock_up_dsum = 0;
		stock_dsum = 0;
		trade_cur_long_gain = 0;
		trade_cur_shrt_gain = 0;

		if( stock_dsum < trade_max_num && stock_proba >= trade_proba_postive ) {
			if( ( stock_pred_sum[stock_idno] < stock_stat_time ) ||
				( stock_pred_sum[stock_idno] >= stock_stat_time && stock_pred_right[stock_idno] / stock_pred_sum[stock_idno] >= 0.50 ) ) {
				trade_cur_long_gain = stock_gain;
				trade_cur_shrt_gain += 1 / ( 1 + stock_gain ) - 1;
				stock_dsum = 1;
				stock_sum++;
				trade_sel_gain[stock_idno] = stock_gain;
				trade_sel_pred[stock_idno] = stcok_pred;
				if( stock_gain > 0 ) { stock_up_dsum++; stock_up_sum++; }
			}
			{
				if( stock_gain > 0 ) stock_pred_right[stock_idno]++;
				stock_pred_sum[stock_idno]++;
				if( stock_pred_sum[stock_idno] == 1 ) {
					stock_pred_gain[stock_idno] = 1;
				}
				stock_pred_gain[stock_idno] *= ( 1 + stock_gain );
			}
		}
		trade_pred_sum = 1;
		trade_pred_uper = 0;
		if( stock_proba >= trade_proba_base ) trade_pred_uper = 1;
	}
}
END {
	if( stock_dsum ) {
		if( stock_dsum != 0 ) {
			trade_cur_long_gain = trade_cur_long_gain/stock_dsum - trade_long_fee;
			trade_cur_shrt_gain = trade_cur_shrt_gain/stock_dsum - trade_shrt_fee;
			trade_cur_wrate = stock_up_dsum / stock_dsum;
			trade_avg_wrate = stock_up_sum / stock_sum;
			trade_pred_rate = trade_pred_uper / trade_pred_sum;
		}

		{
			sum_gain *= ( 1 + trade_cur_long_gain );
			trade_busy_sum++;
		}
		if( trade_disp_mode == 0 ){
			printf( "%s\t%d\t%g\t%g\t%g\t%g\t%g\tDAYWIN[%g%%]\t[[%s%s:%d]]\n",
					   current_date, stock_dsum, sum_gain, trade_cur_long_gain, trade_cur_shrt_gain,
					   trade_cur_wrate, trade_pred_rate, trade_day_wrate * 100,
					   ( wait > 0 ) ? "WAIT":"BUSY", ( trade_cur_long_gain > 0 ) ? "+":"-", 
					   ( wait > 0 ) ? trade_wait_size : (trade_cont_btrue ? trade_cont_btrue : trade_cont_false ));
		}
	}
	if( trade_disp_mode == 0 ){
		printf("END 交易 %4d 等待 %4d 收益率 %g 天级胜率 %g%% 个股胜率 %g%%\n",
				trade_busy_sum, trade_wait_sum, sum_gain, trade_day_wrate * 100, trade_avg_wrate * 100 );
	}
}' 
