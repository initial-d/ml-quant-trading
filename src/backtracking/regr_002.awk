#!/bin/sh
#
awk '{ if( $3 >= 0.003 && $3 <= 1.999) print $1,$2,$4,$3;}'|
awk '
function days_of_date( date )
{
	return substr( date, 1, 4 ) * 365 + month_list[substr( date, 5 , 2 ) - 1] + substr( date, 7, 2 );
}

function create_watch_lst( seq, size, swatch,        i,k,val,uniqer)
{
	k = 0;
	for( i in swatch ) delete swatch[i];
	for( i = 0; i < size; i++ ) {
		val = seq[i];
		if( val in uniqer ) continue;
		uniqer[val] = 1;
		swatch[k++] = val;
	}
}

function search_margin_best( watch_list, seq_data, seq_gain, seq_size, margin,        best_val,max_gain,max_op,key,val,op,gain,i,avg_rate)
{
	best_val = 0;
	max_gain = 0;
	max_op   = 0;
	for( key in watch_list ) {
		val = watch_list[key];
		if( val >= margin ) continue;

		op = 0; gain = 1;
		for( i = 0; i < seq_size - 1; i++ ) {
			if( seq_data[i] < val ) continue;
			if( seq_data[i] >= margin ) continue;
			gain *= ( 1 + seq_gain[i+1] );
			op++;
		}
		if( gain > 1 && gain > max_gain ) {
			max_gain = gain;
			best_val = val;
			max_op = op;
		}
	}
	if( max_gain > 1 ) {
		if( margin >= 1 ) return best_val;
		avg_rate = exp( log( max_gain) / max_op );
		if( avg_rate >= 1.001 && max_op >= 20 && max_gain >= 1.01 ) {
			print best_val, margin, max_gain,max_op,avg_rate;
			return best_val;
		}
	}
	return 1;
}

function search_span_best ( watch_list, seq_data, seq_gain, seq_size,best_span,        margin,size,swatch,i,bst)
{
	margin  = 1.0;
	for( i in best_span ) delete best_span[i];
	size = asort( watch_list, swatch, "@val_num_desc" );
	for( i = 1; i <= size; i++ ) {
		bst = search_margin_best( swatch, seq_data, seq_gain, seq_size, margin );
		if( bst < 1 ) {
			best_span[bst] = margin;
			margin = bst;
		}
		else {
			margin = swatch[i];
		}

		#目前只算一次拉倒
		break;

		while( i <= size ) {
			if( swatch[i] >= margin ) {
				i++; continue;
			}
			break;
		}
		if( i <= size ) margin = swatch[i];
	}
}
BEGIN{
	trade_disp_mode = 0;
	run_policy_level = 9;
	sum_gain = 1;
	cur_d_gain = 0; last_d_gain = 0; trade_num = 0; current_date=""; 
	wait = 1; try_trade = 0; try_rate1 = 0.99; try_rate2 = 0.01;
	trade_fee_rate=0.0011;
	trade_sum = 0;
	UP_num=0; DN_num=0; UP_sum = 0; DN_sum = 0;
	trade_cur_win_rate = 0;
	trade_avg_win_rate = 0; 
	trade_win_day_num = 0;
	trade_win_day_rate = 1.0;

	trade_cont_false = 0;
	trade_cont_miss  = 0;

	trade_busy_sum = 0; 
	trade_wait_sum = 0; 
	trade_cont_up  = 0;

	trade_avg_num = 0;
	trade_max_num =50;
	trade_spec_min_num = 6;

	trade_spec_win_num = 1;
	trade_spec_trade_num = 1;
	trade_spec_win_rate = 1;

	gain_span_signal[0.0135] = 1;
	rate_span_signal[0.545455] = 1;

	wait_dn2up_signal = -0.00147452;
	wait_reverse_signal = -0.0457962;
	busy_up3_signal = 0.0134968;
	busy_up4_signal = 0.0134018;
	busy_rate_signal = 0.545455;

	trade_wait_size = 0;
	trade_wait_seq[0] = 0;

	trade_sel_gain["123456"] = 0;
	trade_sel_pred["123456"] = 0;
	delete trade_sel_gain["123456"];
	delete trade_sel_pred["123456"];

   	trade_seq_size = 0;
   	trade_seq_gain[0] = 0;
   	trade_seq_rate[0] = 0;
	trade_calc_window = 100;

	month_list[0] = 0;
	month_list[1] = 31;
	month_list[2] = 59;
	month_list[3] = 90;
	month_list[4] = 120;
	month_list[5] = 151;
	month_list[6] = 181;
	month_list[7] = 212;
	month_list[8] = 243;
	month_list[9] = 273;
	month_list[10] = 304;
	month_list[11] = 334;
}
{
	if( current_date == $1 ) {
		if( trade_num < trade_max_num ) {
			cur_d_gain += $3;
			trade_num++;
			trade_sum++;
			trade_sel_gain[$2] = $3;
			trade_sel_pred[$2] = $4;
			if( $3 > 0 ) { UP_num++; UP_sum++; }
			else { DN_num++; DN_sum++; }
		}
		next;
	}

	if( current_date == ""  ) {
		current_date = $1;
		cur_d_gain = $3;
		trade_num = 1;
		trade_sum++;
		trade_sel_gain[$2] = $3;
		trade_sel_pred[$2] = $4;

		if( $3 > 0 ) { UP_num++; UP_sum++; }
		else { DN_num++; DN_sum++; }
		next;
	}

	cur_d_gain = cur_d_gain/trade_num - trade_fee_rate;
	trade_cur_win_rate = UP_num / (UP_num + DN_num);
	trade_avg_win_rate = UP_sum / (UP_sum + DN_sum);
	trade_avg_num = trade_sum / ( trade_seq_size + 1 );

	trade_seq_gain[trade_seq_size] = cur_d_gain;
	trade_seq_rate[trade_seq_size] = trade_cur_win_rate;;
	trade_seq_size++;

	#长假模式居然赚钱
	if( run_policy_level >= 3 ) {
		day_span = days_of_date( $1 ) - days_of_date( current_date );
		if( day_span >= 8 ) {
			if( last_d_gain > 0 ) wait = 0;
			if( last_d_gain < 0 && last_d_gain > -0.01 && trade_last_win_rate >= 0.495 ) wait = 0;
		}

		if( day_span < 0 || day_span > 30 ) wait = 1;
	}

	#比较奇怪，小于这个阈值胜率很高
	if( run_policy_level >= 4 ) {
		if( trade_num < trade_avg_num * 0.50 && trade_spec_win_rate >= 0.50 ) {
			if( wait > 0 ) try_trade = 1;
			if( trade_num < trade_spec_min_num ) try_trade = 2;
			wait = 0;
		}
	}

	#特殊日子避过
	if( run_policy_level >= 5 ) {
		monday=substr( current_date,5,4 );
		switch( monday ) {
			case "0106":
			case "0108":
			case "0125":
			case "0226":
			case "0430":
			case "0726":
			case "0828":
			wait = 1;
			break;
		}
	}

	#决策时间线，这之前不能使用cur_d_gain,模型数据只能使用产生的数据量和预测分数
	if( wait > 0 ) {
		trade_wait_sum++;
		if( cur_d_gain > 0 ) trade_cont_miss++;
		else trade_cont_miss = 0;
	}
	else {

		#连续出错，不信任,改为等待,加入自身怀疑因素
		if( trade_cont_false >= 6 ) wait = 1;

		if( wait == 0 ){
			switch( try_trade ) {
				case 2:
				sum_gain *= ( (1-try_rate2) + try_rate2 * ( 1 + cur_d_gain ) );
				break;
				case 1:
				sum_gain *= ( (1-try_rate1) + try_rate1 * ( 1 + cur_d_gain ) );
				break;
				case 0:
				default:
				sum_gain *= ( 1 + cur_d_gain );
				break;
			}
			trade_busy_sum++;
			if( cur_d_gain > 0 ) trade_win_day_num++;
			trade_win_day_rate = trade_win_day_num / trade_busy_sum;
		}

		if( cur_d_gain > 0 ) trade_cont_false = 0;
		else trade_cont_false++;
	}

	#统计小于指定数量的交易,以便后续是否继续交易
	if( trade_num < trade_avg_num * 0.50 ) {
		trade_spec_win_num += ( cur_d_gain > 0 ) ? 1 : 0;
		trade_spec_trade_num++;
		trade_spec_win_rate = trade_spec_win_num / trade_spec_trade_num;
	}

	if( trade_disp_mode == 0 ){
		printf( "%s\t%d\t%g\t%g\t%g\t%g\tDAYWIN[%g%%]\t[[%s%s:%d]]\n",
				current_date, trade_num, sum_gain, cur_d_gain,
				trade_cur_win_rate, trade_avg_win_rate, trade_win_day_rate * 100,
				( wait > 0 ) ? "WAIT":"BUSY", ( cur_d_gain > 0 ) ? "+":"-", ( wait > 0 ) ? trade_wait_size : try_trade );
	}

	for( stk in trade_sel_gain ){
		if( wait <= 0 && trade_disp_mode == 1) {
			printf("%s %s %g %g\n", current_date, stk, trade_sel_gain[stk], trade_sel_pred[stk]);
		}
		delete trade_sel_gain[stk];
		delete trade_sel_pred[stk];
	}

	#归零try_trade
	#换成last_d_gain
	try_trade = 0;
	last_d_gain = cur_d_gain;
	trade_last_win_rate = trade_cur_win_rate;

	#计算新的参数CALC
	if( trade_seq_size >= 100 && trade_seq_size % 1 == 0 ) {
		if( wait ) {
			switch( trade_cont_miss ) {
			case 0:
			case 1:
				break;
			case 2:
			default:
				trade_calc_window /= 2;
				break;
			}
		}
		else {
			switch( trade_cont_false ) {
			case 0:
			case 1:
				break;
			case 2:
			default:
				trade_calc_window *= 2;
				break;
			}
		}
		trade_calc_window = int( trade_calc_window );
		if( trade_calc_window < 100 ) trade_calc_window = 100;
		if( trade_calc_window > 800 ) trade_calc_window = 800;

		begin = trade_seq_size - trade_calc_window;
		if( begin < 0 ) begin  = 0;
		k = 0;
		for( i = begin; i < trade_seq_size; i++ ) {
			seq_gain_select[k] = trade_seq_gain[i];
			seq_rate_select[k] = trade_seq_rate[i];
			k++;
		}
		create_watch_lst( seq_gain_select, k, watch_list );
		search_span_best( watch_list, seq_gain_select, seq_gain_select, k, best_param);
		if( length( best_param ) ) {
			delete gain_span_signal;
			for( i in best_param ) {
				gain_span_signal[i] = best_param[i];
			}
		}
#		create_watch_lst( seq_rate_select, k, watch_list );
#		search_span_best( watch_list, seq_rate_select, seq_gain_select, k, best_param);
#		if( length( best_param ) ) {
#			delete rate_span_signal;
#			for( i in best_param ) {
#				rate_span_signal[i] = best_param[i];
#			}
#		}
	}

	#####################################
	#判断下一天是否交易
	wait = 1;
	#以下都是收益策略
	#####################################

	#各种统计开始
	trade_wait_seq[trade_wait_size++] = last_d_gain;

	wait_cont_up = 0;
	for( i = trade_wait_size - 1; i >= 0; i-- ) {
		if( trade_wait_seq[i] < 0 ) break;
		wait_cont_up++;
	}
	wait_cont_dn = 0;
	for( i = trade_wait_size - 1; i >= 0; i-- ) {
		if( trade_wait_seq[i] > 0 ) {
			if( wait_cont_dn != 0) break;
			continue;
		}
		wait_cont_dn++;
	}

	wait_span_dn = 0;
	wait_span_up = 0;
	wait_gain_up = 0;
	wait_gain_dn = 0;
	wait_gain_sum= 0;
	wait_gain_avg= 0;
	for( i = trade_wait_size - 1; i >= 0; i-- ) {
		if( trade_wait_seq[i] < 0 ) {
			wait_span_dn++;
			wat_gain_dn += trade_wait_seq[i];
		}
		else {
			wait_span_up++;
			wait_gain_up += trade_wait_seq[i];
		}
		wait_gain_sum += trade_wait_seq[i];
	}

	if( trade_wait_size  ) {
		wait_gain_avg = wait_gain_sum / trade_wait_size;
	}

	if( wait_span_up ) {
		wait_gain_up = wait_gain_up / wait_span_up;
	}

	if( wait_span_dn ) {
		wait_gain_dn = wait_gain_dn / wait_span_dn;
	}

	wait_beg_up = 0;
	for( i = 0; i < trade_wait_size; i++ ) {
		if( trade_wait_seq[i] < 0 ) break;
		wait_beg_up++;
	}

	wait_beg_dn = 0;
	for( i = 0; i < trade_wait_size; i++ ) {
		if( trade_wait_seq[i] > 0 ) break;
		wait_beg_dn++;
	}

	######################################
	##以下所有信息可用
	##最简单基本策略
	######################################
	if( run_policy_level >= 0 ) {
		for( i in gain_span_signal ) {
			if( last_d_gain >= i && last_d_gain < gain_span_signal[i] ) {
				wait = 0;
				break;
			}
		}

		#for( i in rate_span_signal ) {
			#if( trade_last_win_rate >= i && trade_last_win_rate < rate_span_signal[i] ) {
				#wait = 0;
			#break;
			#}
		#}

		#if( last_d_gain < 0 && last_d_gain >= busy_dn2up_signal ) wait = 0;
		#if( last_d_gain < 0 && last_d_gain <= wait_reverse_signal ) wait = 0;
	}


	#通用的一些召回策略
	if( run_policy_level >= 1 ) {
		if( last_d_gain >= 0.016 && trade_last_win_rate >= 0.72 ) wait = 0;
		if( last_d_gain >= 0.014 && trade_last_win_rate >= 0.75 ) wait = 0;
		if( last_d_gain >= 0.013 && trade_last_win_rate >= 0.80 ) wait = 0;
		if( last_d_gain >= 0.012 && trade_last_win_rate >= 0.81 ) wait = 0;
		if( last_d_gain >= 0.001 && trade_last_win_rate >= 0.81 ) wait = 0;
		if( last_d_gain >= 0.000 && trade_last_win_rate >= 0.85 ) wait = 0;

		if( trade_wait_size >= 3 && last_d_gain >= 0.014 && trade_last_win_rate >= 0.72 ) wait = 0;
		if( trade_wait_size >= 3 && last_d_gain >= 0.013 && trade_last_win_rate >= 0.72 ) wait = 0;
		if( trade_wait_size >= 3 && last_d_gain >= 0.012 && trade_last_win_rate >= 0.72 ) wait = 0;
		if( trade_wait_size >= 3 && last_d_gain >= 0.011 && trade_last_win_rate >= 0.78 ) wait = 0;

		if( trade_wait_size >= 4 && wait_gain_avg > 0 && last_d_gain > 0.008 ) wait = 0;
		if( trade_wait_size >= 5 && wait_gain_avg > 0 && last_d_gain > -0.005) wait = 0;
		if( trade_wait_size >=12 && wait_cont_up >= 3 && last_d_gain > 0.001 ) wait = 0;
		if( trade_wait_size >=15 && wait_cont_up >= 1 && last_d_gain > 0.001 ) wait = 0;

		if( last_d_gain < 0 && last_d_gain > -0.001 && trade_last_win_rate >= 0.495 ) wait = 0;
		if( wait_cont_up >= 4 && trade_last_win_rate >= 0.41 ) wait = 0;

		if( wait_cont_dn >= 5 && wait_cont_up == 1 && trade_last_win_rate >= 0.46 ) wait = 0;

		if( wait_span_dn >= 6 && last_d_gain >= 0.0130 ) wait = 0;
		if( wait_span_dn >= 7 && last_d_gain >= 0.0130 ) wait = 0;
		if( wait_span_dn >= 8 && last_d_gain >= 0.0130 ) wait = 0;
		if( wait_span_dn >= 9 && last_d_gain >= -0.005 ) wait = 0;
		if( wait_span_dn >=11 && last_d_gain >= -0.008 ) wait = 0;
	}

	#各种奇怪模式判断
	if( run_policy_level >= 2 ) {
		switch( trade_wait_size ) {
			case 1:
			break;
			case 2:
			#00
			if( wait_cont_dn == 2 && wait_cont_up == 0 && trade_last_win_rate >= 0.491 ) wait = 0;
			#01
			if( wait_cont_dn == 1 && wait_cont_up == 1 && trade_last_win_rate >= 0.85 ) wait = 0;
			#10
			if( wait_cont_dn == 1 && wait_beg_up == 1 && trade_last_win_rate >= 0.49 ) wait = 0;
			#11
			if( wait_cont_up == 2 && trade_last_win_rate >= 0.65 ) wait = 0;
			break;
			case 3:
			#000
			if( wait_cont_dn == 3 && wait_cont_up == 0 && trade_last_win_rate >= 0.496 ) wait = 0;
			#100
			if( wait_cont_dn == 2 && wait_beg_up == 1 && trade_last_win_rate >= 0.45 ) wait = 0;
			#110
			if( wait_cont_dn == 1 && wait_beg_up == 2 && trade_last_win_rate >= 0.495 ) wait = 0;
			#111
			if( wait_cont_up == 3 && trade_last_win_rate >= 0.70 ) wait = 0;
			#011
			if( wait_cont_up == 2 && trade_last_win_rate >= 0.70 ) wait = 0;
			#001
			if( wait_cont_dn == 2 && wait_cont_up == 1 && trade_last_win_rate >= 0.80 ) wait = 0;
			#101
			if( wait_cont_dn == 1 && wait_cont_up == 1 && trade_last_win_rate >= 0.78 ) wait = 0;
			break;
			case 4:
			#0000
			if( wait_cont_dn == 4 && wait_cont_up == 0 && trade_last_win_rate >= 0.49 ) wait = 0;
			#0001
			if( wait_cont_dn == 3 && wait_cont_up == 1 && trade_last_win_rate >= 0.70 ) wait = 0;
			#0010
			if( wait_cont_dn == 2 && wait_cont_dn == 1 && trade_last_win_rate >= 0.495 ) wait = 0;
			#0011
			if( wait_cont_dn == 2 && wait_cont_up == 2 && trade_last_win_rate >= 0.40 ) wait = 0;
			#0100
			if( wait_cont_dn == 2 && wait_beg_dn == 1 && trade_last_win_rate >= 0.45 ) wait = 0;
			#0101
			if( wait_cont_dn == 1 && wait_cont_up == 1 && wait_beg_dn == 1 && trade_last_win_rate >= 0.80 ) wait = 0;
			#0110
			if( wait_cont_dn == 1 && wait_cont_up == 0 && wait_beg_dn == 1 && trade_last_win_rate >= 0.49 ) wait = 0;
			#0111
			if( wait_cont_dn == 1 && wait_cont_up == 3 && trade_last_win_rate >= 0.49 ) wait = 0;
			#1000
			if( wait_cont_dn == 3 && wait_beg_up == 1 && trade_last_win_rate >= 0.47 ) wait = 0;
			#1001
			if( wait_beg_up == 1 && wait_cont_dn == 2 && wait_cont_up == 1 && trade_last_win_rate >= 0.70 ) wait = 0;
			#1010
			if( wait_beg_up == 1 && wait_cont_dn == 1 && wait_cont_up == 0 && trade_last_win_rate >= 0.42 ) wait = 0;
			#1011
			if( wait_beg_up == 1 && wait_cont_up == 2 && trade_last_win_rate >= 0.65 ) wait = 0;
			#1100
			if( wait_beg_up == 2 && wait_cont_dn == 2 && trade_last_win_rate >= 0.496 ) wait = 0;
			#1101
			if( wait_beg_up == 2 && wait_cont_dn == 1 && wait_cont_up == 1 && trade_last_win_rate >= 0.40 ) wait = 0;
			#1110
			if( wait_beg_up == 3 && wait_cont_dn == 1 && wait_cont_up == 0 && trade_last_win_rate >= 0.49 ) wait = 0;
			#1111
			if( wait_cont_up == 4 && trade_last_win_rate >= 0.50 ) wait = 0;

			break;
			case 5:
			#00000
			if( wait_cont_dn == 5 && wait_cont_up == 0 && trade_last_win_rate >= 0.45 ) wait = 0;
			#00001
			if( wait_cont_dn == 4 && wait_cont_up == 1 && trade_last_win_rate >= 0.45 ) wait = 0;
			#00010
			if( wait_beg_dn == 3 && wait_cont_dn == 1 && trade_last_win_rate >= 0.49 ) wait = 0;
			#00011
			if( wait_cont_dn == 3 && wait_cont_up == 2 && trade_last_win_rate >= 0.70 ) wait = 0;
			#00100
			if( wait_beg_dn == 2 && wait_cont_dn == 2 && trade_last_win_rate >= 0.49 ) wait = 0;
			#00101
			if( wait_cont_dn == 1 && wait_cont_up == 1 && wait_beg_dn == 2 && trade_last_win_rate >= 0.48 ) wait = 0;
			#00110
			if( wait_beg_dn == 2 && wait_cont_dn == 1 && trade_last_win_rate >= 0.49 ) wait = 0;
			#00111
			if( wait_cont_dn == 2 && wait_cont_up == 3 && trade_last_win_rate >= 0.50 ) wait = 0;
			#01000
			if( wait_cont_dn == 3 && wait_beg_dn == 1 && trade_last_win_rate >= 0.40 ) wait = 0;
			#01001
			if( wait_cont_dn == 2 && wait_cont_up == 1 && wait_beg_dn == 1 && trade_last_win_rate >= 0.80 ) wait = 0;
			#01010
			if( wait_beg_dn == 1 && wait_cont_dn == 1 && trade_wait_seq[2] < 0 && trade_last_win_rate >= 0.49 ) wait = 0;
			#01011
			if( wait_cont_dn == 1 && wait_cont_up == 2 && wait_beg_dn == 1 && trade_last_win_rate >= 0.60 ) wait = 0;
			#01100
			if( wait_cont_dn == 2 && wait_cont_up == 0 && wait_beg_dn == 1 && trade_last_win_rate >= 0.49 ) wait = 0;
			#01101
			if( wait_cont_dn == 1 && wait_cont_up == 1 && wait_beg_dn == 1 && trade_last_win_rate >= 0.51 ) wait = 0;
			#01110
			if( wait_cont_dn == 1 && wait_beg_dn == 1 && trade_wait_seq[2] > 0 && trade_last_win_rate >= 0.49 ) wait = 0;
			#01111
			if( wait_cont_dn == 1 && wait_cont_up == 4 && trade_last_win_rate >= 0.50 ) wait = 0;
			#10000
			if( wait_beg_up == 1 && wait_cont_dn == 4 && wait_cont_up == 0 && trade_last_win_rate >= 0.25 ) wait = 0;
			#10001
			if( wait_beg_up == 1 && wait_cont_dn == 3 && wait_cont_up == 1 && trade_last_win_rate >= 0.80 ) wait = 0;
			#10010
			if( wait_beg_up == 1 && wait_cont_dn == 1 && trade_wait_seq[2] < 0 && trade_last_win_rate >= 0.35 ) wait = 0;
			#10011
			if( wait_beg_up == 1 && wait_cont_dn == 2 && wait_cont_up == 2 && trade_last_win_rate >= 0.80 ) wait = 0;
			#10100
			if( wait_beg_up == 1 && wait_cont_dn == 2 && wait_cont_up == 0 && trade_last_win_rate >= 0.49 ) wait = 0;
			#10101
			if( wait_beg_up == 1 && wait_cont_dn == 1 && wait_cont_up == 1 && trade_last_win_rate >= 0.52 ) wait = 0;
			#10110
			if( wait_beg_up == 1 && wait_cont_dn == 1 && trade_wait_seq[2] > 0 && trade_last_win_rate >= 0.40 ) wait = 0;
			#10111
			if( wait_beg_up == 1 && wait_cont_dn == 1 && wait_cont_up == 3 && trade_last_win_rate >= 0.70 ) wait = 0;
			#11000
			if( wait_beg_up == 2 && wait_cont_dn == 3 && wait_cont_up == 0 && trade_last_win_rate >= 0.49 ) wait = 0;
			#11001
			if( wait_beg_up == 2 && wait_cont_dn == 2 && wait_cont_up == 1 && trade_last_win_rate >= 0.65 ) wait = 0;
			#11010
			if( wait_beg_up == 2 && wait_cont_dn == 1 && wait_cont_up == 0 && trade_last_win_rate >= 0.51 ) wait = 0;
			#11011
			if( wait_beg_up == 2 && wait_cont_dn == 1 && wait_cont_up == 2 && trade_last_win_rate >= 0.51 ) wait = 0;
			#11100
			if( wait_beg_up == 3 && wait_cont_dn == 2 && wait_cont_up == 0 && trade_last_win_rate >= 0.32 ) wait = 0;
			#11101
			if( wait_beg_up == 3 && wait_cont_dn == 1 && wait_cont_up == 1 && trade_last_win_rate >= 0.80 ) wait = 0;
			#11110
			if( wait_beg_up == 4 && wait_cont_dn == 1 && trade_last_win_rate >= 0.24 ) wait = 0;
			#11111
			if( wait_cont_dn == 0 && wait_cont_up == 5 && trade_last_win_rate >= 0.50 ) wait = 0;
			break;
			case 6:
			#000000
			if( wait_cont_dn == 6 && wait_cont_up == 0 && trade_last_win_rate >= 0.51 ) wait = 0;
			#000001
			if( wait_cont_dn == 5 && wait_cont_up == 1 && trade_last_win_rate >= 0.45 ) wait = 0;
			#000010
			#000011
			#if( wait_cont_dn == 4 && wait_cont_up == 2 && trade_last_win_rate >= 0.65 ) wait = 0;
			#000100
			#000101
			#if( wait_cont_dn == 1 && wait_cont_up == 1 && wait_beg_dn == 3 && trade_last_win_rate >= 0.60 ) wait = 0;
			#000110
			#000111
			#if( wait_cont_dn == 3 && wait_cont_up == 3 && trade_last_win_rate >= 0.65 ) wait = 0;
			#001000
			if( wait_cont_dn == 3 && wait_beg_dn == 2 && trade_last_win_rate >= 0.25 ) wait = 0;
			#001001
			#if( wait_cont_dn == 2 && wait_cont_up == 1 && wait_beg_dn == 2 && trade_last_win_rate >= 0.65 ) wait = 0;
			#001010
			#001011
			#001100
			if( wait_cont_dn == 2 && wait_cont_up == 0 && wait_beg_dn == 2 && trade_last_win_rate >= 0.41 ) wait = 0;
			#001101
			#if( wait_cont_dn == 1 && wait_cont_up == 1 && wait_beg_dn == 2 && trade_last_win_rate >= 0.60 ) wait = 0;
			#001110
			#001111
			#if( wait_cont_dn == 2 && wait_cont_up == 4 && trade_last_win_rate >= 0.65 ) wait = 0;
			#010000
			if( wait_cont_dn == 4 && wait_cont_up == 0 && wait_beg_dn == 1 && trade_last_win_rate >= 0.48 ) wait = 0;
			#010001
			#if( wait_cont_dn == 3 && wait_cont_up == 1 && wait_beg_dn == 1 && trade_last_win_rate >= 0.70 ) wait = 0;
			#010010
			#010011
			#if( wait_cont_dn == 2 && wait_cont_up == 2 && wait_beg_dn == 1 && trade_last_win_rate >= 0.60 ) wait = 0;
			#010100
			#010101
			#if( wait_cont_dn == 1 && wait_cont_up == 1 && wait_beg_dn == 1 && trade_wait_seq[2] < 0 && trade_last_win_rate >= 0.60 ) wait = 0;
			#010110
			#010111
			#if( wait_cont_dn == 1 && wait_cont_up == 3 && wait_beg_dn == 1 && trade_last_win_rate >= 0.60 ) wait = 0;
			#011000
			#011001
			if( wait_cont_dn == 2 && wait_cont_up == 1 && wait_beg_dn == 1 && trade_last_win_rate >= 0.60 ) wait = 0;
			#011010
			#011011
			#if( wait_cont_dn == 1 && wait_cont_up == 2 && wait_beg_dn == 1 && trade_last_win_rate >= 0.70 ) wait = 0;
			#011100
			#011101
			#if( wait_cont_dn == 1 && wait_cont_up == 1 && wait_beg_dn == 1 && trade_wait_seq[2] > 0 && trade_last_win_rate >= 0.60 ) wait = 0;
			#011110
			#011111
			#100000
			#if( wait_beg_up == 1 && wait_cont_dn == 5 && trade_last_win_rate >= 0.50 ) wait = 0;
			#100001
			if( wait_beg_up == 1 && wait_cont_dn == 4 && wait_cont_up == 1 && trade_last_win_rate >= 0.62 ) wait = 0;
			#100010
			#100011
			if( wait_beg_up == 1 && wait_cont_dn == 3 && wait_cont_up == 2 && trade_last_win_rate >= 0.51 ) wait = 0;
			#100100
			#100101
			#if( wait_beg_up == 1 && wait_cont_dn == 1 && wait_cont_up == 1 && trade_wait_seq[2] < 0 && trade_last_win_rate >= 0.60 ) wait = 0;
			#100110
			#100111
			#if( wait_beg_up == 1 && wait_cont_dn == 2 && wait_cont_up == 3 && trade_last_win_rate >= 0.51 ) wait = 0;
			#101000
			if( wait_beg_up == 1 && wait_cont_dn == 3 && wait_cont_up == 0 && trade_last_win_rate >= 0.15 ) wait = 0;
			#101001
			if( wait_beg_up == 1 && wait_cont_dn == 2 && wait_cont_up == 1 && trade_last_win_rate >= 0.60 ) wait = 0;
			#101010
			#101011
			#if( wait_beg_up == 1 && wait_cont_dn == 1 && wait_cont_up == 2 && trade_last_win_rate >= 0.60 ) wait = 0;
			#101100
			if( wait_beg_up == 1 && wait_cont_dn == 2 && wait_cont_up == 0 && trade_wait_seq[2] > 0 && trade_last_win_rate >= 0.25 ) wait = 0;
			#101101
			if( wait_beg_up == 1 && wait_cont_dn == 1 && wait_cont_up == 1 && trade_wait_seq[2] > 0 && trade_last_win_rate >= 0.52 ) wait = 0;
			#101110
			#101111
			#110000
			#if( wait_beg_up == 2 && wait_cont_dn == 4 && wait_cont_up == 0 && trade_last_win_rate >= 0.51 ) wait = 0;
			#110001
			#if( wait_beg_up == 2 && wait_cont_dn == 3 && wait_cont_up == 1 && trade_last_win_rate >= 0.65 ) wait = 0;
			#110010
			#110011
			#if( wait_beg_up == 2 && wait_cont_dn == 2 && wait_cont_up == 2 && trade_last_win_rate >= 0.65 ) wait = 0;
			#110100
			#if( wait_beg_up == 2 && wait_cont_dn == 2 && wait_cont_up == 0 && trade_last_win_rate >= 0.51 ) wait = 0;
			#110101
			#if( wait_beg_up == 2 && wait_cont_dn == 1 && wait_cont_up == 1 && trade_last_win_rate >= 0.65 ) wait = 0;
			#110110
			#if( wait_beg_up == 2 && wait_cont_dn == 1 && trade_wait_seq[3] > 0 && trade_last_win_rate >= 0.51 ) wait = 0;
			#110111
			#if( wait_beg_up == 2 && wait_cont_dn == 1 && wait_cont_up == 3 && trade_last_win_rate >= 0.65 ) wait = 0;
			#111000
			if( wait_beg_up == 3 && wait_cont_dn == 3 && wait_cont_up == 0 && trade_last_win_rate >= 0.18 ) wait = 0;
			#111001
			if( wait_beg_up == 3 && wait_cont_dn == 2 && wait_cont_up == 1 && trade_last_win_rate >= 0.65 ) wait = 0;
			#111010
			#if( wait_beg_up == 3 && wait_cont_dn == 1 && trade_last_win_rate >= 0.51 ) wait = 0;
			#111011
			#if( wait_beg_up == 3 && wait_cont_dn == 1 && wait_cont_up == 2 && trade_last_win_rate >= 0.65 ) wait = 0;
			#111100
			#if( wait_beg_up == 4 && wait_cont_dn == 2 && trade_last_win_rate >= 0.51 ) wait = 0;
			#111101
			#if( wait_beg_up == 4 && wait_cont_dn == 1 && wait_cont_up == 1 && trade_last_win_rate >= 0.52 ) wait = 0;
			#111110
			#if( wait_beg_up == 5 && wait_cont_dn == 1 && trade_last_win_rate >= 0.51 ) wait = 0;
			#111111
			break;
			case 7:
			#0000001
			#X000001
			#XX00001
			#XXX0001
			#XXXX001
			if( wait_cont_dn == 6 && wait_cont_up == 1 && trade_last_win_rate >= 0.60 ) wait = 0;
			if( wait_cont_dn == 5 && wait_cont_up == 1 && trade_last_win_rate >= 0.60 ) wait = 0;
			if( wait_cont_dn == 4 && wait_cont_up == 1 && trade_last_win_rate >= 0.75 ) wait = 0;
			if( wait_cont_dn == 3 && wait_cont_up == 1 && trade_last_win_rate >= 0.70 ) wait = 0;
			if( wait_cont_dn == 2 && wait_cont_up == 1 && trade_last_win_rate >= 0.85 ) wait = 0;
			break;
			case 8:
			#00000001
			#X0000001
			#XX000001
			#XXX00001
			#XXXX0001
			#XXXXX001
			if( wait_cont_dn == 7 && wait_cont_up == 1 && trade_last_win_rate >= 0.60 ) wait = 0;
			if( wait_cont_dn == 6 && wait_cont_up == 1 && trade_last_win_rate >= 0.60 ) wait = 0;
			if( wait_cont_dn == 5 && wait_cont_up == 1 && trade_last_win_rate >= 0.60 ) wait = 0;
			if( wait_cont_dn == 4 && wait_cont_up == 1 && trade_last_win_rate >= 0.60 ) wait = 0;
			if( wait_cont_dn == 3 && wait_cont_up == 1 && trade_last_win_rate >= 0.60 ) wait = 0;
			if( wait_cont_dn == 2 && wait_cont_up == 1 && trade_last_win_rate >= 0.80 ) wait = 0;
			break;
			case 9:
			#000000001
			#X00000001
			#XX0000001
			#XXX000001
			#XXXX00001
			#XXXXX0001
			#XXXXXX001
			if( wait_cont_dn == 8 && wait_cont_up == 1 && trade_last_win_rate >= 0.65 ) wait = 0;
			if( wait_cont_dn == 7 && wait_cont_up == 1 && trade_last_win_rate >= 0.65 ) wait = 0;
			if( wait_cont_dn == 6 && wait_cont_up == 1 && trade_last_win_rate >= 0.53 ) wait = 0;
			if( wait_cont_dn == 5 && wait_cont_up == 1 && trade_last_win_rate >= 0.65 ) wait = 0;
			if( wait_cont_dn == 4 && wait_cont_up == 1 && trade_last_win_rate >= 0.65 ) wait = 0;
			if( wait_cont_dn == 3 && wait_cont_up == 1 && trade_last_win_rate >= 0.65 ) wait = 0;
			if( wait_cont_dn == 2 && wait_cont_up == 1 && trade_last_win_rate >= 0.65 ) wait = 0;

			if( wait_beg_up == 2 && wait_cont_dn == 3 && wait_cont_up == 0 && trade_wait_seq[3] < 0 && trade_wait_seq[4] < 0 ) wait = 0;
			break;
			case 10:
			#0000000001
			#X000000001
			#XX00000001
			#XXX0000001
			#XXXX000001
			#XXXXX00001
			#XXXXXX0001
			#XXXXXXX001
			if( wait_cont_dn == 9 && wait_cont_up == 1 && trade_last_win_rate >= 0.75 ) wait = 0;
			if( wait_cont_dn == 8 && wait_cont_up == 1 && trade_last_win_rate >= 0.75 ) wait = 0;
			if( wait_cont_dn == 7 && wait_cont_up == 1 && trade_last_win_rate >= 0.40 ) wait = 0;
			if( wait_cont_dn == 6 && wait_cont_up == 1 && trade_last_win_rate >= 0.75 ) wait = 0;
			if( wait_cont_dn == 5 && wait_cont_up == 1 && trade_last_win_rate >= 0.75 ) wait = 0;
			if( wait_cont_dn == 4 && wait_cont_up == 1 && trade_last_win_rate >= 0.60 ) wait = 0;
			if( wait_cont_dn == 3 && wait_cont_up == 1 && trade_last_win_rate >= 0.75 ) wait = 0;
			if( wait_cont_dn == 3 && wait_cont_up == 1 && trade_last_win_rate >= 0.75 ) wait = 0;

			break;
		}
		#printf("ABC");
		#for( i = 0; i < trade_wait_size; i++ ) printf("%d", trade_wait_seq[i]>0 );
		#for( i = 0; i < trade_wait_size; i++ ) printf("\t%d:%g", i, trade_wait_seq[i] );
		#printf("\twait:%d\n", wait);

	}
	
	#############################################
	#兜底检查不合理的问题,前面各种考虑可能会冲突#
	#############################################
	if( wait == 0 ) trade_wait_size = 0;
	#############################################

	#开始处理新一天的数据
	{
		UP_num = 0; DN_num = 0;
		current_date = $1;
		cur_d_gain = $3;
		trade_num = 1;
		trade_sum++;
		trade_sel_gain[$2] = $3;
		trade_sel_pred[$2] = $4;

		if( $3 > 0 ) { UP_num++; UP_sum++; }
		else { DN_num++; DN_sum++; }
	}
}
END {
	if( trade_num ) {
		trade_seq_size++;
		cur_d_gain = ( cur_d_gain/trade_num - trade_fee_rate );
		trade_cur_win_rate = UP_num /(UP_num + DN_num);
		trade_avg_win_rate = UP_sum /(UP_sum + DN_sum);

		if( wait ) { trade_wait_sum++; }
		else {
			sum_gain *= ( 1 + cur_d_gain );
			trade_busy_sum++;
		}
		if( trade_disp_mode == 0 ){
			printf( "%s\t%d\t%g\t%g\t%g\t%g\tDAYWIN[%g%%]\t[[%s%s:%d]]\n",
					current_date, trade_num, sum_gain, cur_d_gain,
					trade_cur_win_rate, trade_avg_win_rate, trade_win_day_rate * 100,
					( wait > 0 ) ? "WAIT":"BUSY", ( cur_d_gain > 0 ) ? "+":"-", ( wait > 0 ) ? trade_wait_size : try_trade );
		}
	}
	if( trade_disp_mode == 0 ){
		printf("END 交易 %4d 等待 %4d 收益率 %g 天级胜率 %g%% 个股胜率 %g%%\n",
				trade_busy_sum, trade_wait_sum, sum_gain, trade_win_day_rate * 100, trade_avg_win_rate * 100 );
	}
}' 
