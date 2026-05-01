#!/bin/sh
#
#awk '{ if( $3 >= 0.486 && $3 <= 1.999) print $0}'|sort -k1 |
#awk '{ if( $3 >= 0.480 && $3 <= 1.999) print $0}'|sort -k1,1 -k3,3gr|
awk '{ if( $3 >= 0 ) print $0}'|
awk 'BEGIN{
	cash = 0; giftpoint = 1; sum_gain = 1; cur_d_gain = 0; trade_num = 0; current_date=""; 
	wait = 1; busy_sum = 0; wait_sum = 0; 
	feerate=0.00125; trade_sum = 0; trade_days = 0;
	UP_num=0; DN_num=0; win_rate = 0; UP_sum = 0; DN_sum = 0; avg_win_rate = 0; 
	dn_conitue_num = 0;
	trade_max=1;
	arraylen = 0;
	gtlen = 0;

}
{
	if( current_date != $1 ) {
		if( current_date != "" ) {
			trade_days++;
			cur_d_gain = ( cur_d_gain/trade_num - feerate );
			win_rate = UP_num /(UP_num + DN_num );
			avg_win_rate  = UP_sum /(UP_sum + DN_sum );

			wait = 0
			#if (gtlen < 160 ) wait =1;
			#if (arraylen < 100 ) wait =1;
			if( wait ) { wait_sum++; }
			else {
				sum_gain *= ( 1 + cur_d_gain );
				busy_sum++;
				for ( i in array )print array[i]" ##";
			}
			printf( "%s\t%d\t%.4g\t%.4g\tWINRATE[%g%%]\tAVGWIN[%g%%]\t[[%s]]\t%g\n",
					current_date, trade_num, sum_gain, cur_d_gain, win_rate * 100, avg_win_rate * 100, ( wait > 0 ) ? "WAIT":"BUSY", gtlen);
			if( wait > 0 ) {
				if( cur_d_gain >= 0.0 ) { 
					wait = 0;
				}
				else wait++;
			}
			else {
				if( cur_d_gain < 0.002 ) wait = 1;
			}
			if( wait < 0 ) wait = 0;
			if( cur_d_gain > 0 ) dn_continue = 0; else dn_continue++;

			UP_num = 0; DN_num = 0;
		}
		cur_d_gain = $4;
		trade_num  = 1;
		current_date = $1;
		trade_sum++;
		if( $4 > 0 ) { UP_num += 1; UP_sum += 1; }
		else { DN_num += 1; DN_sum += 1; }
		delete array;
		arraylen = 0;
		array[arraylen++] = $0;
		gtlen = 0;
		gtlen++;
	}
	else {
		if( trade_num < trade_max ) {
			cur_d_gain += $4;
			trade_num++;
			trade_sum++;
			if( $4 > 0 ) { UP_num += 1; UP_sum += 1; }
			else { DN_num += 1; DN_sum += 1; }
			array[arraylen++] = $0;
		}
		gtlen++;
		
	}
}
END {
	trade_days++;
	cur_d_gain = ( cur_d_gain/trade_num - feerate );
	win_rate = UP_num /(UP_num + DN_num + 1);
	avg_win_rate  = UP_sum /(UP_sum + DN_sum + 1);

	if( wait ) { wait_sum++; }
	else {
		sum_gain *= ( 1 + cur_d_gain );
		busy_sum++;
	}
	for(i in array)print array[i]" ##";
	printf("busy %d days, wait %d days, gain rate:%g winrate:%g\n", busy_sum, wait_sum, cash + sum_gain, avg_win_rate );
}' 
