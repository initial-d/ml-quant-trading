



cat /home/s/apps/xproject-data/stock_info/stock_records/20[1-2]*.csv.all.limit |
awk -v d="$d" -F ',' '{
	printf( "%s\t%s\t%g\t%g\t%g\t%g\t%g\t%g\t%s\t%g\t%g\t%g\n",
		$3,$2,$4,$7,$5,$6,( $12*10/$11 + 0.001 ),$11,( ($48+$49+$50+$51) > 0.001 ) ? "1":"0", $59,$60,$8);
}'
#sort 	
