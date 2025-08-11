#/bin/sh
#
disp_month=$1
[ ! "$disp_month" ] && disp_month=0

disp_year=$2
[ ! "$disp_year" ] && disp_year=1

disp_all=$3
[ ! "$disp_all" ] && disp_all=1

awk '{ print $1,$3,$4,$NF; }' |
awk -v disp_month=$disp_month -v disp_year=$disp_year -v disp_all=$disp_all '
function calcSP( date1, date2, gain, dgun, state, size,        i,sumWorked,dif_gain,dif_sum,dif_mean,rId,Rgain,Ygain,Dgain,Ogain,vsum,v,sap)
{
	if( size > 2 ) {
		sumWin = 0;
		sumWorked = 0;
		Rgain = 1;
		for( i = 0; i < size; i++ ) {
			if( state[i]~/BUY|SELL|BUSY/ ) {
				sumWorked++;
				if( dgun[i] > 0 ) sumWin++;
			}
		}
		if( sumWorked == 0 ) sumWorked = 1;

		dif_gain[0] = 0;
		for( i = 0; i < size - 1; i++ ) {
			dif_gain[i] = gain[i+1]/gain[i] - 1;
			dif_sum += dif_gain[i];
		}
		dif_mean = dif_sum / ( size - 1 );

		for( rId = 1; rId < size; rId++ ) {
			if( dgun[rId] != 0 ) break;
		}
		if( dgun[rId] == 0 ) {
			Rgain = gain[size - 1] / gain[0];
		}
		else {
			Rgain = gain[size - 1] / gain[0] * ( 1 + dgun[0] * ( gain[rId] / gain[rId-1] - 1.0 ) / dgun[rId] );
		}
		Ygain = Rgain ^ ( 245.0 / size ) - 1;
		Dgain = Rgain ^ ( 1 / size ) - 1;
		Ogain = Rgain ^ ( 1 / sumWorked ) - 1;

		sap = 0;
		vsum = 0;
		for( i = 0; i < size - 1; i++ ) {
			v = dif_gain[i] - dif_mean;
			vsum += v * v;
		}
		vsum /= ( size - 2 );
		vsum = sqrt( vsum );
		if( vsum ) sap = Ygain / vsum / sqrt( 245 );

		printf( "%4s %4s SARP: %.2f\tBUSY: %4d DSum: %4d WRate: %.2f%% RGain: %.3f\tYGain: %.3f\tDGain: %.4f\topGain: %.4f\n",
				date1, date2, sap, sumWorked, size, sumWin / sumWorked * 100, Rgain - 1, Ygain, Dgain, Ogain );
	}
}
BEGIN{
	sizeA = 0;
}
{
	if( /^[0-9]/ ) {
		dateA[sizeA] = $1;
		gainA[sizeA] = $2;
		dgunA[sizeA] = $3;
		statA[sizeA] = $4;
		sizeA++;
	}
}
END{
	if( sizeA > 2 ) {
		sizeM = 0;
		sizeY = 0;
		Year = substr( dateA[0], 1, 4 );
		Month= substr( dateA[0], 5, 2 );
		if( disp_month ) printf("%d MON:\n", Year);

		#<=sizeA,利用最后一个数据为空，少写几行代码
		for( i = 0; i <= sizeA; i++ ) {
			current_Month= substr( dateA[i], 5, 2 );
			if( current_Month == Month ) {
				gainM[sizeM] = gainA[i];
				dgunM[sizeM] = dgunA[i];
				statM[sizeM] = statA[i];
				sizeM++;
			}
			else {
				if( disp_month ) {
					calcSP( "    ", Month, gainM, dgunM, statM, sizeM );
				}
				Month = current_Month;
				sizeM = 0;
				gainM[sizeM] = gainA[i];
				dgunM[sizeM] = dgunA[i];
				statM[sizeM] = statA[i];
				sizeM++;
			}
			current_Year = substr( dateA[i], 1, 4 );
			if( current_Year == Year ) {
				gainY[sizeY] = gainA[i];
				dgunY[sizeY] = dgunA[i];
				statY[sizeY] = statA[i];
				sizeY++;
			}
			else {
				if( disp_year ) {
					calcSP( Year, "SUM:", gainY, dgunY, statY, sizeY );
				}
				Year = current_Year;
				sizeY = 0;
				if( i < sizeA ) {
					gainY[sizeY] = gainA[i];
					dgunY[sizeY] = dgunA[i];
					statY[sizeY] = statA[i];
					sizeY++;
					if( disp_month ) printf("%d MON:\n", Year);
				}
			}
		}
		if( disp_all ) {
			calcSP( "ALLY", "SUM:", gainA, dgunA, statA, sizeA );
		}
	}
}'
