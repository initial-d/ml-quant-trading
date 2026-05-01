#/bin/sh
#
awk -F'\t' '{ print $1,$3; }' |
awk 'BEGIN{
	sizeA = 0;
	dataA[0] = 0;
	sizeY = 0;
	dataY[0] = 0;
	current_year = "";
}
{
	if( /^[0-9]/ ) {
		dataA[sizeA++] = $2;
		year = substr( $1, 1, 4 );
		if( current_year != year ) {
			if( current_year != "" ) {

				dif_data[0] = 0;
				for( i = 0; i < sizeY - 1; i++ ) {
					dif_data[i] = log( dataY[i+1]/dataY[i] );
					dif_sum += dif_data[i];
				}
				dif_mean = dif_sum / ( sizeY - 1 );
				gain = dataY[sizeY-1] / dataY[0];
				gain = gain ^ ( 252.0 / sizeY ) - 1;

				vsum = 0;
				for( i = 0; i < sizeY - 1; i++ ) {
					v = dif_data[i] - dif_mean;
					vsum += v * v;
				}
				vsum /= ( sizeY - 2 );
				vsum = sqrt( vsum );

				sap = gain / vsum / sqrt( 252 );
				printf( "%s SP:%.3f\tYGain:%g\n", current_year, sap, gain);
				
			}
			sizeY = 0;
			current_year = year;
			dataY[sizeY++] = $2;
		}
		else {
			dataY[sizeY++] = $2;
		}
	}
}
END{
	if( sizeY ) {
				dif_data[0] = 0;
				for( i = 0; i < sizeY - 1; i++ ) {
					dif_data[i] = log( dataY[i+1]/dataY[i] );
					dif_sum += dif_data[i];
				}
				dif_mean = dif_sum / ( sizeY - 1 );
				gain = dataY[sizeY-1] / dataY[0];
				gain = gain ^ ( 252.0 / sizeY ) - 1;

				vsum = 0;
				for( i = 0; i < sizeY - 1; i++ ) {
					v = dif_data[i] - dif_mean;
					vsum += v * v;
				}
				vsum /= ( sizeY - 2 );
				vsum = sqrt( vsum );

				sap = gain / vsum / sqrt( 252 );
				printf( "%s SP:%.3f\tYGain:%g\n", current_year, sap, gain);
	}
	dif_data[0] = 0;
	for( i = 0; i < sizeA - 1; i++ ) {
		dif_data[i] = log( dataA[i+1]/dataA[i] );
		dif_sum += dif_data[i];
	}
	dif_mean = dif_sum / ( sizeA - 1 );
	gain = dataA[sizeA-1] / dataA[0];
	gain = gain ^ ( 252.0 / sizeA ) - 1;

	vsum = 0;
	for( i = 0; i < sizeA - 1; i++ ) {
		v = dif_data[i] - dif_mean;
		vsum += v * v;
	}
	vsum /= ( sizeA - 2 );
	vsum = sqrt( vsum );

	sap = gain / vsum / sqrt( 252 );
	printf( "sum  SP:%.3f\tYGain:%g\n", sap, gain);
		
}'
