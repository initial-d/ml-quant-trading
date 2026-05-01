#include <ctype.h>
#include <string.h>
#include <math.h>

#ifndef __STOCK_UTILS_H
#define __STCCK_UTILE_H

//600360.SH->600360
inline void __stk_format( char *__s )
{
	__s[6] = 0;
}

//YYYY-MM-DD YYYY/MM/DD ->YYYYMMDD
inline void __time_format( char *__s, size_t __len )
{
	if( __len > 8 ) {
		if( strchr( __s, '-' ) || strchr( __s, '/' ) ) {
			__s[4] = __s[5]; __s[5] = __s[6]; __s[6] = __s[8]; __s[7] = __s[9]; __s[8] = 0;
		}
	}
}

inline int __time_2_days( int __date )
{
static int mon_list[12] = { 0, 31, 59, 90, 120, 151, 181, 212, 243, 273, 304, 334 };
	int Y = ( __date / 10000 );
	int M = ( __date % 10000 ) / 100 - 1;
	int D = ( __date % 10000 ) % 100;

	if( M < 0 ) M = 0; if( M > 11 ) M = 11;

	return Y * 365 + mon_list[M] + D;
}

int inline __time_dec( int __t1, int __t2 )
{
	return __time_2_days(__t1) - __time_2_days( __t2 );
}

inline double __calc_mean( float *__ary, size_t __size )
{
	double __sum = 0;
	for( size_t i = 0; i < __size; i++ ) {
		__sum += __ary[i];
	}
	__sum /= __size;
	return __sum;

}

inline void __calc_mean_vec( float *__out, float *__ary, size_t __size ) 
{
	double __sum = 0;
	for( size_t i = 0; i < __size; i++ ) {
		__sum += __ary[i];
		__out[i] = __sum / ( i + 1 );
	}
}

inline double __calc_wght_mean( float *__ary, float *__wary, size_t __size )
{
	double __sum = 0;
	double __wgt = 0;
	for( size_t i = 0; i < __size; i++ ) {
		__sum += __ary[i] * __wary[i];
		__wgt += __wary[i];
	}
	__sum /= __wgt;
	return __sum;

}

inline void __calc_wght_mean_vec( float *__out, float *__ary, float *__wary, size_t __size )
{
	double __sum = 0;
	double __wgt = 0;
	for( size_t i = 0; i < __size; i++ ) {
		__sum += __ary[i] * __wary[i];
		__wgt += __wary[i];
		__out[i] = __sum / __wgt;
	}
}

inline void __calc_dif_vec( float *__out, float *__ary1, float *__ary2, size_t __size )
{
	for( size_t i = 0; i < __size; i++ ) {
		__out[i] = __ary1[i] / __ary2[i];
	}
}

inline void __calc_dif_log_vec( float *__out, float *__ary1, float *__ary2, size_t __size )
{
	for( size_t i = 0; i < __size; i++ ) {
		__out[i] = log( __ary1[i] / __ary2[i] );
	}
}

inline void __calc_gt_vec( float *__out, float *__ary1, float *__ary2, size_t __size )
{
	float __sum = 0;
	for( size_t i = 0; i < __size; i++ ) {
		if( __ary1[i] > __ary2[i] ) __sum += 1;
		__out[i] = __sum / ( i + 1 );
	}
}

inline void __calc_lt_vec( float *__out, float *__ary1, float *__ary2, size_t __size )
{
	float __sum = 0;
	for( size_t i = 0; i < __size; i++ ) {
		if( __ary1[i] < __ary2[i] ) __sum += 1;
		__out[i] = __sum / ( i + 1 );
	}
}

inline void __calc_mtm_vec( float *__out, float *__ary, size_t __size )
{
	__size -= 1;
	for( size_t i = 0; i < __size; i++ ) {
		__out[i] = __ary[0] / __ary[i+1];
	}
}

inline void __calc_mtm_log_vec( float *__out, float *__ary, size_t __size )
{
	__size -= 1;
	for( size_t i = 0; i < __size; i++ ) {
		__out[i] = log( __ary[0] / __ary[i+1] );
	}
}

inline void __calc_avg_mtm_vec( float *__out, float *__ary, size_t __size )
{
	__size -= 1;
	for( size_t i = 0; i < __size; i++ ) {
		__out[i] = __ary[0] / __ary[i+1] / ( i + 2 );
	}
}

inline void __calc_avg_mtm_log_vec( float *__out, float *__ary, size_t __size )
{
	__size -= 1;
	for( size_t i = 0; i < __size; i++ ) {
		__out[i] = log( __ary[0] / __ary[i+1] / ( i + 2 ) );
	}
}


inline double __calc_var( float *__ary, size_t __size )
{
	double __sum = 0;
	double __mean = __calc_mean( __ary, __size );
	for( size_t i = 0; i < __size; i++ ) {
		double v = ( __ary[i] - __mean ) ;
		__sum += v * v;
	}
	__sum /= ( __size - 1 );
	return __sum;
}

inline double __calc_std_dev( float *__ary, size_t __size )
{
	double __sum = __calc_var( __ary, __size );
	return sqrt( __sum );
}

inline double __calc_var_rate( float *__ary, size_t __size )
{
	double __sum = 0;
	double __mean = __calc_mean( __ary, __size );
	if( __mean == 0 ) return 1;
	for( size_t i = 0; i < __size; i++ ) {
		double v = ( __ary[i] - __mean ) ;
		__sum += v * v;
	}
	__sum /= ( __size - 1 );
	return sqrt( __sum ) / fabs( __mean );
}

inline double __calc_var_rate( float *__ary, float __mean, size_t __size )
{
	double __sum = 0;
	if( __mean == 0 ) return 1;
	for( size_t i = 0; i < __size; i++ ) {
		double v = ( __ary[i] - __mean ) ;
		__sum += v * v;
	}
	__sum /= ( __size - 1 );
	return sqrt( __sum ) / fabs( __mean );
}

inline void __calc_var_rate_vec( float *__out, float *__ary, size_t __size )
{
	double __sum = 0;
	float __mean[__size];
	__calc_mean_vec( __mean, __ary, __size );
	for( size_t i = 0; i < __size - 1; i++ ) {
		__out[i] = __calc_var_rate( __ary, __mean[i+1], i + 2 );
	}
}

inline double __calc_cov( float *__aryX, float *__aryY, size_t __size )
{
	double __sum = 0;
	double __meanX = __calc_mean( __aryX, __size );
	double __meanY = __calc_mean( __aryY, __size );
	for( size_t i = 0; i < __size; i++ ) {
		__sum += ( __aryX[i] - __meanX ) * ( __aryY[i] - __meanY );
	}
	__sum /= ( __size - 1 );
	return  __sum;
}

inline double __calc_cov_rate( float *__aryX, float *__aryY, size_t __size )
{
	double __sum = 0, __sumX = 0, __sumY = 0;
	double __meanX = __calc_mean( __aryX, __size );
	double __meanY = __calc_mean( __aryY, __size );

	for( size_t i = 0; i < __size; i++ ) {
		double __x = __aryX[i] - __meanX;
		double __y = __aryY[i] - __meanY;
		__sum  += __x * __y;
		__sumX += __x * __x;
		__sumY += __y * __y;
	}
	__sumX = __sumX * __sumY;
	if( __sumX == 0 ) { __sum = 0; }
	else { __sum = __sum / sqrt( __sumX ); }
	return  __sum;
}

inline double __calc_cov_rate( float *__aryX, float __meanX, float *__aryY, float __meanY,size_t __size )
{
	double __sum = 0, __sumX = 0, __sumY = 0;

	for( size_t i = 0; i < __size; i++ ) {
		double __x = __aryX[i] - __meanX;
		double __y = __aryY[i] - __meanY;
		__sum  += __x * __y;
		__sumX += __x * __x;
		__sumY += __y * __y;
	}
	__sumX = __sumX * __sumY;
	if( __sumX == 0 ) { __sum = 0; }
	else { __sum = __sum / sqrt( __sumX ); }
	return  __sum;
}

inline void __calc_cov_rate_vec( float *__out, float *__aryX, float *__aryY, size_t __size )
{
	double __sum = 0, __sumX = 0, __sumY = 0;
	float __meanX[__size];
	float __meanY[__size];

	__calc_mean_vec( __meanX, __aryX, __size );
	__calc_mean_vec( __meanY, __aryY, __size );

	for( size_t i = 0; i < __size - 1; i++ ) {
		__out[i] = __calc_cov_rate( __aryX, __meanX[i+1], __aryY, __meanY[i+1], i+2);
	}
}

inline double __calc_cosine( float *__aryX, float *__aryY, size_t __size )
{
	double __sumD = 0, __sumX = 0, __sumY = 0;

	for( size_t i = 0; i < __size; i++ ) {
		__sumD += __aryX[i] * __aryY[i];
		__sumX += __aryX[i] * __aryX[i];
		__sumY += __aryY[i] * __aryY[i];
	}
	if( __sumX * __sumY == 0 ) return 0;
	return __sumD / sqrt( __sumX * __sumY );
}

#endif
