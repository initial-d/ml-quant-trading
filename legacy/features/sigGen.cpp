#include <stdio.h>
#include <stdlib.h>
#include <ctype.h>
#include <string.h>
#include <math.h>
#include <string>
#include <map>
#include <algorithm>
#include "dyncAry.h"
#include "stkutils.h"

using namespace std;

struct tradeCol {
	int     org_stkno;
	int		org_dtime;
	int     org_gday;
	float	org_open;
	float	org_close;
	float	org_high;
	float	org_low;
	float	org_avg;
	float	org_vol;
	float	org_up;
	float	org_dp;
};


struct tradeSeq {
	dyncAry<int>	org_dtime;
	dyncAry<int>	org_gday;
	dyncAry<float>	org_open;;
	dyncAry<float>	org_close;
	dyncAry<float>	org_high;
	dyncAry<float>	org_low;
	dyncAry<float>	org_avg;
	dyncAry<float>	org_vol;
	dyncAry<float>	org_up;
};

typedef map<int, tradeSeq *> tradeMix;

static const int _G_data_win_size = 21;
struct seqWind {
	int     label;
	int     org_dtime;
	float	dst_gain;
	float	max_gain;
	float	min_gain;
	float	cur_gain;
	float	org_open[_G_data_win_size];
	float	org_close[_G_data_win_size];
	float	org_high[_G_data_win_size];
	float	org_low[_G_data_win_size];;
	float	org_avg[_G_data_win_size];
	float	org_vol[_G_data_win_size];
	float	adj_open[_G_data_win_size];
	float	adj_close[_G_data_win_size];

	//窗口自己处理产生的
	float	kmean_org_open[_G_data_win_size];
	float	kmean_org_close[_G_data_win_size];
	float	kmean_org_high[_G_data_win_size];
	float	kmean_org_low[_G_data_win_size];
	float	kmean_org_avg[_G_data_win_size];
	float	kmean_org_vol[_G_data_win_size];
	float	kmean_adj_open[_G_data_win_size];
	float	kmean_adj_close[_G_data_win_size];
	float	wmean_org_avg[_G_data_win_size];

	float   org_close_open[_G_data_win_size];
	float   org_close_high[_G_data_win_size];
	float   org_close_low[_G_data_win_size];
	float   org_close_avg[_G_data_win_size];
	float   org_high_open[_G_data_win_size];
	float   org_high_low[_G_data_win_size];
	float   org_high_avg[_G_data_win_size];
	float   org_open_low[_G_data_win_size];
	float   org_open_avg[_G_data_win_size];
	float   org_avg_low[_G_data_win_size];
	float   adj_close_open[_G_data_win_size];

	float   kmean_org_close_open[_G_data_win_size];
	float   kmean_org_close_high[_G_data_win_size];
	float   kmean_org_close_low[_G_data_win_size];
	float   kmean_org_close_avg[_G_data_win_size];
	float   kmean_org_high_open[_G_data_win_size];
	float   kmean_org_high_low[_G_data_win_size];
	float   kmean_org_high_avg[_G_data_win_size];
	float   kmean_org_open_low[_G_data_win_size];
	float   kmean_org_open_avg[_G_data_win_size];
	float   kmean_org_avg_low[_G_data_win_size];
	float   kmean_adj_close_open[_G_data_win_size];

	float   kmean_org_open_wavg[_G_data_win_size];
	float   kmean_org_close_wavg[_G_data_win_size];
	float   kmean_org_high_wavg[_G_data_win_size];
	float   kmean_org_low_wavg[_G_data_win_size];
	float   kmean_org_avg_wavg[_G_data_win_size];


	float   mtm_vec[400][_G_data_win_size-1];
};

static int   __io_size = 0;
static char *__io_buff = NULL;
bool read_line(FILE *f)
{
	int __len;

	if( !__io_size ) {
		__io_size += 1024 * 32;
		__io_buff = (char *)malloc( __io_size );
		if( __io_buff == NULL ) return false;
	}
	if( fgets( __io_buff, __io_size, f ) == NULL) return false;
	while (strrchr(__io_buff, '\n') == NULL) {
		__io_size += 1024 * 32;
		__io_buff = (char *)realloc(__io_buff, __io_size);
		__len = (int)strlen(__io_buff);
		if (fgets(__io_buff + __len, __io_size - __len , f) == NULL) break;
	}
	__len = strlen(__io_buff);
	while ( __len && isspace(__io_buff[__len - 1])) __len--;
	__io_buff[__len] = 0;
	return true;
}


bool parse_input( tradeCol &__tr )
{
	char *p = strtok( __io_buff, " \t");
	if( p == NULL ) return false;

	__time_format( p, strlen( p ) );
	__tr.org_dtime = atoi( p );

	__tr.org_gday = 0;
	for( int i = 0; i < 10; i++ ) {
		p = strtok(NULL, " \t");
		if ( p == NULL) return false;

		while (isspace(*p)) p++;

		switch(i) {
		case 0:
			__stk_format( p );
			__tr.org_stkno = atoi( p );
			break;
		case 1:
			__tr.org_open = strtod(p, NULL);
			break;
		case 2:
			__tr.org_close = strtod(p, NULL);
			break;
		case 3:
			__tr.org_high = strtod(p, NULL);
			break;
		case 4:
			__tr.org_low = strtod(p, NULL);
			break;
		case 5:
			__tr.org_avg = strtod(p, NULL);
			break;
		case 6:
			__tr.org_vol = strtod(p, NULL);
			break;
		case 7:
			__tr.org_gday = atoi(p);
		case 8:
			__tr.org_up = strtod(p,NULL);
			break;
		case 9:
			__tr.org_dp = strtod(p,NULL);
			break;
		}
	}

	if( __tr.org_open <= 0.001 ) return false;
	if( __tr.org_close <= 0.001 ) return false;
	if( __tr.org_high <= 0.001 ) return false;
	if( __tr.org_low <= 0.001 ) return false;
	if( __tr.org_avg <= 0.001 ) return false;
	if( __tr.org_vol <= 0.001 ) return false;

	return true;
}

void print_usage(const char *name)
{
    fprintf(stderr, "usage: %s [-s begin] [-e end] [inputfile=stdin]\n", name);
    fprintf(stderr, "   -s: start date YYYY-MM-DD\n");
    fprintf(stderr, "   -e: end date YYYY-MM-DD\n");
    fprintf(stderr, "   -d: predict n day later[1]\n");
    fprintf(stderr, "   -v: postive smaple win val[1.01]\n");
    fprintf(stderr, "   -b: [2|1|0=default] output format\n");
    fprintf(stderr, "   -w: sample add weight, gain/0.01\n");
    fprintf(stderr, "   -f: fea.txt: feature select lst\n");
    fprintf(stderr, "   -c: [0-1=default:data range check 2-gain check 3-all]\n");
    exit(0);
}

static int    _G_bin_out = 0;
static int    _G_begin_date = 0;
static int    _G_end_date = 30000101;
static int    _G_dst_span = 1;
static float  _G_dst_winval = 1.01;
static float  _G_cur_filter = -0.01;
static int    _G_gen_wght = 0;
static int    _G_range_check = 1;
static int    _G_gain_check = 1;
static FILE  *_G_input = stdin;
static int    _G_select_sig = 0;
static int    _G_sig_sel_buf[4096] = { 0 };

void parse_args( int argc, char **argv )
{
char __stime[16];

	for (int i = 1; i < argc; i++) {

		if (argv[i][0] != '-') {
			_G_input = fopen(argv[i], "rb");
			if( _G_input == NULL ) {
				fprintf( stderr, "cannot open: %s\n", argv[i] );
				exit(1);
			}
			break;
		}

		char *op_arg = NULL;
		char op = argv[i][1];


		if (strlen(argv[i]) > 2) op_arg = argv[i] + 2;
		else if (i < argc - 1) op_arg = argv[++i];
		else print_usage(argv[0]);

		switch( op ) {
		case 'h':
		case '?':
			print_usage(argv[0]);
			break;
		case 's':
			strncpy( __stime, op_arg, sizeof( __stime ) );
			__time_format( __stime, strlen( __stime ) );
			_G_begin_date = atoi( __stime );
			break;
		case 'e':
			strncpy( __stime, op_arg, sizeof( __stime ) );
			__time_format( __stime, strlen( __stime ) );
			_G_end_date = atoi( __stime );
			break;
		case 'd':
			_G_dst_span = atoi( op_arg );
			if( _G_dst_span < 1 ) _G_dst_span = 1;
			break;
		case 'v':
			_G_dst_winval = atof( op_arg );
			if( _G_dst_winval < 1.001 ) _G_dst_winval = 1.001;
			break;
		case 'b':
			_G_bin_out = atoi( op_arg );
			break;
		case 'c':
			{
				int v = atoi( op_arg ) & 0x03;
				switch( v ) {
				case 0:
					_G_range_check = 0;
					_G_gain_check = 0;
					break;
				case 1:
					_G_range_check = 1;
					_G_gain_check = 0;
					break;
				case 2:
					_G_range_check = 0;
					_G_gain_check = 1;
					break;
				case 3:
					_G_range_check = 1;
					_G_gain_check = 1;
					break;
				}
			}
			break;
		case 'w':
			_G_gen_wght = atoi( op_arg );
			break;
		case 'l':
			_G_cur_filter  = atof( op_arg );
			break;
		case 'f':
		{
			FILE *fp = fopen( op_arg, "rb");
			if( fp == NULL ) {
				fprintf( stderr, "cannot open: %s\n", op_arg );
				exit(1);
			}

			size_t __sum = 0;
			char __stmp[128];
			while( __sum < sizeof( _G_sig_sel_buf ) / sizeof( int ) ) {
				if( fgets( __stmp, sizeof( __stmp ) , fp ) == NULL) break;
				int __sig = atoi( __stmp );
				if( __sig <= 0 || __sig >= sizeof( _G_sig_sel_buf ) / sizeof( int ) ) continue;
				__sig--;
				_G_sig_sel_buf[__sig] = 1;
				__sum++;
			}
			fclose( fp );
			_G_select_sig = __sum;
			break;
		}
		defalt:
			print_usage(argv[0]);
			break;
		}
	}
}

struct seq_rank {
	size_t __pos;
	float  __val;
};
struct seq_less
{
	bool operator()(const seq_rank& __x, const seq_rank& __y )
	{
		return  __x.__val < __y.__val;
	}
};

static const int _G_next_days = 6;
void  __calc_rank_sys( dyncAry<double> &__out, float *__ary, size_t __size )
{
	int      __curpos = 0;
	seq_rank __rank[ __size];
	int      __seqn[ __size];

	for( int i = 0; i < __size; i++ ) {
		__rank[i].__pos = i;
		__rank[i].__val = __ary[i];
	}
	stable_sort( __rank, __rank + __size, seq_less());

	//升序的原始天号21
	for( int i = 0; i < __size; i++ ) {
		__out.push_back( __rank[i].__pos );
		__seqn[__rank[i].__pos ] = i;
		if( __rank[i].__pos == (__size - 1 ) ) {
			__curpos = i;
		}
	}
	//每天的排名21
	for( int i = 0; i < __size; i++ ) {
		__out.push_back( __seqn[i] );
	}
	//+2
	__out.push_back( __size - __rank[0].__pos  );		//相对最小值相隔天数
	__out.push_back( __size - __rank[__size-1].__pos );	//相对最大值相隔天数
	//最大最小值比值+2
	__out.push_back( log( __rank[__curpos].__val / __rank[0].__val ) );
	__out.push_back( log( __rank[__curpos].__val / __rank[__size-1].__val ) );
	//中位数比值+2
	__out.push_back( log( __rank[__curpos].__val / __rank[__size/2].__val ) );
	__out.push_back( log( __rank[__curpos].__val / __rank[__size/2 + 1].__val ) );

	//排序最近6天的上涨趋势+6
	{
		int __sum = 0;
		for( int i = __curpos + 1; ( i < __size ) && ( __sum < _G_next_days ) ; i++ ) {
			double __span = log( __rank[i].__val / __rank[__curpos].__val );
			__out.push_back( __span );
			__sum++;
		}
		for( int i = __sum; i < _G_next_days; i++ ) {
			__out.push_back( 0 );
		}
	}

	//排序最近6天的下降趋势+6
	{
		int __sum = 0;
		for( int i = __curpos - 1; ( i >= 0) && ( __sum < _G_next_days ) ; i-- ) {
			double __span = log( __rank[__curpos].__val / __rank[i].__val );
			__out.push_back( __span );
			__sum++;
		}
		for( int i = __sum; i < _G_next_days; i++ ) {
			__out.push_back( 0 );
		}
	}
}

void  __calc_upnum( dyncAry<double> &__out, seqWind &__seqWin )
{
	float __cc[_G_data_win_size -1];
	float __oc[_G_data_win_size -1];

	__calc_gt_vec( __cc, __seqWin.org_close, __seqWin.org_close + 1, _G_data_win_size - 1);
	__calc_gt_vec( __oc, __seqWin.org_open,  __seqWin.org_close + 1, _G_data_win_size - 1);

	for( int i = 0; i < _G_data_win_size - 1; i++ ) __out.push_back( __cc[i] );
	for( int i = 0; i < _G_data_win_size - 1; i++ ) __out.push_back( __oc[i] );
}

void  __calc_RSV( dyncAry<double> &__out, seqWind &__seqWin )
{
	float __rsv[_G_data_win_size -1];
	float __Hn  = __seqWin.org_high[0];
	float __Ln  = __seqWin.org_low[0];

	for( int i = 1; i < _G_data_win_size; i++ ) {
		if( __seqWin.org_high[i] > __Hn ) __Hn = __seqWin.org_high[i];
		if( __seqWin.org_low[i]  < __Ln ) __Ln = __seqWin.org_low[i];
		if( ( __Hn - __Ln ) < 0.0001 ) __rsv[i-1] = 0;
		else {
			__rsv[i-1] = ( __seqWin.org_close[i] - __Ln ) / ( __Hn - __Ln );
		}
	}
	for( int i = 0; i < _G_data_win_size - 1; i++ ) {
		__out.push_back( __rsv[i] );
	}
}

size_t deal_input( tradeMix &__trMix )
{
size_t __sum = 0;
tradeCol __trCol;

	while( read_line( _G_input ) ) {
		if( !parse_input( __trCol ) ) {
			continue;
		}

		if( _G_range_check ) {
			//Now DONT carefully check the range,predeal function can do this
			if( __time_dec( __trCol.org_dtime, _G_begin_date ) < -( _G_data_win_size  + 10 ) ) {
				continue;
			}
			if( __time_dec( __trCol.org_dtime, _G_end_date   ) > ( _G_dst_span + 10 ) ) {
				break;
			}
		}

		__sum++;
		tradeMix::iterator it = __trMix.find( __trCol.org_stkno );
		if( it == __trMix.end() ) {
			tradeSeq *__trSeq = new tradeSeq;;
			__trMix[__trCol.org_stkno] = __trSeq;
			it = __trMix.find( __trCol.org_stkno );
		}
		it->second->org_dtime.push_back( __trCol.org_dtime );
		it->second->org_gday.push_back ( __trCol.org_gday );
		it->second->org_open.push_back ( __trCol.org_open );
		it->second->org_close.push_back( __trCol.org_close );
		it->second->org_high.push_back ( __trCol.org_high );
		it->second->org_low.push_back  ( __trCol.org_low );
		it->second->org_avg.push_back  ( __trCol.org_avg );
		it->second->org_vol.push_back  ( __trCol.org_vol );
		it->second->org_up.push_back   ( __trCol.org_up  );
	}
	return __sum;
}


//RETURN SKIP DAY NUM
int predeal_seqwind( tradeSeq *__trSeq, size_t __begin, seqWind &__seqWin )
{

	//FILL BASE INFO
	{
		size_t __D0 = __begin + _G_data_win_size  - 1;
		size_t __D1 = __D0 + _G_dst_span;

		//gift day
		if( __trSeq->org_gday[__D1] ) {
			return __D1 - __begin;;
		}
		if( _G_range_check ) {
			//long time no trade
			int __time_span = __time_dec( __trSeq->org_dtime[__D1], __trSeq->org_dtime[__D0] );
			if( __time_span > _G_dst_span + 10 ) {
				return __D1 - __begin;;
			}
		}

		__seqWin.dst_gain = __trSeq->org_close[__D1] / __trSeq->org_close[__D0] - 1;
		//gift data no cover
		if( __seqWin.dst_gain > 0.21 || __seqWin.dst_gain < -0.21 ) {
			return __D1 - __begin;;
		}

		if( _G_range_check ) {
			//RANGE CHECK
			if( __trSeq->org_dtime[__D0] < _G_begin_date ) return 1;
			if( __trSeq->org_dtime[__D0] > _G_end_date ) {
				return __trSeq->org_dtime.size() - __begin;
			}
		}

		__seqWin.label= ( __seqWin.dst_gain >= _G_dst_winval - 1 ) ? 1 : 0;;
		__seqWin.org_dtime= __trSeq->org_dtime[__D0];
		__seqWin.max_gain = __trSeq->org_high[__D1];
		__seqWin.min_gain = __trSeq->org_low[__D1];
		for( int i = __D0 + 1; i < __D1; i++ ) {
			if( __trSeq->org_high[i] > __seqWin.max_gain ) __seqWin.max_gain = __trSeq->org_high[i];
			if( __trSeq->org_low[i]  < __seqWin.min_gain ) __seqWin.min_gain = __trSeq->org_low[i];
		}
		__seqWin.max_gain = __seqWin.max_gain / __trSeq->org_close[__D0] - 1;
		__seqWin.min_gain = __seqWin.min_gain / __trSeq->org_close[__D0] - 1;

		//delete the current day donw's stock
		__seqWin.cur_gain = __trSeq->org_close[__D0] / __trSeq->org_close[__D0 - 1] - 1;

		if( _G_gain_check ) {
			if( __seqWin.cur_gain < _G_cur_filter ) return 1;
			//去掉涨停,两分钱
			if( __trSeq->org_close[__D0]  > ( __trSeq->org_up[__D0] - 0.02 ) ) return 1;
		}
	}

	//COPY ORG_DATA TO WINDOW
	int __end = __begin + _G_data_win_size;
	for( int i = 0; i < _G_data_win_size; i++ ) {
		int __cursor = __end - i - 1;
		__seqWin.org_open[i]  = __trSeq->org_open[__cursor];
		__seqWin.org_close[i] = __trSeq->org_close[__cursor ];
		__seqWin.org_high[i]  = __trSeq->org_high[__cursor];
		__seqWin.org_low[i]   = __trSeq->org_low[__cursor];
		__seqWin.org_avg[i]   = __trSeq->org_avg[__cursor];
		__seqWin.org_vol[i]   = __trSeq->org_vol[__cursor];

		__seqWin.adj_open[i]  = ( __seqWin.org_open[i]  + __seqWin.org_high[i] + __seqWin.org_low[i] ) / 3 ;
		__seqWin.adj_close[i] = ( __seqWin.org_close[i] + __seqWin.org_high[i] + __seqWin.org_low[i] ) / 3 ;

		__seqWin.org_close_open[i]= __seqWin.org_close[i] / __seqWin.org_open[i];
		__seqWin.org_close_high[i]= __seqWin.org_close[i] / __seqWin.org_high[i];
		__seqWin.org_close_low[i] = __seqWin.org_close[i] / __seqWin.org_low[i] ;
		__seqWin.org_close_avg[i] = __seqWin.org_close[i] / __seqWin.org_avg[i] ;
		__seqWin.org_high_open[i] = __seqWin.org_high[i]  / __seqWin.org_open[i];
		__seqWin.org_high_low[i]  = __seqWin.org_high[i]  / __seqWin.org_low[i] ;
		__seqWin.org_high_avg[i]  = __seqWin.org_high[i]  / __seqWin.org_avg[i] ;
		__seqWin.org_open_low[i]  = __seqWin.org_open[i]  / __seqWin.org_low[i] ;
		__seqWin.org_open_avg[i]  = __seqWin.org_open[i]  / __seqWin.org_avg[i] ;
		__seqWin.org_avg_low[i]   = __seqWin.org_avg [i]  / __seqWin.org_low[i] ;
		__seqWin.adj_close_open[i]= __seqWin.adj_close[i] / __seqWin.adj_open[i];
	}

	//CALC BASE DATA
	__calc_mean_vec( __seqWin.kmean_org_open, __seqWin.org_open, _G_data_win_size);
	__calc_mean_vec( __seqWin.kmean_org_close,__seqWin.org_close,_G_data_win_size);
	__calc_mean_vec( __seqWin.kmean_org_high, __seqWin.org_high, _G_data_win_size);
	__calc_mean_vec(__seqWin.kmean_org_low,   __seqWin.org_low, _G_data_win_size);
	__calc_mean_vec(__seqWin.kmean_org_avg,   __seqWin.org_avg, _G_data_win_size);
	__calc_mean_vec(__seqWin.kmean_org_vol,   __seqWin.org_vol, _G_data_win_size);
	__calc_mean_vec(__seqWin.kmean_adj_open,  __seqWin.adj_open, _G_data_win_size);
	__calc_mean_vec(__seqWin.kmean_adj_close, __seqWin.adj_close, _G_data_win_size);
	__calc_wght_mean_vec(__seqWin.wmean_org_avg, __seqWin.org_avg, __seqWin.org_vol, _G_data_win_size);

	for( int i = 0; i < _G_data_win_size; i++ ) {
		__seqWin.kmean_org_close_open[i]= __seqWin.kmean_org_close[i] / __seqWin.kmean_org_open[i];
		__seqWin.kmean_org_close_high[i]= __seqWin.kmean_org_close[i] / __seqWin.kmean_org_high[i];
		__seqWin.kmean_org_close_low[i] = __seqWin.kmean_org_close[i] / __seqWin.kmean_org_low[i];
		__seqWin.kmean_org_close_avg[i] = __seqWin.kmean_org_close[i] / __seqWin.kmean_org_avg[i];
		__seqWin.kmean_org_high_open[i] = __seqWin.kmean_org_high[i]  / __seqWin.kmean_org_open[i];
		__seqWin.kmean_org_high_low[i]  = __seqWin.kmean_org_high[i]  / __seqWin.kmean_org_low[i];
		__seqWin.kmean_org_high_avg[i]  = __seqWin.kmean_org_high[i]  / __seqWin.kmean_org_avg[i];
		__seqWin.kmean_org_open_low[i]  = __seqWin.kmean_org_open[i]  / __seqWin.kmean_org_low[i];
		__seqWin.kmean_org_open_avg[i]  = __seqWin.kmean_org_open[i]  / __seqWin.kmean_org_avg[i];
		__seqWin.kmean_org_avg_low[i]   = __seqWin.kmean_org_avg [i]  / __seqWin.kmean_org_low[i];
		__seqWin.kmean_adj_close_open[i]= __seqWin.kmean_adj_close[i] / __seqWin.kmean_adj_open[i];

		__seqWin.kmean_org_open_wavg[i] = __seqWin.kmean_org_open[i]  / __seqWin.wmean_org_avg[i];
		__seqWin.kmean_org_close_wavg[i]= __seqWin.kmean_org_close[i] / __seqWin.wmean_org_avg[i];
		__seqWin.kmean_org_high_wavg[i] = __seqWin.kmean_org_high[i]  / __seqWin.wmean_org_avg[i];
		__seqWin.kmean_org_low_wavg[i]  = __seqWin.kmean_org_low[i]   / __seqWin.wmean_org_avg[i];
		__seqWin.kmean_org_avg_wavg[i]  = __seqWin.kmean_org_avg[i]   / __seqWin.wmean_org_avg[i];
	}

	//MUST RETURN 0
	return 0;
}

void do_seqwind( dyncAry<double> &__signal, seqWind &__seqWin )
{
	size_t __cursor = 0;

	//0
	__calc_mtm_log_vec( __seqWin.mtm_vec[__cursor++], __seqWin.org_open, _G_data_win_size );
	__calc_mtm_log_vec( __seqWin.mtm_vec[__cursor++], __seqWin.org_close, _G_data_win_size );
	__calc_mtm_log_vec( __seqWin.mtm_vec[__cursor++], __seqWin.org_high, _G_data_win_size );
	__calc_mtm_log_vec( __seqWin.mtm_vec[__cursor++], __seqWin.org_low, _G_data_win_size );
	__calc_mtm_log_vec( __seqWin.mtm_vec[__cursor++], __seqWin.org_avg, _G_data_win_size );
	__calc_mtm_log_vec( __seqWin.mtm_vec[__cursor++], __seqWin.org_vol, _G_data_win_size );
	__calc_mtm_log_vec( __seqWin.mtm_vec[__cursor++], __seqWin.adj_open, _G_data_win_size );
	__calc_mtm_log_vec( __seqWin.mtm_vec[__cursor++], __seqWin.adj_close, _G_data_win_size );
	__calc_mtm_log_vec( __seqWin.mtm_vec[__cursor++], __seqWin.kmean_org_open, _G_data_win_size );
	__calc_mtm_log_vec( __seqWin.mtm_vec[__cursor++], __seqWin.kmean_org_close, _G_data_win_size );

	//10
	__calc_mtm_log_vec( __seqWin.mtm_vec[__cursor++], __seqWin.kmean_org_high, _G_data_win_size );
	__calc_mtm_log_vec( __seqWin.mtm_vec[__cursor++], __seqWin.kmean_org_low, _G_data_win_size );
	__calc_mtm_log_vec( __seqWin.mtm_vec[__cursor++], __seqWin.kmean_org_avg, _G_data_win_size );
	__calc_mtm_log_vec( __seqWin.mtm_vec[__cursor++], __seqWin.kmean_org_vol, _G_data_win_size );
	__calc_mtm_log_vec( __seqWin.mtm_vec[__cursor++], __seqWin.kmean_adj_open, _G_data_win_size );
	__calc_mtm_log_vec( __seqWin.mtm_vec[__cursor++], __seqWin.kmean_adj_close, _G_data_win_size );
	__calc_mtm_log_vec( __seqWin.mtm_vec[__cursor++], __seqWin.wmean_org_avg, _G_data_win_size );

	__calc_avg_mtm_log_vec( __seqWin.mtm_vec[__cursor++], __seqWin.org_open, _G_data_win_size );
	__calc_avg_mtm_log_vec( __seqWin.mtm_vec[__cursor++], __seqWin.org_close, _G_data_win_size );
	__calc_avg_mtm_log_vec( __seqWin.mtm_vec[__cursor++], __seqWin.org_high, _G_data_win_size );
	//20
	__calc_avg_mtm_log_vec( __seqWin.mtm_vec[__cursor++], __seqWin.org_low, _G_data_win_size );
	__calc_avg_mtm_log_vec( __seqWin.mtm_vec[__cursor++], __seqWin.org_avg, _G_data_win_size );
	__calc_avg_mtm_log_vec( __seqWin.mtm_vec[__cursor++], __seqWin.org_vol, _G_data_win_size );
	__calc_avg_mtm_log_vec( __seqWin.mtm_vec[__cursor++], __seqWin.adj_open, _G_data_win_size );
	__calc_avg_mtm_log_vec( __seqWin.mtm_vec[__cursor++], __seqWin.adj_close, _G_data_win_size );
	__calc_avg_mtm_log_vec( __seqWin.mtm_vec[__cursor++], __seqWin.kmean_org_open, _G_data_win_size );
	__calc_avg_mtm_log_vec( __seqWin.mtm_vec[__cursor++], __seqWin.kmean_org_close, _G_data_win_size );
	__calc_avg_mtm_log_vec( __seqWin.mtm_vec[__cursor++], __seqWin.kmean_org_high, _G_data_win_size );
	__calc_avg_mtm_log_vec( __seqWin.mtm_vec[__cursor++], __seqWin.kmean_org_low, _G_data_win_size );
	__calc_avg_mtm_log_vec( __seqWin.mtm_vec[__cursor++], __seqWin.kmean_org_avg, _G_data_win_size );
	//30
	__calc_avg_mtm_log_vec( __seqWin.mtm_vec[__cursor++], __seqWin.kmean_org_vol, _G_data_win_size );
	__calc_avg_mtm_log_vec( __seqWin.mtm_vec[__cursor++], __seqWin.kmean_adj_open, _G_data_win_size );
	__calc_avg_mtm_log_vec( __seqWin.mtm_vec[__cursor++], __seqWin.kmean_adj_close, _G_data_win_size );
	__calc_avg_mtm_log_vec( __seqWin.mtm_vec[__cursor++], __seqWin.wmean_org_avg, _G_data_win_size );

	__calc_dif_log_vec( __seqWin.mtm_vec[__cursor++], __seqWin.org_close, __seqWin.org_close + 1, _G_data_win_size - 1);
	__calc_dif_log_vec( __seqWin.mtm_vec[__cursor++], __seqWin.org_open, __seqWin.org_open + 1, _G_data_win_size - 1);
	__calc_dif_log_vec( __seqWin.mtm_vec[__cursor++], __seqWin.org_open, __seqWin.org_close + 1, _G_data_win_size -1);
	__calc_dif_log_vec( __seqWin.mtm_vec[__cursor++], __seqWin.org_high, __seqWin.org_close + 1, _G_data_win_size -1);
	__calc_dif_log_vec( __seqWin.mtm_vec[__cursor++], __seqWin.org_avg, __seqWin.org_avg + 1, _G_data_win_size -1);
	__calc_dif_log_vec( __seqWin.mtm_vec[__cursor++], __seqWin.adj_close, __seqWin.adj_close + 1, _G_data_win_size -1);
	//40
	__calc_dif_log_vec( __seqWin.mtm_vec[__cursor++], __seqWin.adj_open, __seqWin.adj_open + 1, _G_data_win_size -1);
	__calc_dif_log_vec( __seqWin.mtm_vec[__cursor++], __seqWin.adj_open, __seqWin.adj_close + 1, _G_data_win_size-1);
	__calc_dif_log_vec( __seqWin.mtm_vec[__cursor++], __seqWin.org_high, __seqWin.adj_close + 1, _G_data_win_size-1 );


	__calc_var_rate_vec( __seqWin.mtm_vec[__cursor++], __seqWin.org_open, _G_data_win_size );
	__calc_var_rate_vec( __seqWin.mtm_vec[__cursor++], __seqWin.org_close, _G_data_win_size );
	__calc_var_rate_vec( __seqWin.mtm_vec[__cursor++], __seqWin.org_high, _G_data_win_size );
	__calc_var_rate_vec( __seqWin.mtm_vec[__cursor++], __seqWin.org_low, _G_data_win_size );
	__calc_var_rate_vec( __seqWin.mtm_vec[__cursor++], __seqWin.org_avg, _G_data_win_size );
	__calc_var_rate_vec( __seqWin.mtm_vec[__cursor++], __seqWin.org_vol, _G_data_win_size );
	__calc_var_rate_vec( __seqWin.mtm_vec[__cursor++], __seqWin.adj_open, _G_data_win_size );
	//50
	__calc_var_rate_vec( __seqWin.mtm_vec[__cursor++], __seqWin.adj_close, _G_data_win_size );

	__calc_var_rate_vec( __seqWin.mtm_vec[__cursor++], __seqWin.kmean_org_open, _G_data_win_size );
	__calc_var_rate_vec( __seqWin.mtm_vec[__cursor++], __seqWin.kmean_org_close,_G_data_win_size );
	__calc_var_rate_vec( __seqWin.mtm_vec[__cursor++], __seqWin.kmean_org_high, _G_data_win_size );
	__calc_var_rate_vec( __seqWin.mtm_vec[__cursor++], __seqWin.kmean_org_low, _G_data_win_size );
	__calc_var_rate_vec( __seqWin.mtm_vec[__cursor++], __seqWin.kmean_org_avg, _G_data_win_size );
	__calc_var_rate_vec( __seqWin.mtm_vec[__cursor++], __seqWin.kmean_org_vol, _G_data_win_size );
	__calc_var_rate_vec( __seqWin.mtm_vec[__cursor++], __seqWin.kmean_adj_open, _G_data_win_size );
	__calc_var_rate_vec( __seqWin.mtm_vec[__cursor++], __seqWin.kmean_adj_close, _G_data_win_size );

	__calc_var_rate_vec( __seqWin.mtm_vec[__cursor++], __seqWin.kmean_org_close_open, _G_data_win_size );
	//60
	__calc_var_rate_vec( __seqWin.mtm_vec[__cursor++], __seqWin.kmean_org_close_high, _G_data_win_size );
	__calc_var_rate_vec( __seqWin.mtm_vec[__cursor++], __seqWin.kmean_org_close_low, _G_data_win_size );
	__calc_var_rate_vec( __seqWin.mtm_vec[__cursor++], __seqWin.kmean_org_close_avg, _G_data_win_size );
	__calc_var_rate_vec( __seqWin.mtm_vec[__cursor++], __seqWin.kmean_org_high_open, _G_data_win_size );
	__calc_var_rate_vec( __seqWin.mtm_vec[__cursor++], __seqWin.kmean_org_high_low, _G_data_win_size );
	__calc_var_rate_vec( __seqWin.mtm_vec[__cursor++], __seqWin.kmean_org_high_avg, _G_data_win_size );
	__calc_var_rate_vec( __seqWin.mtm_vec[__cursor++], __seqWin.kmean_org_open_low, _G_data_win_size );
	__calc_var_rate_vec( __seqWin.mtm_vec[__cursor++], __seqWin.kmean_org_avg_low, _G_data_win_size );
	__calc_var_rate_vec( __seqWin.mtm_vec[__cursor++], __seqWin.kmean_adj_close_open, _G_data_win_size );

	__calc_var_rate_vec( __seqWin.mtm_vec[__cursor++], __seqWin.kmean_org_open_wavg, _G_data_win_size );
	//70
	__calc_var_rate_vec( __seqWin.mtm_vec[__cursor++], __seqWin.kmean_org_close_wavg, _G_data_win_size );
	__calc_var_rate_vec( __seqWin.mtm_vec[__cursor++], __seqWin.kmean_org_high_wavg, _G_data_win_size );
	__calc_var_rate_vec( __seqWin.mtm_vec[__cursor++], __seqWin.kmean_org_low_wavg, _G_data_win_size );
	__calc_var_rate_vec( __seqWin.mtm_vec[__cursor++], __seqWin.kmean_org_avg_wavg, _G_data_win_size );

	__calc_cov_rate_vec( __seqWin.mtm_vec[__cursor++], __seqWin.org_open       , __seqWin.org_vol, _G_data_win_size );
	__calc_cov_rate_vec( __seqWin.mtm_vec[__cursor++], __seqWin.kmean_org_open , __seqWin.org_vol, _G_data_win_size );
	__calc_cov_rate_vec( __seqWin.mtm_vec[__cursor++], __seqWin.adj_open       , __seqWin.org_vol, _G_data_win_size );
	__calc_cov_rate_vec( __seqWin.mtm_vec[__cursor++], __seqWin.kmean_adj_open , __seqWin.org_vol, _G_data_win_size );
	__calc_cov_rate_vec( __seqWin.mtm_vec[__cursor++], __seqWin.org_close      , __seqWin.org_vol, _G_data_win_size );
	__calc_cov_rate_vec( __seqWin.mtm_vec[__cursor++], __seqWin.kmean_org_close, __seqWin.org_vol, _G_data_win_size );
	//80
	__calc_cov_rate_vec( __seqWin.mtm_vec[__cursor++], __seqWin.adj_close      , __seqWin.org_vol, _G_data_win_size );
	__calc_cov_rate_vec( __seqWin.mtm_vec[__cursor++], __seqWin.kmean_adj_close, __seqWin.org_vol, _G_data_win_size );
	__calc_cov_rate_vec( __seqWin.mtm_vec[__cursor++], __seqWin.org_high       , __seqWin.org_vol, _G_data_win_size );
	__calc_cov_rate_vec( __seqWin.mtm_vec[__cursor++], __seqWin.kmean_org_high , __seqWin.org_vol, _G_data_win_size );
	__calc_cov_rate_vec( __seqWin.mtm_vec[__cursor++], __seqWin.org_low        , __seqWin.org_vol, _G_data_win_size );
	__calc_cov_rate_vec( __seqWin.mtm_vec[__cursor++], __seqWin.kmean_org_low  , __seqWin.org_vol, _G_data_win_size );
	__calc_cov_rate_vec( __seqWin.mtm_vec[__cursor++], __seqWin.org_avg        , __seqWin.org_vol, _G_data_win_size );
	__calc_cov_rate_vec( __seqWin.mtm_vec[__cursor++], __seqWin.kmean_org_avg  , __seqWin.org_vol, _G_data_win_size );
	__calc_cov_rate_vec( __seqWin.mtm_vec[__cursor++], __seqWin.wmean_org_avg  , __seqWin.org_vol, _G_data_win_size );

	__calc_cov_rate_vec( __seqWin.mtm_vec[__cursor++], __seqWin.org_open       , __seqWin.kmean_org_vol, _G_data_win_size );
	//90
	__calc_cov_rate_vec( __seqWin.mtm_vec[__cursor++], __seqWin.kmean_org_open , __seqWin.kmean_org_vol, _G_data_win_size );
	__calc_cov_rate_vec( __seqWin.mtm_vec[__cursor++], __seqWin.adj_open       , __seqWin.kmean_org_vol, _G_data_win_size );
	__calc_cov_rate_vec( __seqWin.mtm_vec[__cursor++], __seqWin.kmean_adj_open , __seqWin.kmean_org_vol, _G_data_win_size );
	__calc_cov_rate_vec( __seqWin.mtm_vec[__cursor++], __seqWin.org_close      , __seqWin.kmean_org_vol, _G_data_win_size );
	__calc_cov_rate_vec( __seqWin.mtm_vec[__cursor++], __seqWin.kmean_org_close, __seqWin.kmean_org_vol, _G_data_win_size );
	__calc_cov_rate_vec( __seqWin.mtm_vec[__cursor++], __seqWin.adj_close      , __seqWin.kmean_org_vol, _G_data_win_size );
	__calc_cov_rate_vec( __seqWin.mtm_vec[__cursor++], __seqWin.kmean_adj_close, __seqWin.kmean_org_vol, _G_data_win_size );
	__calc_cov_rate_vec( __seqWin.mtm_vec[__cursor++], __seqWin.org_high       , __seqWin.kmean_org_vol, _G_data_win_size );
	__calc_cov_rate_vec( __seqWin.mtm_vec[__cursor++], __seqWin.kmean_org_high , __seqWin.kmean_org_vol, _G_data_win_size );
	__calc_cov_rate_vec( __seqWin.mtm_vec[__cursor++], __seqWin.org_low        , __seqWin.kmean_org_vol, _G_data_win_size );
	//100
	__calc_cov_rate_vec( __seqWin.mtm_vec[__cursor++], __seqWin.kmean_org_low  , __seqWin.kmean_org_vol, _G_data_win_size );
	__calc_cov_rate_vec( __seqWin.mtm_vec[__cursor++], __seqWin.org_avg        , __seqWin.kmean_org_vol, _G_data_win_size );
	__calc_cov_rate_vec( __seqWin.mtm_vec[__cursor++], __seqWin.kmean_org_avg  , __seqWin.kmean_org_vol, _G_data_win_size );
	__calc_cov_rate_vec( __seqWin.mtm_vec[__cursor++], __seqWin.wmean_org_avg  , __seqWin.kmean_org_vol, _G_data_win_size );

	__calc_cov_rate_vec( __seqWin.mtm_vec[__cursor++], __seqWin.org_close , __seqWin.org_avg, _G_data_win_size );
	__calc_cov_rate_vec( __seqWin.mtm_vec[__cursor++], __seqWin.adj_close , __seqWin.org_avg, _G_data_win_size );
	__calc_cov_rate_vec( __seqWin.mtm_vec[__cursor++], __seqWin.org_close , __seqWin.kmean_org_avg, _G_data_win_size );
	__calc_cov_rate_vec( __seqWin.mtm_vec[__cursor++], __seqWin.adj_close , __seqWin.kmean_org_avg, _G_data_win_size );
	__calc_cov_rate_vec( __seqWin.mtm_vec[__cursor++], __seqWin.org_close , __seqWin.wmean_org_avg, _G_data_win_size );
	__calc_cov_rate_vec( __seqWin.mtm_vec[__cursor++], __seqWin.adj_close , __seqWin.wmean_org_avg, _G_data_win_size );

	//110
	for( size_t i = 0; i < __cursor; i++ ) {
		for( size_t k = 0; k < _G_data_win_size - 1; k++ ) {
			__signal.push_back( __seqWin.mtm_vec[i][k] );
		}
	}
	//RANKING 60 * 2
	//2201-2260
	__calc_rank_sys( __signal, __seqWin.org_close, _G_data_win_size );
	//2261-2320
	__calc_rank_sys( __signal, __seqWin.org_vol,   _G_data_win_size );

	//2320-2360
	__calc_upnum( __signal, __seqWin );

	//2361-2380:
	__calc_RSV  ( __signal, __seqWin );
}


void display( char *__s, size_t __len, float __wght )
{
	fwrite( __s, 1, __len, stdout);
	if( _G_gen_wght ){
		int __sum = 0;
		if( __wght > 0 ) {
			__sum = floor( __wght / 0.01 ) - 1;
		}
		else {
			__sum = floor( fabs( __wght ) / 0.01 ) - 1;
			__sum *= 2;
		}
		for( int i = 0; i < __sum; i++ ) fwrite( __s, 1, __len, stdout);
	}
}

//OUTPUT
void output_seqwin( dyncAry<double> &__signal, seqWind &__seqWin, int __stkno )
{
	static char *__sbin = NULL;

	size_t __size = 0;
	if( __sbin == NULL ) {
		__size = ( __signal.size() * 32 + 1024 + 4095) & ~4095;
		__sbin = (char *)malloc( __size );
		if( __sbin == NULL ) abort();
		memset( __sbin, 0, __size );
	}

	switch( _G_bin_out ) {
	case 2:
		{
			float *__f  = (float *)__sbin;
			*__f++ = __seqWin.dst_gain;
			for( int i = 0; i < __signal.size(); i++ ) {
				*__f++ = __signal[i];
			}
			__size = ( __signal.size() + 1 ) * sizeof(float);
		}
		break;
	case 1:
		{
			float *__f  = (float *)__sbin;
			*__f++ = __seqWin.label;
			for( int i = 0; i < __signal.size(); i++ ) {
				if( _G_select_sig ) { if( _G_sig_sel_buf[i] ) *__f++ = __signal[i]; }
				else { *__f++ = __signal[i]; }
			}
			//COMMENT INFO
			unsigned int *__i = (unsigned int *)__f;
			*__i++ = __stkno;   
			*__i++ = __seqWin.org_dtime;;   
			__f    = (float *)__i;
			*__f++ = __seqWin.dst_gain;
			*__f++ = __seqWin.max_gain;
			*__f++ = __seqWin.min_gain;
			*__f++ = __seqWin.cur_gain;

			__size = (char *)__f - __sbin;
		}
		break;
	case 0:
	default:
		{
			char *__s = __sbin;

			__s += sprintf( __s, "%d\t", __seqWin.label );

			int __sel = 0;
			for( int i = 0; i < __signal.size(); i++ ) {
				if( _G_select_sig ) {
					if( _G_sig_sel_buf[i] ) {
						__s += sprintf( __s, "%d:%g\t", __sel+1, __signal[i] );
						__sel++;
					}
				}
				else {
					__s += sprintf( __s, "%d:%g\t", i+1, __signal[i] );
				}
			}
			__s += sprintf( __s, "#%06d %d %g %g %g %g\n", __stkno, __seqWin.org_dtime,
					__seqWin.dst_gain, __seqWin.max_gain, __seqWin.min_gain, __seqWin.cur_gain );

			__size = __s - __sbin;
		}
		break;
	}

	display( __sbin, __size, __seqWin.dst_gain );
	{
		static size_t __deal_sum = 0;
		int __sig_num = ( _G_select_sig > 0 ) ? _G_select_sig : __signal.size();
		if( __deal_sum % 10000 == 0 ) {
			fprintf( stderr, "[%06d] signum:[%d] record size:[%d] current deal sum:[%d]\n", __stkno, __sig_num, __size, __deal_sum );
			fflush( stderr );
		}
		__deal_sum++;
	}
}

int main(int argc, char **argv)
{
tradeMix __trMix;
seqWind __seqWin;
dyncAry<double> __signal;

	parse_args( argc, argv );
	deal_input( __trMix );

	tradeMix::iterator it;
	for( it = __trMix.begin(); it != __trMix.end(); it++ ) {

		tradeSeq *__trSeq = it->second;

		int __seq_size = (int)__trSeq->org_open.size();

		__seq_size -= _G_data_win_size + _G_dst_span - 1;

		for( int __cursor = 0; __cursor < __seq_size; __cursor++ ) {

			__signal.clear();
			int __skip = predeal_seqwind( __trSeq, __cursor, __seqWin );
			if( __skip == 0 ) {
				do_seqwind( __signal, __seqWin );
				output_seqwin( __signal, __seqWin, it->first );
			}
			else {
				__cursor += __skip - 1;
			}
		}
	}
	fflush( stdout );
	return 0;
}
