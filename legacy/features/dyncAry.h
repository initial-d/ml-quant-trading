///////////////////////////////////////////////////////////////////////////////////////////////////////////
//AUTHOR: DongYi
//FUNCTION: dyncAry
//SIMILAR TO VECTOR
///////////////////////////////////////////////////////////////////////////////////////////////////////////
#ifndef __DYNC_ARY_H_
#define __DYNC_ARY_H_

#include <stdlib.h>
#include <string.h>

static const int __dyncAry_page_num = 1024;

template <class _Tp>
class dyncAry 
{
public:
  typedef _Tp value_type;
  typedef value_type* pointer;
  typedef const value_type* const_pointer;
  typedef value_type& reference;
  typedef const value_type& const_reference;
  typedef size_t size_type;

protected:
	
	int	_M_size;
	int _M_cursor;
	int _M_usedpage;
	int	_M_pagesize;
	int _M_end_of_storage;
	
	pointer	_M_pageary[__dyncAry_page_num];
	value_type _M_finish;
	
public:
	dyncAry( int nPagesize = 4 * 1024 ) 
		: _M_pagesize( nPagesize ), _M_size( 0 ), _M_usedpage(0), _M_cursor(0), _M_end_of_storage(0)
	{
		memset( _M_pageary, 0, sizeof( pointer ) * __dyncAry_page_num );
		if( _M_pagesize <= 1024 ) _M_pagesize = 1024;
	}
  
	~dyncAry()
	{
		clean();
	}

private:
	pointer __pageAlloc( )
	{
		pointer p = _M_pageary[_M_usedpage];
	   	if( p == NULL ) {
			p = (pointer)malloc( _M_pagesize * sizeof( value_type ) );
			_M_pageary[_M_usedpage] = p;
			if( p ) {
				_M_usedpage++;
				_M_end_of_storage = _M_pagesize * _M_usedpage;
				memset( p, 0, _M_pagesize * sizeof( value_type ) );
			}
		}
		return p;
	}

	pointer __Location( int __pos )
	{
		if( __pos >= _M_size ) return &_M_finish;
		int __pageno = __pos / _M_pagesize;
		int __pageoff= __pos % _M_pagesize;
		pointer p = _M_pageary[__pageno];
		return p + __pageoff;
	}
	
public:
	bool reserve( int __n ) 
	{
		if( __n <= _M_end_of_storage ) {
			if( __n > _M_size ) _M_size = __n;
			return true;
		}
		int __pos = __n - 1;
		int __pageno = __pos / _M_pagesize;
		int __pageoff= __pos % _M_pagesize;
	   	if( __pageno >= __dyncAry_page_num ) return false;
		for( ; _M_usedpage <= __pageno; ) {
			if( NULL == __pageAlloc( ) ) { return false; }
		}
		_M_size = __n;
		return true;
	}

	bool insert( int __pos, const _Tp& __x )
	{
		int __pageno = __pos / _M_pagesize;
		int __pageoff= __pos % _M_pagesize;
	   	if( __pageno >= __dyncAry_page_num ) return false;

		for( ; _M_usedpage <= __pageno; ) {
			if( NULL == __pageAlloc( ) ) { return false; }
		}
		pointer p = _M_pageary[__pageno] + __pageoff; *p = __x; 
		if( __pos + 1 > _M_size ) _M_size = __pos + 1;
		return true;
	}
	
	bool push_back( const _Tp& __x) 
	{
		return insert( _M_size , __x );
	}
	
	reference operator[](int __n) { return *__Location( __n ) ; }
	const_reference operator[](int __n) const { return *__Location( __n ); }
	
	pointer operator+(int __n) { return __Location( __n ); }
	const_pointer operator+(int __n) const { return __Location( __n ); }
	
	int size() { return _M_size; }
	
	pointer begin() { _M_cursor = 0; return __Location( 0 ); }
	pointer next()  { return __Location( ++_M_cursor ); }
	pointer end() 	{ return &_M_finish; }
	
	void clear( ) { 
		_M_size = 0; 
		for( int i = 0; i < _M_usedpage; i++ ) {
			memset( _M_pageary[i], 0, sizeof( value_type ) * _M_pagesize );
		}
	}
	void clean( ) {
		_M_size = 0; _M_usedpage = 0;
		for( int i = 0; i < __dyncAry_page_num; i++ ){ 
			if( _M_pageary[i] ) free( _M_pageary[i] ); 
			_M_pageary[i] = NULL;
		}
	}
};

#endif
