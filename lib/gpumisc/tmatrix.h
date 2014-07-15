#pragma once
#define _USE_MATH_DEFINES
#include <math.h>
#include "tvector.h"

#define DEG_TO_RAD(x) ((x)*(M_PI/180.))
#define RAD_TO_DEG(x) ((x)*(180./M_PI))

template<int rows, typename t, int cols=rows>
class tmatrix
{
public:
	tmatrix( ) {}
/*	tmatrix( const tmatrix<rows, t, cols -1> &b )
	{
		*this = identity();
		for(int i=0; i<cols-1; i++)
			m[i] = b[i];
	}*/
	tmatrix( const tmatrix<rows-1, t, cols> &b )
	{
		*this = identity();
		for(int i=0; i<cols; i++)
		for(int a=0; a<rows-1; a++)
			m[i][a] = b[i][a];
	}
	/*tmatrix( const tmatrix<rows-1, t, cols -1> &b )
	{
		*this = identity();
		for(int i=0; i<cols-1; i++)
			m[i] = b[i];
	}*/
	tmatrix( const t *b )
	{
		for(int i=0; i<cols; i++)
            m[i] = b + i*rows;
	}
	tmatrix( const tmatrix &b )
	{
		for(int i=0; i<cols; i++)
			m[i] = b[i];
	}
	static tmatrix identity()
	{
		tmatrix m;
		for( int i=0; i<rows && i<cols; i++)
			m[i][i]=1;
		return m;
	}
	tvector<rows, t>& operator[](unsigned i){return m[i];}
	const tvector<rows, t>& operator[] (unsigned i)const {return m[i];}

	tmatrix( const tvector<rows, t> &b );
	operator tvector<rows, t>&();
	operator tvector<rows-1, t>();

        template<typename t2 >
        tvector<rows, t> operator*( const tvector<cols, t2> &n ) {
                return *this * tmatrix<cols,t2,1>( n );
        }

        template<int cols2, typename t2 >
	tmatrix<rows, t, cols2> operator*( const tmatrix<cols, t2, cols2> &n ) {
		tmatrix<rows,t,cols2> r;
		for(int a=0; a<cols2; a++)
		for(int b=0; b<rows; b++)
		for(int c=0; c<cols; c++)
			r[a][b] = r[a][b] + m[c][b]*n[a][c];
		return r;
	}
    tmatrix operator*( const t &v ) {
		tmatrix r;
		for(int a=0; a<cols; a++)
		for(int b=0; b<rows; b++)
			r[a][b] = m[a][b]*v;
		return r;
	}
    tmatrix operator+( const t &v ) {
		tmatrix r;
		for(int a=0; a<cols; a++)
		for(int b=0; b<rows; b++)
			r[a][b] = m[a][b]+v;
		return r;
	}
    template<typename T2>
    bool operator==( const tmatrix<rows,T2,cols> &v ) const {
        for(int a=0; a<cols; a++)
        for(int b=0; b<rows; b++)
            if (m[a][b] != v[a][b])
                return false;
        return true;
    }
    template<typename T2>
    bool operator!=( const tmatrix<rows,T2,cols> &v ) const {
        return !(*this == v);
    }
    tmatrix<cols,t,rows> transpose() {
        tmatrix<cols,t,rows> r;
		for(int a=0; a<cols; a++)
		for(int b=0; b<rows; b++)
            r[b][a] = m[a][b];
	}
	static tmatrix<4,t,4> rotHead( tvector<3, t> r )
	{
		return
			rot(tvector<3,t>(0,1,0), DEG_TO_RAD(r[1]))*
			rot(tvector<3,t>(1,0,0), DEG_TO_RAD(r[0])) *
			rot(tvector<3,t>(0,0,1), DEG_TO_RAD(r[2]));
	}
	static tmatrix<4,t,4> rotHeadAnti( tvector<3, t> r )
	{
		return
			rot(tvector<3,t>(0,1,0), -DEG_TO_RAD(r[1])) *
			rot(tvector<3,t>(1,0,0), -DEG_TO_RAD(r[0])) *
			rot(tvector<3,t>(0,0,1), -DEG_TO_RAD(r[2]));
	}
	static tmatrix<4,t,4> rot( tvector<3, t> r, t d )
	{
		t
			c = cos(d),
			s = sin(d),
			i = 1-cos(d),
			X = r[0],
			Y = r[1],
			Z = r[2];

		t p[]=
		{
			i*X*X + c,		i*Y*X + s*Z,		i*Z*X - s*Y,		0,
			i*X*Y - s*Z,		i*Y*Y + c,			i*Z*Y + s*X,		0,
			i*X*Z + s*Y,		i*Y*Z - s*X,		i*Z*Z + c,			0,
			0,						0,						0,						1
		};
		return tmatrix<4,t,4>(p);
	}
	static tmatrix<4,t,4> move( tvector<3, t> r )
	{
		double p[]=
		{
			1,0,0, r[0],
			0,1,0, r[1],
			0,0,1, r[2],
			0,0,0, 1
		};
		return tmatrix<4,t,4>(p);
	}

    t* v() { return m[0].v; }
    const t* v() const { return m[0].v; }

private:
    tvector<rows, t> m[cols];
};

template<>
inline tmatrix<3,double,1>::tmatrix( const tvector<3, double> &b )
{
	m[0] = b;
}
template<>
inline tmatrix<3,float,1>::tmatrix( const tvector<3, float> &b )
{
	m[0] = b;
}
template<>
inline tmatrix<4,double,1>::tmatrix( const tvector<4, double> &b )
{
	m[0] = b;
}
template<>
inline tmatrix<4,double,1>::operator tvector<4, double>&()
{
	return m[0];
}
template<>
inline tmatrix<4,float,1>::operator tvector<4, float>&()
{
	return m[0];
}

template<>
inline tmatrix<4,double,1>::operator tvector<3, double>()
{
	return tvector<3, double>(&m[0][0]);
}
template<>
inline tmatrix<4,float,1>::operator tvector<3, float>()
{
	return tvector<3, float>(&m[0][0]);
}
