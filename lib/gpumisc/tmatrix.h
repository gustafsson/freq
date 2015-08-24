#pragma once
#define _USE_MATH_DEFINES
#include <math.h>
#include <ostream>
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
    tmatrix( tmatrix &&a ) = default;
    tmatrix( const tmatrix &b )
    {
        for(int i=0; i<cols; i++)
            m[i] = b[i];
    }
    template<class t2,
             class = typename std::enable_if <std::is_convertible<t2, t>::value>::type>
    explicit tmatrix( const tmatrix<rows,t2,cols> &b )
	{
		for(int i=0; i<cols; i++)
            m[i] = tvector<rows,t>(b[i]);
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

    tmatrix( tvector<rows, t> &&b );
	operator tvector<rows, t>&();
	operator tvector<rows-1, t>();

        template<typename t2 >
        tvector<rows, t2> operator*( const tvector<cols, t2> &n ) const {
            tvector<rows, t2> v;
            for(int a=0; a<rows; a++)
            {
                t f = 0;
                for(int b=0; b<cols; b++)
                    f += m[b][a]*n[b];
                v[a] = f;
            }
            return v;
        }

        template<int cols2, typename t2 >
    tmatrix<rows, t, cols2> operator*( const tmatrix<cols, t2, cols2> &n ) const {
		tmatrix<rows,t,cols2> r;
		for(int a=0; a<cols2; a++)
		for(int b=0; b<rows; b++)
		for(int c=0; c<cols; c++)
			r[a][b] = r[a][b] + m[c][b]*n[a][c];
		return r;
	}
    tmatrix operator*( const t &v ) const {
		tmatrix r;
		for(int a=0; a<cols; a++)
		for(int b=0; b<rows; b++)
			r[a][b] = m[a][b]*v;
		return r;
	}
    tmatrix operator+( const t &v ) const {
		tmatrix r;
		for(int a=0; a<cols; a++)
		for(int b=0; b<rows; b++)
			r[a][b] = m[a][b]+v;
		return r;
	}
    template< typename t2 >
    tmatrix& operator*=( const tmatrix<cols, t2, rows> &n ) {
        return *this = *this * n;
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
    tmatrix<cols,t,rows> transpose() const {
        tmatrix<cols,t,rows> r;
		for(int a=0; a<cols; a++)
		for(int b=0; b<rows; b++)
            r[b][a] = m[a][b];
        return r;
	}
    static tmatrix<4,t,4> rotFpsHead( tvector<3, t> r )
	{
		return
			rot(tvector<3,t>(0,1,0), DEG_TO_RAD(r[1]))*
			rot(tvector<3,t>(1,0,0), DEG_TO_RAD(r[0])) *
			rot(tvector<3,t>(0,0,1), DEG_TO_RAD(r[2]));
	}
    static tmatrix<4,t,4> rotFpsHeadAnti( tvector<3, t> r )
    {
        return
            rot(tvector<3,t>(0,1,0), -DEG_TO_RAD(r[1])) *
            rot(tvector<3,t>(1,0,0), -DEG_TO_RAD(r[0])) *
            rot(tvector<3,t>(0,0,1), -DEG_TO_RAD(r[2]));
    }
    static tmatrix<4,t,4> rot( tvector<3, t> axis, t rad )
	{
		t
            c = cos(rad),
            s = sin(rad),
            i = 1-cos(rad),
            X = axis[0],
            Y = axis[1],
            Z = axis[2];

        // transposed layout
        t p[]=
		{
			i*X*X + c,		i*Y*X + s*Z,		i*Z*X - s*Y,		0,
			i*X*Y - s*Z,		i*Y*Y + c,			i*Z*Y + s*X,		0,
			i*X*Z + s*Y,		i*Y*Z - s*X,		i*Z*Z + c,			0,
			0,						0,						0,						1
		};
		return tmatrix<4,t,4>(p);
	}
    static tmatrix<4,t,4> rot( t deg, t x, t y, t z )
    {
        // glRotate
        return rot(tvector<3,t>(x,y,z), DEG_TO_RAD(deg));
    }

    static tmatrix<4,t,4> translate( tvector<3, t> r )
	{
        // transposed layout
        t p[]=
		{
            1,0,0, 0,
            0,1,0, 0,
            0,0,1, 0,
            r[0], r[1], r[2], 1
		};
        return tmatrix<4,t,4>(p);
	}
    static tmatrix<4,t,4> translate( t x, t y, t z )
    {
        // glTranslate
        return translate(tvector<3,t>(x,y,z));
    }

    static tmatrix<4,t,4> scale( tvector<3, t> r )
    {
        // transposed layout
        t p[]=
        {
            r[0],0,   0,    0,
            0,   r[1],0,    0,
            0,   0,   r[2], 0,
            0,   0,   0,    1
        };
        return tmatrix<4,t,4>(p);
    }
    static tmatrix<4,t,4> scale( t x, t y, t z )
    {
        // glScale
        return scale(tvector<3,t>(x,y,z));
    }

    t* v() { return m[0].v; }
    const t* v() const { return m[0].v; }

private:
    tvector<rows, t> m[cols];
};

template<>
inline tmatrix<3,double,1>::tmatrix( tvector<3, double> &&b )
    : m{b}
{
}
template<>
inline tmatrix<3,float,1>::tmatrix( tvector<3, float> &&b )
    : m{b}
{
}
template<>
inline tmatrix<4,double,1>::tmatrix( tvector<4, double> &&b )
    : m{b}
{
}
template<>
inline tmatrix<4,float,1>::tmatrix( tvector<4, float> &&b )
    : m{b}
{
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

template<class st, int rows, typename t, int cols=rows>
std::basic_ostream<st>& operator<<(std::basic_ostream<st>& o, const tmatrix<rows,t,cols>& M) {
    for (int i=0; i<4; i++)
    {
        for (int j=0; j<4; j++)
            o << M[i][j] << " ";
        o << std::endl;
    }
    return o;
}
