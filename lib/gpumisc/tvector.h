/*
I really don't need another tvector class, but I didn't bother trying to find my old one. 
This tvector class just happened to be partly implemented by accident just because I found it useful and added functions.
hmm, I'm going to need a matrix class for rotating vectors quite soon...
*/
#pragma once

#include <cmath>

/**
  See tvectorstring.h

  TODO is "for(int i=N; i--;)" faster than "for(int i=0; i<N; i++)"?
  Isn't linear access faster?
*/
template<int N, typename type=float, typename baseType = type>
class tvector
{
public:
    typedef type T;
    typedef baseType baseT;

    type v[N];
	tvector( ) { for(int i=N; i--;) v[i] = 0; }
    tvector( tvector &&a ) = default;
    tvector( const tvector &a ) = default;
    template<typename t2>
    explicit tvector( const tvector<N, t2> &a ) { for(int i=N; i--;) v[i] = a[i]; }
	tvector( const baseType *a ) { for(int i=N; i--;) v[i] = a[i]; }
	tvector( const type& x );
	tvector( const type& x, const type& y );
	tvector( const type& x, const type& y, const type& z );
        tvector( const type& x, const type& y, const type& z, const type& w );
    tvector& operator=( tvector &&a ) = default;
    tvector& operator=( const tvector &a ) = default;
        const type& operator[](const unsigned n) const { return v[n];}
        type& operator[](const unsigned n){ return v[n];}
        bool operator==(const tvector &b) const {
                bool r = true;
                for(int i=N; i--;) r &= v[i] == b[i];
                return r;
        }
        bool operator!=(const tvector &b) const {
            return !(*this==b);
        }
        tvector operator-(tvector const& b) const {
        tvector r;
                for(int i=N; i--;) r[i] = v[i] - b[i];
		return r;
	}
	template<typename t2>
	tvector operator*(const tvector<N,t2>& b) const {
        tvector r;
                for(int i=N; i--;) r[i] = v[i] * b[i];
		return r;
	}
	template<typename t2>
	tvector operator/(const tvector<N,t2>& b) const {
                tvector r;
                for(int i=N; i--;) r[i] = v[i] / b[i];
		return r;
	}
    template<typename t2>
    tvector operator+(const tvector<N,t2>& b) const {
                tvector r;
                for(int i=N; i--;) r[i] = v[i] + b[i];
        return r;
    }
    template<typename t2>
    tvector& operator+=(const tvector<N,t2>& b) {
        for(int i=N; i--;) v[i] += b[i];
        return *this;
    }
    template<typename t2>
    tvector& operator-=(const tvector<N,t2>& b) {
        for(int i=N; i--;) v[i] -= b[i];
        return *this;
    }
    tvector operator*(type b) const {
                tvector r;
                for(int i=N; i--;) r[i] = v[i] * b;
        return r;
    }
    tvector operator*=(type b) {
        for(int i=N; i--;) v[i] *= b;
        return *this;
    }
    tvector operator-() const {
                tvector r;
                for(int i=N; i--;) r[i] = -v[i];
		return r;
	}
        tvector operator<<(unsigned n ) const
        {
                tvector r;
                for(int i=N; i--;) r[i] = v[i] << n;
                return r;
        }
        tvector operator>>( unsigned n ) const
        {
                tvector r;
                for(int i=N; i--;) r[i] = v[i] >> n;
                return r;
        }

	baseType dot() const {
		return (*this)%(*this);
	}
    baseType length() const {
        return std::sqrt(dot());
    }
#ifdef rsqrt
        // gcc: -mrecip
        baseType rlength() const {
                return rsqrt(dot());
        }
#else
        baseType rlength() const {
                return 1/length();
        }
#endif
        baseType operator%(const tvector &b) const {
		baseType r = 0;
		for(int i=N; i--;) r=r+v[i]*b[i];
		return r;
	}

        tvector& Normalized()
        {
                return *this = (*this) * rlength();
        }
};

template<>
inline tvector<2, int>::tvector( const int& x, const int& y ) { v[0] = x, v[1] = y; }
template<>
inline tvector<4, int>::tvector( const int& x, const int& y, const int& z, const int& w ) { v[0] = x, v[1] = y; v[2] = z; v[3] = w; }
template<>
inline tvector<2, unsigned>::tvector( const unsigned& x, const unsigned& y ) { v[0] = x, v[1] = y; }
template<>
inline tvector<1, double>::tvector( const double& x ) { v[0] = x; }
template<>
inline tvector<2, double>::tvector( const double& x, const double& y ) { v[0] = x, v[1] = y; }
template<>
inline tvector<3, double>::tvector( const double& x, const double& y, const double& z ) { v[0] = x, v[1] = y, v[2] = z; }
template<>
inline tvector<4, double>::tvector( const double& x, const double& y, const double& z, const double& w ) { v[0] = x, v[1] = y, v[2] = z, v[3] = w; }
template<>
inline tvector<1, float>::tvector( const float& x ) { v[0] = x; }
template<>
inline tvector<2, float>::tvector( const float& x, const float& y ) { v[0] = x, v[1] = y; }
template<>
inline tvector<3, float>::tvector( const float& x, const float& y, const float& z ) { v[0] = x, v[1] = y, v[2] = z; }
template<>
inline tvector<4, float>::tvector( const float& x, const float& y, const float& z, const float& w ) { v[0] = x, v[1] = y, v[2] = z; v[3] = w; }
// 3d-vectors have the ^ operator defined as cross product
template<typename T>
inline tvector<3, T> operator^(const tvector<3, T> &v, const tvector<3, T> &b) {
    tvector<3, T> r;
    r[0] = v[1]*b[2] - v[2]*b[1];
    r[1] = v[2]*b[0] - v[0]*b[2];
    r[2] = v[0]*b[1] - v[1]*b[0];
    return r;
}
template<typename T>
inline T operator^(const tvector<2, T> &v, const tvector<2, T> &b) {
    return v[0]*b[1] - v[1]*b[0];
}
