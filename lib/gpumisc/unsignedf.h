#ifndef UNSIGNEDF_H
#define UNSIGNEDF_H

#include <cmath>
#include "exceptionassert.h"

/**
  IntegerFloat is used to guarantee the integer precision of a given integer
  type while still allowing fractional parts with a separate internal float.

  TODO handle operations more accurately
  */
template<typename IntegerType, typename FloatType, typename LongerFloatType>
class IntegerFloat
{
public:
    IntegerFloat( int val )
        :   int_part( val ),
            fractional_part( 0 )
    {
        EXCEPTION_ASSERT( val == (int)int_part );
    }


    IntegerFloat( IntegerType val=0, FloatType valf=0 )
        :   int_part( val ),
            fractional_part( valf )
    {
        int_part += floor(valf);
        fractional_part -= floor(valf); // works well for negative numbers too
    }


    IntegerFloat( FloatType val ) {
        FloatType dint;
        fractional_part = std::modf(val, &dint);
        int_part = (IntegerType)dint;
    }


    IntegerFloat( LongerFloatType val ) {
        LongerFloatType dint;
        fractional_part = std::modf(val, &dint);
        int_part = (IntegerType)dint;
    }


    LongerFloatType asFloat() const { return int_part + (LongerFloatType)fractional_part; }
    IntegerType asInteger() const { return int_part; }

    FloatType fractional() const { return fractional_part; }


    IntegerFloat operator+(IntegerFloat const& b) const {  IntegerFloat r = *this; r += b; return r; }
    IntegerFloat& operator+=(IntegerFloat const& b) { int_part += b.int_part; fractional_part += b.fractional_part; evenOut(); return *this; }
    IntegerFloat operator+(long long const& b) const { IntegerFloat r = *this; r += b; return r; }
    IntegerFloat& operator+=(long long const& b) { int_part += b; return *this; }
    IntegerFloat operator+(unsigned long long const& b) const { IntegerFloat r = *this; r += b; return r; }
    IntegerFloat& operator+=(unsigned long long const& b) { int_part += b; return *this; }
    IntegerFloat operator+(int const& b) const { IntegerFloat r = *this; r += b; return r; }
    IntegerFloat& operator+=(int const& b) { int_part += b; return *this; }
    IntegerFloat operator+(unsigned const& b) const { IntegerFloat r = *this; r += b; return r; }
    IntegerFloat& operator+=(unsigned const& b) { int_part += b; return *this; }
    IntegerFloat operator+(FloatType const& b) const { return *this + (LongerFloatType)b; }
    IntegerFloat& operator+=(FloatType const& b) { return *this += (LongerFloatType)b; }
    IntegerFloat operator+(LongerFloatType const& b) const { IntegerFloat r = *this; r += b; return r; }
    IntegerFloat& operator+=(LongerFloatType const& b) {
        IntegerFloat bf(b);
        int_part += bf.int_part;
        fractional_part += bf.fractional_part;
        if (1<=fractional_part) {
            --fractional_part;
            ++int_part;
        }
        if (0>=fractional_part) {
            ++fractional_part;
            --int_part;
        }
        return *this;
    }

    IntegerFloat operator-(IntegerFloat const& b) const {  IntegerFloat r = *this; r -= b; return r; }
    IntegerFloat& operator-=(IntegerFloat const& b) { int_part -= b.int_part; fractional_part -= b.fractional_part; evenOut(); return *this; }
    IntegerFloat operator-(IntegerType const& b) const { IntegerFloat r = *this; r -= b; return r; }
    IntegerFloat& operator-=(IntegerType const& b) { return *this += -b; /* integer underflow on purpose */ }
    IntegerFloat operator-(FloatType const& b) const { return *this - (LongerFloatType)b; }
    IntegerFloat& operator-=(FloatType const& b) { return *this -= (LongerFloatType)b; }
    IntegerFloat operator-(LongerFloatType const& b) const { IntegerFloat r = *this; r -= b; return r; }
    IntegerFloat& operator-=(LongerFloatType const& b) { return *this += -b; }


    IntegerFloat operator>>(int const& b) const { IntegerFloat r = *this; r >>= b; return r; }
    IntegerFloat& operator>>=(int const& b) {
        if (b>=0)
            int_part >>= b;
        else
            int_part <<= -b;
        fractional_part = ldexp( fractional_part, -b );
        evenOut();
        return *this;
    }
    IntegerFloat operator<<(int const& b) const { IntegerFloat r = *this; r <<= b; return r; }
    IntegerFloat& operator<<=(int const& b) { return *this >>= -b; }


    IntegerFloat  operator* (FloatType const& b) const { return asFloat()*b; }
    IntegerFloat& operator*=(FloatType const& b)       { return *this = *this * b; }
    IntegerFloat  operator* (LongerFloatType const& b) const { return asFloat()*b; }
    IntegerFloat& operator*=(LongerFloatType const& b)       { return *this = *this * b; }
    IntegerFloat  operator/ (FloatType const& b) const { return asFloat()/b; }
    IntegerFloat& operator/=(FloatType const& b)       { return *this = *this / b; }
    IntegerFloat  operator/ (LongerFloatType const& b) const { return asFloat()/b; }
    IntegerFloat& operator/=(LongerFloatType const& b)       { return *this = *this / b; }

private:
    void evenOut()
    {
        IntegerFloat bf( fractional_part );
        int_part += bf.int_part;
        fractional_part = bf.fractional_part;
    }

    IntegerType int_part;
    FloatType fractional_part;
};

template<typename I, typename F, typename D>
IntegerFloat<I,F,D> operator-(I const& a, IntegerFloat<I,F,D> const& b) { return IntegerFloat<I,F,D>(a)-b; }

typedef IntegerFloat<long long, float, double> UnsignedF;
#endif // UNSIGNEDF_H
