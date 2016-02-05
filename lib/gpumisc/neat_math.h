#pragma once

// random collection of stuff
// more here for instance: http://www.catch22.net/tuts/c-c-tricks

#include <cstddef> // size_t
#include <cmath> // frexp
#include <limits>

// stdint
#ifdef _MSC_VER
typedef unsigned __int64 uint64_t;
typedef unsigned __int32 uint32_t;
typedef __int64 __int64_t;
#else
#include <stdint.h>
#endif

#include "exceptionassert.h"


#ifdef __CUDACC__
#define NEAT_MATH_CALL static inline __device__ __host__
#else
#define NEAT_MATH_CALL static inline
#endif


// integer division, rounded off upwards
NEAT_MATH_CALL size_t int_div_ceil( const size_t& x, const unsigned& y ) {
    return (x+y-1)/y;
}

NEAT_MATH_CALL int int_div_ceil( const int& x, const unsigned& y ) {
    return (x+y-1)/y;
}

template<typename T>
NEAT_MATH_CALL T absint(T i) {
    EXCEPTION_ASSERT( i != std::numeric_limits<T>::min() );
    return i < 0 ? -i : i;
}

template<>
inline unsigned absint(unsigned i) {
    return i;
}

template<>
inline long unsigned absint(long unsigned i) {
    return i;
}

template<>
inline long long unsigned absint(long long unsigned i) {
    return i;
}


template<typename T>
NEAT_MATH_CALL T isneg(T i) {
    return i < 0;
}

template<>
inline unsigned isneg(unsigned /*i*/) {
    return false;
}

template<>
inline long unsigned isneg(long unsigned /*i*/) {
    return false;
}

template<>
inline long long unsigned isneg(long long unsigned /*i*/) {
    return false;
}


/**
 * Align integer 'x' to a multiple of divisor 'y' towards negative infinity but
 * within range of the type of 'x'.
 *
 * The complete range of x is valid.
 * y==0 and y==MIN_VALUE is invalid.
 * An exception is thrown on invalid input.
 *
 * Correctness for complete range is favored over performance.
 */
template <typename T>
NEAT_MATH_CALL T align_down( T x, T y ) {
    EXCEPTION_ASSERT( y != 0 );
    EXCEPTION_ASSERT( y != std::numeric_limits<T>::min() );

    y = absint(y);
    if (2*y < y)
        return x != absint(x) ? -y : x < y ? 0 : y;

    T d = ((x%y) + y)%y;
    d -= (x < std::numeric_limits<T>::min()+d) * y; // avoid overflow
    return x - d;
}


/**
 * Align integer 'x' to a multiple of divisor 'y' towards positive infinity but
 * within range of the type of 'x'.
 *
 * The complete range of x is valid.
 * y==0 and y==MIN_VALUE is invalid.
 * An exception is thrown on invalid input.
 *
 * Correctness for complete range and clamping at min and max is favored over performance.
 */
template <typename T>
NEAT_MATH_CALL T align_up( T x, T y ) {
    EXCEPTION_ASSERT( y != 0 );
    EXCEPTION_ASSERT( y != std::numeric_limits<T>::min() );

    y = absint(y);
    if (2*y < y)
        return x <= -y ? -y : x <= 0 ? 0 : y;

    if ( x > std::numeric_limits<T>::max() - (y-1)) // handle overflow
    {
        T d = x%y;
        if ( x + y - d < x)
            return x - x%y;
        else
            return x + y - d;
    }
    else
    {
        T d = (((x+y-1)%y) + y)%y;
        return x + y - 1 - d;
    }
}


template<typename T>
int inline floor_log2(T f)
{
    int e;
    frexp(f, &e);
    return e - 1;
}


// Smallest power of two strictly greater than x
unsigned int inline
spo2g(register unsigned int x)
{
    x |= (x >> 1);
    x |= (x >> 2);
    x |= (x >> 4);
    x |= (x >> 8);
    x |= (x >> 16);
    return x + 1;
}


// Largest power of two strictly smaller than x
unsigned int inline
lpo2s(register unsigned int x)
{
    return spo2g(x-1) >> 1;
}


#if defined(_WIN32) || !defined(DARWIN_NO_CARBON)
static inline uint32_t log2(uint32_t x) {
  uint32_t y;
#ifdef _WIN32
  __asm
  {
      bsr eax, x
      mov y, eax
  }
//#elif defined(DARWIN_NO_CARBON)
 // Use <cmath> instead
//  y = 0;
//  while(x>>=1) y++;
#else
  asm ( "\tbsr %1, %0\n"
      : "=r"(y)
      : "r" (x)
  );
#endif
  return y;
}
#else
static inline uint32_t log2(uint32_t x) {
    return log2(double(x));
}
#endif


// Quadratic interpolation
float quad_interpol(float i, float* v, unsigned N, unsigned stride = 1,
                    float* local_max_i=0);

// round
// http://blog.frama-c.com/index.php?post/2013/05/02/nearbyintf1
// also see #include <boost/math/special_functions/round.hpp>
// boost::math::round
// boost::math::iround
/*static inline float frama_round(float f)
{
  // 0x1.0p23 is not supported by msvc. Have not tested if 1.0e23f yields the exact same result.
  if (f >= 0x1.0p23) return f;
  return (float) (unsigned int) (f + 0.49999997f);
}*/

template<typename T> T clamped_add(T a, T b) {
    if (isneg(b) && a < std::numeric_limits<T>::min() - b) {
        return std::numeric_limits<T>::min();
    }
    if (!isneg(b) && a > std::numeric_limits<T>::max() - b) {
        return std::numeric_limits<T>::max();
    }

    return a + b;
}

template<typename T> T clamped_sub(T a, T b) {
    if (!isneg(b) && a < std::numeric_limits<T>::min() + b) {
        return std::numeric_limits<T>::min();
    }
    if (isneg(b) && a > std::numeric_limits<T>::max() + b) {
        return std::numeric_limits<T>::max();
    }
    return a - b;
}

template<typename T>
void swap_plus( T& x, T& y) {
    x = x + y;
    y = x - y;
    x = x - y;
}

class neat_math {
public:
    static void test();
};
