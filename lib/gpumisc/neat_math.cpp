#include "neat_math.h"
#include "exceptionassert.h"
#include "timer.h"
#include "detectgdb.h"

#include <random>

using namespace std;

float quad_interpol(float i, float* v, unsigned N, unsigned stride,
                    float* local_max_i)
{
    EXCEPTION_ASSERT( N > 2 );
    EXCEPTION_ASSERT( i < N );
    EXCEPTION_ASSERT( 0 <= i);

    unsigned y0 = (unsigned)(i + .5f);
    unsigned yb = y0;
    if (0==yb) yb++;
    if (N==yb+1) yb--;
    const float v1 = v[ (yb-1)*stride ];
    const float v2 = v[ yb*stride ];
    const float v3 = v[ (yb+1)*stride ];

    // v1 = a*(-1)^2 + b*(-1) + c
    // v2 = a*(0)^2 + b*(0) + c
    // v3 = a*(1)^2 + b*(1) + c
    const float k = 0.5f*v1 - v2 + 0.5f*v3;
    const float p = -0.5f*v1 + 0.5f*v3;
    const float q = v2;

    float m0;
    if (0==local_max_i)
    {
        m0 = i - yb;
    }
    else
    {
        // fetch max
        m0 = -p/(2*k);
    }

    float r;
    if (m0 > -2 && m0 < 2)
    {
        r = k*m0*m0 + p*m0 + q;
        if (local_max_i)
            *local_max_i = y0 + m0;
    }
    else
    {
        r = v2;
        if (local_max_i)
            *local_max_i = (float)y0;
    }

    return r;
}

void neat_math::
        test()
{
#ifdef _DEBUG
    bool debug_build = true;
#else
    bool debug_build = false;
#endif
    bool gdb = DetectGdb::is_running_through_gdb();

    // It should compute correct answers for the complete range of inputs
    {
        Timer t;

        // integer division, rounded off upwards
        EXCEPTION_ASSERT_EQUALS( int_div_ceil(10,3), 4u );
        EXCEPTION_ASSERT_EQUALS( int_div_ceil(9,3), 3u );
        EXCEPTION_ASSERT_EQUALS( int_div_ceil(8,3), 3u );

        double T = t.elapsed();
        EXCEPTION_ASSERT_LESS (T, gdb ? 2e-3 : 20e-6);
    }

    {
        Timer t;
        EXCEPTION_ASSERT_EQUALS( ((-10%3)+3)%3, 2 );
        EXCEPTION_ASSERT_EQUALS( 1%3u, 1u );
        EXCEPTION_ASSERT_EQUALS( 0%3u, 0u );
        EXCEPTION_ASSERT_EQUALS( -1%3, -1 );
        EXCEPTION_ASSERT_EQUALS( -2%3, -2 );
        EXCEPTION_ASSERT_EQUALS( -3%3, 0 );

        // integer align to a multiple of y towards negative infinity
        EXCEPTION_ASSERT_EQUALS( align_down(10,3), 9 );
        EXCEPTION_ASSERT_EQUALS( align_down(9,3), 9 );
        EXCEPTION_ASSERT_EQUALS( align_down(8,3), 6 );
        EXCEPTION_ASSERT_EQUALS( align_down(-10,3), -12 );
        EXCEPTION_ASSERT_EQUALS( align_down(-9,3), -9 );
        EXCEPTION_ASSERT_EQUALS( align_down(-8,3), -9 );
        EXCEPTION_ASSERT_EQUALS( align_down(LLONG_MIN,3ll), LLONG_MIN+2 );
        EXCEPTION_ASSERT_EQUALS( align_down(LLONG_MIN+1,3ll), LLONG_MIN+2 );
        EXCEPTION_ASSERT_EQUALS( align_down(LLONG_MIN,4ll), LLONG_MIN );
        EXCEPTION_ASSERT_EQUALS( align_down(LLONG_MIN+1,4ll), LLONG_MIN );
        EXCEPTION_ASSERT_EQUALS( align_down(LLONG_MAX,3ll), LLONG_MAX-1 );
        EXCEPTION_ASSERT_EQUALS( align_down(LLONG_MAX-1,3ll), LLONG_MAX-1 );
        EXCEPTION_ASSERT_EQUALS( align_down(LLONG_MAX,4ll), LLONG_MAX-3 );
        EXCEPTION_ASSERT_EQUALS( align_down(LLONG_MAX-1,4ll), LLONG_MAX-3 );

        // integer align to a multiple of y towards positive infinity
        EXCEPTION_ASSERT_EQUALS( align_up(10,3), 12 );
        EXCEPTION_ASSERT_EQUALS( align_up(9,3), 9 );
        EXCEPTION_ASSERT_EQUALS( align_up(8,3), 9 );
        EXCEPTION_ASSERT_EQUALS( align_up(-10,3), -9 );
        EXCEPTION_ASSERT_EQUALS( align_up(-9,3), -9 );
        EXCEPTION_ASSERT_EQUALS( align_up(-8,3), -6 );
        EXCEPTION_ASSERT_EQUALS( align_up(LLONG_MIN,3ll), LLONG_MIN+2 );
        EXCEPTION_ASSERT_EQUALS( align_up(LLONG_MIN+1,3ll), LLONG_MIN+2 );
        EXCEPTION_ASSERT_EQUALS( align_up(LLONG_MIN,4ll), LLONG_MIN );
        EXCEPTION_ASSERT_EQUALS( align_up(LLONG_MIN+1,4ll), LLONG_MIN+4 );
        EXCEPTION_ASSERT_EQUALS( align_up(LLONG_MAX,3ll), LLONG_MAX-1 );
        EXCEPTION_ASSERT_EQUALS( align_up(LLONG_MAX-1,3ll), LLONG_MAX-1 );
        EXCEPTION_ASSERT_EQUALS( align_up(LLONG_MAX,4ll), LLONG_MAX-3 );
        EXCEPTION_ASSERT_EQUALS( align_up(LLONG_MAX-1,4ll), LLONG_MAX-3 );
        EXCEPTION_ASSERT_EQUALS( align_up(7,-9), 9 );
        EXCEPTION_ASSERT_EQUALS( align_up(7,3), 9 );
        EXCEPTION_ASSERT_EQUALS( align_down(7u,3u), 6u );
        EXCEPTION_ASSERT_EQUALS( align_up(7u,3u), 9u );

        EXCEPTION_ASSERT_EQUALS( align_down(-8616761059752331528ll, 2456955197560417229ll), -7370865592681251687ll);
        EXCEPTION_ASSERT_EQUALS( align_up(-8616761059752331528ll, 2456955197560417229ll), -7370865592681251687ll);
        EXCEPTION_ASSERT_EQUALS( align_down(-8616761059752331528ll, -2456955197560417229ll), -7370865592681251687ll);
        EXCEPTION_ASSERT_EQUALS( align_up(-8616761059752331528ll, -2456955197560417229ll), -7370865592681251687ll);

        EXCEPTION_ASSERT_EQUALS( align_down(7615252200817428560, 3755833054923903685), 7511666109847807370);
        EXCEPTION_ASSERT_EQUALS( align_down(7615252200817428560llu, 3755833054923903685llu), 7511666109847807370llu);
        EXCEPTION_ASSERT_EQUALS( align_up(7615252200817428560, 3755833054923903685), 7511666109847807370);
        EXCEPTION_ASSERT_EQUALS( align_up(7615252200817428560llu, 3755833054923903685llu), 11267499164771711055llu);

        double T_align = t.elapsed();
        EXCEPTION_ASSERT_LESS (T_align, gdb ? 2e-3 : 15e-6);

        long long l = LLONG_MIN;
        EXCEPTION_ASSERT_EQUALS( l, -l );
        EXCEPTION_ASSERT_EQUALS( -LLONG_MAX, LLONG_MIN+1 );

        // in general: align_up(x,y) == -align_down(-x,y), except for extreme cases
        // align_up(x,y) == align_up(x,-y) for all valid x,y
        // align_down(x,y) == align_down(x,-y) for all valid x,y
        typedef long long T;
        typedef unsigned long long Tu;
        uniform_int_distribution<T> rTx( numeric_limits<T>::min(), numeric_limits<T>::max());
        uniform_int_distribution<T> rTy( numeric_limits<T>::min()+1, numeric_limits<T>::max());
        uniform_int_distribution<Tu> rTxu( numeric_limits<Tu>::min(), numeric_limits<Tu>::max()/2);
        uniform_int_distribution<Tu> rTyu( numeric_limits<Tu>::min()+1, numeric_limits<Tu>::max()/2-1);
        default_random_engine re;
        for (int i=0;i<100;i++)
        {
            T x = rTx(re);
            T y = 0; while(y==0) y = rTy(re);
            EXCEPTION_ASSERTX( align_up(x,y) == -align_down((T)-x,y),
                boost::format("x=%d, y=%d, align_up(x,y)=%d, -align_down(-x,y)=%d")
                               % x % y
                               % align_up(x,y)
                               % -align_down((T)-x,y));
        }
        for (int i=0;i<100;i++)
        {
            Tu x = rTxu(re);
            Tu y = rTyu(re);
            EXCEPTION_ASSERTX( T(align_down(x,y)) == align_down(T(x),T(y)),
                boost::format("x=%d, y=%d, align_down(x,y)=%d, align_down(T(x),T(y))=%d")
                               % x % y
                               % align_down(x,y)
                               % align_down(T(x),T(y)));

            if (T(x+y)>T(x))
                EXCEPTION_ASSERTX( T(align_up(x,y)) == align_up(T(x),T(y)),
                    boost::format("x=%d, y=%d, align_up(x,y)=%d, align_up(T(x),T(y))=%d")
                                   % x % y
                                   % align_up(x,y)
                                   % align_up(T(x),T(y)));
        }

        T_align = t.elapsed();
        EXCEPTION_ASSERT_LESS (T_align, gdb ? 3e-3 : 0.2e-3);
    }

    {
        Timer t;

        EXCEPTION_ASSERT_EQUALS( floor_log2(1567.f), 10 );
        EXCEPTION_ASSERT_EQUALS( floor_log2(516.0), 9 );
        EXCEPTION_ASSERT_EQUALS( floor_log2(511.f), 8 );
        EXCEPTION_ASSERT_EQUALS( floor_log2(512.0), 9 );
        EXCEPTION_ASSERT_EQUALS( floor_log2(2107612212.f), 30 );
        EXCEPTION_ASSERT_EQUALS( floor_log2(2107612212123456789012.0), 70 );
        EXCEPTION_ASSERT_EQUALS( floor_log2(2107612212123456789012.f), 70 );
        EXCEPTION_ASSERT_EQUALS( floor_log2(-2107612212123456789012.0), 70 );
        EXCEPTION_ASSERT_EQUALS( floor_log2(-2107612212123456789012.f), 70 );

        double T = t.elapsed();
        EXCEPTION_ASSERT_LESS (T, 25e-6);

        // in general floor(log2(x)) == floor_log2(x)
        uniform_real_distribution<double> unif(
                    numeric_limits<double>::min(),
                    numeric_limits<double>::max());
        default_random_engine re;
        for (int i=0;i<100;i++)
        {
            double x = unif(re);

            EXCEPTION_ASSERTX( floor(log2(x)) == floor_log2(x), boost::format("x=%d") % x);
        }

        t.restart ();
        for (int i=0;i<1000;i++)
            T += floor(log2(2134567.232435+i));
        double T1 = t.elapsed ();

        t.restart ();
        for (int i=0;i<1000;i++)
            T += floor_log2(2134567.232435+i);
        double T2 = t.elapsed ();

        T2 += T*1e-10; // prevent removal of for-loop
#ifdef __GCC__
        EXCEPTION_ASSERT_LESS(debug_build ? T2*1.2 : T2*1.6, T1); // floor_log2 is at least a little faster
#else
        EXCEPTION_ASSERT_LESS(debug_build ? T2*0 : T2*1.1, T1); // floor_log2 is at least a little faster in release builds
#endif
        EXCEPTION_ASSERT_LESS(T1, debug_build ? T2*4 : T2*1.8); // how much faster?
    }

    {
        Timer t;

        EXCEPTION_ASSERT_EQUALS( spo2g(1567), 2048u );
        EXCEPTION_ASSERT_EQUALS( spo2g(516), 1024u );
        EXCEPTION_ASSERT_EQUALS( spo2g(511), 512u );
        EXCEPTION_ASSERT_EQUALS( spo2g(512), 1024u );
        EXCEPTION_ASSERT_EQUALS( spo2g(2107612212), 1u<<31 );
        EXCEPTION_ASSERT_EQUALS( lpo2s(1567), 1024u );
        EXCEPTION_ASSERT_EQUALS( lpo2s(516), 512u );
        EXCEPTION_ASSERT_EQUALS( lpo2s(511), 256u );
        EXCEPTION_ASSERT_EQUALS( lpo2s(512), 256u );
        EXCEPTION_ASSERT_EQUALS( lpo2s(2107612212), 1u<<30 );

        double T = t.elapsed();
        EXCEPTION_ASSERT_LESS (T, 25e-6);
    }

    {
        Timer t;

        EXCEPTION_ASSERT_EQUALS( log2(1567u), 10u );
        EXCEPTION_ASSERT_EQUALS( log2(516u), 9u );
        EXCEPTION_ASSERT_EQUALS( log2(511u), 8u );
        EXCEPTION_ASSERT_EQUALS( log2(512u), 9u );
        EXCEPTION_ASSERT_EQUALS( log2(2107612212u), 30u );

        double T = t.elapsed();
        EXCEPTION_ASSERT_LESS (T, debug_build ? 10e-6 : 5e-6);

        // in general log2(x) == floor_log2(x) if x is uint32_t
        srand(0);
        for (int i=0;i<100;i++)
        {
            unsigned x = rand() + rand()*RAND_MAX;
            EXCEPTION_ASSERTX( log2(x) == (unsigned)floor_log2(x), boost::format("x=%d") % x);
        }
    }

    {
        // time for int iterator
        Timer t;
        int j=0;
        for (int i=0; i<1000000; i++) {j+=i;}
        double T = t.elapsed()/1000000;
        T *= 1 + j*1e-30;

        // time for unsigned iterator
        t.restart ();
        unsigned ju=0;
        for (unsigned i=0; i<1000000u; i++) {ju+=i;}
        double T2 = t.elapsed()/1000000;
        T2 *= 1 + ju*1e-30;

        if (debug_build) {
            EXCEPTION_ASSERT_LESS(T, T2*1.3);
            EXCEPTION_ASSERT_LESS(T2, T*2);
        } else {
            EXCEPTION_ASSERT_LESS_OR_EQUAL(T, 1e-12);
            EXCEPTION_ASSERT_LESS_OR_EQUAL(T2, 3e-12);
        }

        // time for float
        t.restart ();
        j=0;
        float jf=0;
        for (float i=1; i<1000000.f; i++) {jf++;}
        double T3 = t.elapsed()/1000000;
        T3 *= 1 + jf*1e-30;

        // time for float
        t.restart ();
        j=0;
        for (float i=1; i<1000000.f; i*=1.00001f) {j++;}
        double T4 = t.elapsed()/j;
        T4 *= 1 + j*1e-30;

        EXCEPTION_ASSERT_LESS(debug_build? 0: T*1.1, T3);
#ifdef __GCC__
        EXCEPTION_ASSERT_LESS(1.4*T, T4);
#else
        EXCEPTION_ASSERT_LESS(T*(debug_build? 1.03: 1.2), T4);
#endif
        double ghz = 1e-9/T;
        EXCEPTION_ASSERT_LESS(debug_build ? 0.1 : 0.3, ghz);
    }

    {
        Timer t;

        EXCEPTION_ASSERT_EQUALS( clamped_add(1,2), 3 );
        EXCEPTION_ASSERT_EQUALS( clamped_sub(1,2), -1 );
        EXCEPTION_ASSERT_EQUALS( clamped_add(LLONG_MIN,2ll), LLONG_MIN+2 );
        EXCEPTION_ASSERT_EQUALS( clamped_sub(LLONG_MIN,2ll), LLONG_MIN );
        EXCEPTION_ASSERT_EQUALS( clamped_sub(2ll,LLONG_MIN), LLONG_MAX );
        EXCEPTION_ASSERT_EQUALS( clamped_add(2ll,LLONG_MAX), LLONG_MAX );
        EXCEPTION_ASSERT_EQUALS( clamped_sub(2ll,LLONG_MAX), LLONG_MIN+3 );
        EXCEPTION_ASSERT_EQUALS( clamped_add(2ll,LLONG_MIN), LLONG_MIN+2 );
        EXCEPTION_ASSERT_EQUALS( clamped_sub(-2ll,LLONG_MIN), LLONG_MAX-1 );
        EXCEPTION_ASSERT_EQUALS( clamped_add(-2ll,LLONG_MAX), LLONG_MAX-2 );
    }

    {
        double T1, T2, T3;
        {
            Timer t;
            for (int i=0,j=0; i<1000000 && j<1000000;) {
                std::swap(i,++j);
            }
            T1 = t.elapsed ();
        }
        {
            Timer t;
            for (int i=0,j=0; i<1000000 && j<1000000;) {
                swap_plus(i,++j);
            }
            T2 = t.elapsed ();
        }
        {
            Timer t;
            for (int i=0,j=0; i<1000000 && j<1000000;) {
                int k=++j;
                j=i;
                i=k;
            }
            T3 = t.elapsed ();
        }
        EXCEPTION_ASSERT_LESS( T1, 25e-3 );
        EXCEPTION_ASSERT_LESS( T2, debug_build? 60e-3: 20e-3 );
        EXCEPTION_ASSERT_LESS( T3, 10e-3 );
#ifdef __GCC__
        EXCEPTION_ASSERT_LESS( 1.2*T2, T1 );
#else
        if (debug_build) {
            EXCEPTION_ASSERT_LESS( 0, T2 );
        } else {
            EXCEPTION_ASSERT_LESS ( T2, T1*1.3 );
            EXCEPTION_ASSERT_LESS ( T1, T2*2 );
        }
#endif
        if (debug_build) {
            EXCEPTION_ASSERT_LESS( 1.5*T3, T2 );
        } else {
            EXCEPTION_ASSERT_LESS ( T2, T3*2 );
            EXCEPTION_ASSERT_LESS ( T3, T2*1.8 );
        }
    }
}
