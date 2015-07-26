#include "float16.h"
#include "trace_perf.h"
#include "exceptionassert.h"
#include <cmath>

// 3 to 4 times faster than 'compress' but produces incorrect results for values that don't satisy
// min_float16() <= fabs(value) && fabs(value) <= max_float16()
// i.e denormalized numbers (value<6e-5) produce undefined float16 values.
static uint16_t compress_fast(float value)
{
    union { float f; int32_t si; } v;
    v.f = value;
    uint16_t h = ((v.si & 0x7fffffff) >> 13) - (0x38000000 >> 13);
    return h | ((v.si & 0x80000000) >> 16);
}

static float decompress_fast(uint16_t value)
{
    union { float f; int32_t si; } v;
    v.si =  ((value & 0x8000) << 16);
    v.si |= ((value & 0x7fff) << 13) + 0x38000000;
    return v.f;
}

void Float16Compressor::
        test()
{
    // It should convert float values to 16 bit representations and back
    {
        uint16_t v = 0;
        // warmup
        for (int i=0; i<10; i++)
            v += compress (i);
        for (int i=0; i<10; i++)
            v += compress_fast (i);
        {
            TRACE_PERF("correct compress");
            for (int i=0; i<100000; i++)
                v += compress (i);
        }

        {
            TRACE_PERF("fast compress");
            for (int i=0; i<100000; i++)
                v += compress_fast (i);
        }

        EXCEPTION_ASSERTX(6.103515625e-05f == min_float16(), boost::format("%.20g") % min_float16());
        EXCEPTION_ASSERTX(65504.f == max_float16(), boost::format("%.20g") % max_float16());

        for (float i=min_float16(); i<max_float16(); i*=1.01)
        {
            uint16_t c = compress (i);
            uint16_t cf = compress_fast (i);
            EXCEPTION_ASSERT_EQUALS(c,cf);

            float d = decompress(c);
            float drf = decompress(cf);
            float dfr = decompress_fast(c);
            float dff = decompress_fast(cf);
            EXCEPTION_ASSERT_EQUALS(d,drf);
            EXCEPTION_ASSERT_EQUALS(d,dfr);
            EXCEPTION_ASSERT_EQUALS(d,dff);

            EXCEPTION_ASSERT_LESS_OR_EQUAL(0, i-d);
            EXCEPTION_ASSERT_LESS_OR_EQUAL(i-d, i*1.03);
        }

        EXCEPTION_ASSERT(std::isinf(decompress(compress(max_float16()+1))));
        EXCEPTION_ASSERT_LESS(0, decompress(compress(min_float16()/100)));
        EXCEPTION_ASSERT_LESS(decompress(compress(min_float16()/100)), min_float16()/100);
    }


    // The 16 bit representation should be compatible with OpenGL
    {
        // verify by inspection
    }
}

