#include "randombuffer.h"

#include <random>

using namespace std;
using namespace Signal;

namespace Test {

pBuffer RandomBuffer::
        randomBuffer(Interval I, float sample_rate, int number_of_channels, int seed)
{
    pBuffer b(new Buffer(I, sample_rate, number_of_channels));

    uniform_int_distribution<int> unif(-9, 9);
    default_random_engine re(seed+1024);

    for (int c=0; c<number_of_channels; c++) {
        float*p = b->getChannel (c)->waveform_data ()->getCpuMemory ();
        for (unsigned i=0; i<I.count (); i++) {
            p[i] = unif(re);
        }
    }

    return b;
}


Signal::pBuffer RandomBuffer::
        smallBuffer(int seed)
{
    return randomBuffer(Interval(4,9), 5, 2, seed);
}


void RandomBuffer::
        test()
{
    // It should produce signal buffers with pseudo-random content.
    {
        pBuffer b = randomBuffer(Interval(4,9), 5, 2);
        float b1[] = {-3, 7, -8, 5, 3};
        float b2[] = {1, -9, -3, 6, -2};
        float b3[] = {1, -9, -3, 6, -1};

        EXCEPTION_ASSERT(0 == memcmp(b1, b->getChannel (0)->waveform_data ()->getCpuMemory (), sizeof(b1)));
        EXCEPTION_ASSERT(0 == memcmp(b2, b->getChannel (1)->waveform_data ()->getCpuMemory (), sizeof(b2)));
        EXCEPTION_ASSERT(0 != memcmp(b3, b->getChannel (1)->waveform_data ()->getCpuMemory (), sizeof(b3)));

        pBuffer c = smallBuffer();
        EXCEPTION_ASSERT(*b == *c);

        pBuffer d = smallBuffer(1);
        EXCEPTION_ASSERT(*b != *d);
    }
}

} // namespace Test
