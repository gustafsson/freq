#ifndef TEST_RANDOMBUFFER_H
#define TEST_RANDOMBUFFER_H

#include "signal/buffer.h"

namespace Test {

/**
 * @brief The RandomBuffer class should produce signal buffers with pseudo-random content.
 */
class RandomBuffer
{
public:
    static Signal::pBuffer randomBuffer(Signal::Interval I, float sample_rate, int number_of_channels, int seed=0);

    static Signal::pBuffer smallBuffer(int seed=0);

public:
    static void test();
};

} // namespace Test

#endif // TEST_RANDOMBUFFER_H
