#include "printbuffer.h"

#include <sstream>

// gpumisc
#include "Statistics.h"

using namespace boost;

namespace Test {

std::string PrintBuffer::
        printBuffer(Signal::pBuffer b)
{
    std::stringstream ss;
    ss << b->getInterval () << ", " << b->number_of_channels () << " channels" << std::endl;

    for (unsigned c=0; c<b->number_of_channels(); ++c) {
        ss << "[" << c << "] = ";
        float *p = b->getChannel (c)->waveform_data()->getCpuMemory ();

        if (b->number_of_samples ())
            ss << "{ " << p[0];

        for (unsigned j=1; j<b->number_of_samples (); ++j)
            ss << ", " << p[j];

        ss << " }" << std::endl;
    }

    return ss.str();
}


std::string PrintBuffer::
        printBufferStats(Signal::pBuffer b)
{
    std::stringstream ss;
    ss << b->getInterval () << ", " << b->number_of_channels () << " channels" << std::endl;

    for (unsigned c=0; c<b->number_of_channels(); ++c) {
        Statistics<float> s(b->getChannel (c)->waveform_data(), false, true);
        ss << "[" << c << "]: max = " << *s.getMax () << ", min = " << *s.getMin ()
           << ", mean = " << s.getMean () << ", std = " << s.getStd () << std::endl;
    }

    return ss.str();
}

} // namespace Test

#include "randombuffer.h"

namespace Test {

void PrintBuffer::
        test()
{
    // It should print buffer contents for debugging purposes.
    {
        Signal::pBuffer b = RandomBuffer::smallBuffer();
        std::string s = PrintBuffer::printBuffer(b);
        std::string expected = "[4, 9)5#, 2 channels\n[0] = { -3, 7, -8, 5, 3 }\n[1] = { 1, -9, -3, 6, -2 }\n";
        EXCEPTION_ASSERT_EQUALS(s, expected);

        std::string stats = PrintBuffer::printBufferStats(b);
        expected = "[4, 9)5#, 2 channels\n[0]: max = 7, min = -8, mean = 0.8, std = 5.52811\n[1]: max = 6, min = -9, mean = -1.4, std = 4.92341\n";
        EXCEPTION_ASSERT_EQUALS(stats, expected);
    }
}

} // namespace Test
