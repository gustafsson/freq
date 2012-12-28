#include "freqaxis.h"
#include "exceptionassert.h"

#include <vector>
using namespace std;

namespace Tfr {

void FreqAxis::
        test ()
{
    Tfr::FreqAxis fa;
    fa.setLinear(1);
    fa.max_frequency_scalar = 1;
    fa.min_hz = -20;
    fa.f_step = -2*fa.min_hz;

    float correct1[][2] =
    {
        {0.0, -20},
        {0.1, -16},
        {0.2, -12},
        {0.3, -8},
        {0.4, -4},
        {0.5, 0},
        {0.6, 4},
        {0.7, 8},
        {0.8, 12},
        {0.9, 16}
    };

    float correct2[][2] =
    {
        {0, -20},
        {1, -17.8947},
        {2, -15.7895},
        {3, -13.6842},
        {4, -11.5789},
        {5, -9.47368},
        {6, -7.36842},
        {7, -5.26316},
        {8, -3.15789},
        {9, -1.05263},
        {10, 1.05263},
        {11, 3.1579},
        {12, 5.26316},
        {13, 7.36842},
        {14, 9.47369},
        {15, 11.5789},
        {16, 13.6842},
        {17, 15.7895},
        {18, 17.8947},
        {19, 20}
    };

    int i=0;
    for (float f=0; f<=fa.max_frequency_scalar; f+=0.1)
    {
        EXCEPTION_ASSERT_FUZZYEQUALS(f, correct1[i][0], 1e-6);
        EXCEPTION_ASSERT_FUZZYEQUALS(fa.getFrequency(f), correct1[i][1], 1e-5);
        i++;
    }

    float maxValue = 20;
    fa.max_frequency_scalar = 20 - 1;
    fa.min_hz = -maxValue;
    fa.f_step = (1/fa.max_frequency_scalar) * 2*maxValue;

    for (int f=0; f<=fa.max_frequency_scalar; f++)
    {
        EXCEPTION_ASSERT_FUZZYEQUALS(f, correct2[f][0], 0);
        EXCEPTION_ASSERT_FUZZYEQUALS(fa.getFrequency((float)f), correct2[f][1], 1e-4);
    }
}

} // namespace Tfr
