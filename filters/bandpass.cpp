#include "bandpass.h"
#include <sstream>

using namespace std;

namespace Filters {

Bandpass::
        Bandpass(float f1, float f2, bool save_inside)
            :
            _f1(f1),
            _f2(f2),
            _save_inside(save_inside)
{

}


std::string Bandpass::
        name()
{
    stringstream ss;
    ss << "Bandpass [" << min(_f1, _f2) << ", " << max(_f1, _f2) << "] hz";
    return ss.str();
}


void Bandpass::
        operator()( Tfr::Chunk& c )
{
    float minf = min(_f1, _f2);
    float maxf = max(_f1, _f2);
    float a = c.freqAxis.getFrequencyScalar( minf );
    float b = c.freqAxis.getFrequencyScalar( maxf );
    float2 *p = c.transform_data->getCpuMemory();
    float2 zero = make_float2(0, 0);

    // assuming STFT is not redundant (meaning that R2C and C2R transforms are being used)
    unsigned window = (unsigned)(c.freqAxis.max_frequency_scalar*2 + 0.5f);
    unsigned actualSize = window/2 + 1;
    unsigned windows = c.transform_data->getNumberOfElements().width / actualSize;

    if (_save_inside)
    {
        for (unsigned t=0; t<windows; ++t)
        {
            for (unsigned s=0; s < max(0.f, a) && s<window/2 && s<actualSize; ++s)
                p[t*actualSize + s] = zero;

            for (unsigned s=b+1; s<window-1-b && s<actualSize; ++s)
                p[t*actualSize + s] = zero;

            for (unsigned s=window-1; s > max(window - 1.f, window - 1 - a) && s>=window/2 && s<actualSize; --s)
                p[t*actualSize + s] = zero;
        }
    }
    else
    {
        for (unsigned t=0; t<windows; ++t)
        {
            for (unsigned s=max(0.f, a); s<=b && s<window/2 && s<actualSize; ++s)
                p[t*actualSize + s] = zero;

            for (unsigned s=max(window - 1.f, window - 1 - a); s>=window - 1 - b && s>=window/2 && s<actualSize; --s)
                p[t*actualSize + s] = zero;
        }
    }
}

} // namespace Filters
