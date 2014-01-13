#include "bandpass.h"
#include <sstream>

#include "tfr/chunk.h"
#include "tfr/stft.h"
#include "signal/computingengine.h"

#define TIME_BANDPASS
//#define TIME_BANDPASS if(0)

using namespace std;

namespace Filters {

BandpassKernel::
        BandpassKernel(float f1, float f2, bool save_inside)
            :
            _f1(f1),
            _f2(f2),
            _save_inside(save_inside)
{

}


std::string BandpassKernel::
        name()
{
    stringstream ss;
    ss << "Bandpass [" << min(_f1, _f2) << ", " << max(_f1, _f2) << "] hz";
    return ss.str();
}


void BandpassKernel::
        operator()( Tfr::ChunkAndInverse& chunk )
{
    Tfr::Chunk& c = *chunk.chunk.get ();
    TIME_BANDPASS TaskTimer tt("%s (save %sside) on %s",
                               name().c_str(),
                               _save_inside?"in":"out",
                               c.getInterval().toString().c_str());

    float minf = min(_f1, _f2);
    float maxf = max(_f1, _f2);
    float a = c.freqAxis.getFrequencyScalar( minf );
    float b = c.freqAxis.getFrequencyScalar( maxf );
    std::complex<float> *p = c.transform_data->getCpuMemory();
    std::complex<float> zero( 0, 0 );

    // assuming STFT is not redundant (meaning that R2C and C2R transforms are being used)
    unsigned window = (unsigned)(c.freqAxis.max_frequency_scalar*2 + 0.5f);
    unsigned actualSize = window/2 + 1;
    unsigned windows = c.transform_data->size().width / actualSize;

    EXCEPTION_ASSERT( c.nScales() == actualSize );
    EXCEPTION_ASSERT( c.nSamples() == windows );
    EXCEPTION_ASSERT( ((Tfr::StftChunk&)c).window_size() == window );

    TIME_BANDPASS TaskInfo("window = %u, actualSize = %u, windows = %u, a=%g, b=%g",
                               window, actualSize, windows, a, b);

    if (_save_inside)
    {
        for (unsigned t=0; t<windows; ++t)
        {
            for (unsigned s=0; s < max(0.f, a) && s<window/2 && s<actualSize; ++s)
                p[t*actualSize + s] = zero;

            for (unsigned s=b+1; s<window-1-b && s<actualSize; ++s)
                p[t*actualSize + s] = zero;

            for (unsigned s=min(actualSize-1, window-1); s > max(window - 1.f, window - 1 - a) && s>=window/2; --s)
                p[t*actualSize + s] = zero;
        }
    }
    else
    {
        for (unsigned t=0; t<windows; ++t)
        {
            for (unsigned s=max(0.f, a); s<=b && s<window/2 && s<actualSize; ++s)
                p[t*actualSize + s] = zero;

            for (unsigned s=min(actualSize-1, max(window - 1, (unsigned)(window - 1 - a))); s>=window - 1 - b && s>=window/2; --s)
                p[t*actualSize + s] = zero;
        }
    }
}


Bandpass::
        Bandpass(float f1, float f2, bool save_inside)
    :
      _f1(f1),
      _f2(f2),
      _save_inside(save_inside)
{
}


Tfr::pChunkFilter Bandpass::
        createChunkFilter(Signal::ComputingEngine* engine) const
{
    if (engine==0 || dynamic_cast<Signal::ComputingCpu*>(engine))
        return Tfr::pChunkFilter(new BandpassKernel(_f1, _f2, _save_inside));
    return Tfr::pChunkFilter();
}


Tfr::ChunkFilterDesc::Ptr Bandpass::
        copy() const
{
    return ChunkFilterDesc::Ptr(new Bandpass(_f1, _f2, _save_inside));
}


void Bandpass::
        test()
{
    // It should apply a bandpass filter between f1 and f2 to a signal.
    EXCEPTION_ASSERT(false);
}



} // namespace Filters
