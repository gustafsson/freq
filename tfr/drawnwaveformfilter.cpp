#include "drawnwaveformfilter.h"

#include "drawnwaveform.h"
#include "drawnwaveform.cu.h"

// GPUMISC
#include "neat_math.h"

namespace Tfr {


DrawnWaveformFilter::
        DrawnWaveformFilter(Signal::pOperation source, pTransform t)
:   Filter(source)
{
//    if (!t)
//        t = DrawnWaveform::SingletonP();

    if (t)
    {
        BOOST_ASSERT( dynamic_cast<DrawnWaveform*>(t.get()));

        _transform = t;
    }
}


Signal::Interval DrawnWaveformFilter::
        requiredInterval( const Signal::Interval& I )
{
    DrawnWaveform* w = dynamic_cast<DrawnWaveform*>(transform().get());
    if (0 == w)
        throw std::invalid_argument("'transform' must be an instance of Tfr::DrawnWaveform");

    unsigned blobsize = std::max(1.f, w->blob( this->sample_rate() ));
    w->signal_length = this->number_of_samples();

    Signal::Interval J = Signal::Intervals(I).enlarge(1).coveredInterval();
    J.last = J.first + align_up( J.count(), blobsize*drawWaveform_BLOCK_SIZE );

    return J;
}


ChunkAndInverse DrawnWaveformFilter::
        computeChunk( const Signal::Interval& I )
{
    ChunkAndInverse ci;

    Signal::Interval J = requiredInterval( I );
    ci.inverse = source()->readFixedLength( J );

    float *p = ci.inverse->waveform_data()->getCpuMemory();
    float maxValue=0;
    for(unsigned i=0; i<ci.inverse->number_of_samples(); ++i)
        maxValue = std::max(std::abs(p[i]), maxValue);
    maxValue *= 1.1;

    DrawnWaveform* w = dynamic_cast<DrawnWaveform*>(transform().get());
    if (0 == w)
        throw std::invalid_argument("'transform' must be an instance of Tfr::DrawnWaveform");
    if (maxValue > w->maxValue)
    {
        invalidate_samples(Signal::Intervals::Intervals_ALL);
        w->maxValue = maxValue;
    }

    // Compute the continous wavelet transform
    ci.chunk = (*transform())( ci.inverse );

#ifdef _DEBUG
    Signal::Interval cii = ci.chunk->getInterval();

    BOOST_ASSERT( cii & I );
#endif

    return ci;
}


pTransform DrawnWaveformFilter::
        transform() const
{
    return _transform ? _transform : DrawnWaveform::SingletonP();
}


void DrawnWaveformFilter::
        transform( pTransform t )
{
    if (0 == dynamic_cast<DrawnWaveform*>(t.get ()))
        throw std::invalid_argument("'transform' must be an instance of Tfr::DrawnWaveform");

    if ( t == transform() && !_transform )
        t.reset();

    if (_transform == t )
        return;

    invalidate_samples( Signal::Interval(0, number_of_samples() ));

    _transform = t;
}

} // namespace Tfr
