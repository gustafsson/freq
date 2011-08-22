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

    unsigned blobsize = w->blob( this->sample_rate() );
    w->signal_length = this->number_of_samples();

    Signal::Interval J = I;
            /*Signal::Intervals(I)
                .enlarge(
                        ceilf(
                                blobsize * drawWaveform_BLOCK_SIZE
                             )
                        )
                .coveredInterval();*/

    J.last = J.first + int_div_ceil( J.count(), blobsize*drawWaveform_BLOCK_SIZE)*blobsize*drawWaveform_BLOCK_SIZE;
    //J.last = J.first + spo2g(J.count() - 1);

    return J;
}


ChunkAndInverse DrawnWaveformFilter::
        computeChunk( const Signal::Interval& I )
{
    ChunkAndInverse ci;

    ci.inverse = source()->readFixedLength( requiredInterval( I ) );

    // Compute the continous wavelet transform
    ci.chunk = (*transform())( ci.inverse );

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
