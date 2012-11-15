#include "drawnwaveformfilter.h"

#include "drawnwaveform.h"
#include "drawnwaveformkernel.h"

// gpumisc
#include "neat_math.h"

namespace Tfr {


DrawnWaveformFilter::
        DrawnWaveformFilter(Signal::pOperation source, pTransform t)
:   Filter(source),
    max_value_(0)
{
    if (!t)
        t = pTransform(new DrawnWaveform());

    DrawnWaveform* c = dynamic_cast<DrawnWaveform*>(t.get());
    EXCEPTION_ASSERT( c );

    max_value_ = c->maxValue;

    transform( t );
}


Signal::Interval DrawnWaveformFilter::
        requiredInterval( const Signal::Interval& I, Tfr::pTransform t )
{
    DrawnWaveform* w = dynamic_cast<DrawnWaveform*>(t.get ());
    if (0 == w)
        throw std::invalid_argument("'transform' must be an instance of Tfr::DrawnWaveform");

    unsigned blobsize = std::max(1.f, w->blob( this->sample_rate() ));
    w->signal_length = this->number_of_samples();

    Signal::Interval J = Signal::Intervals(I).enlarge(blobsize).spannedInterval();
    J.last = J.first + align_up( J.count(), (Signal::IntervalType) blobsize*drawWaveform_BLOCK_SIZE );

    return J;
}


bool DrawnWaveformFilter::
        applyFilter( ChunkAndInverse &chunk )
{
    DrawnWaveform* w = dynamic_cast<DrawnWaveform*>(chunk.t.get ());
    if (w->maxValue != max_value_)
    {
        max_value_ = w->maxValue;
        invalidate_samples(Signal::Intervals::Intervals_ALL);
    }

    Filter::applyFilter (chunk);
    return false;
}


} // namespace Tfr
