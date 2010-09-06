#include "filter.h"

namespace Tfr {

//////////// Filter
Filter::
        Filter()
            :
            Operation( pSource() ),
            enabled(true)
{}

Filter::
        Filter( pSource source )
            :
            Operation( source ),
            enabled(true)
{}


Signal::pBuffer Filter::
        read(  const Signal::Interval& I )
{
    unsigned firstSample = I.first;
    unsigned numberOfSamples = I.count;

    // If we're not asked to compute a chunk, try to take shortcuts
    if (!_save_previous_chunk)
    {
        Intervals work(first_valid_sample, first_valid_sample + numberOfSamples);

        // If filter would make all these samples zero, make them zero and return immediately
        if ((work - _filter->ZeroedSamples( _source->sample_rate() )).isEmpty())
        {
            // Doesn't have to read from source, just create a buffer with all samples set to 0
            pBuffer b( new Buffer( first_valid_sample, numberOfSamples, _source->sample_rate() ));

            ::memset( b->waveform_data->getCpuMemory(), 0, b->waveform_data->getSizeInBytes1D());

            TIME_CwtFilter Intervals(b->getInterval()).print("CwtFilter silent");
            return b;
        }

        // If filter would leave all these samples unchanged, return immediately
        if ((work & _filter->NeededSamples()).isEmpty())
        {
            // Attempt a regular simple read
            pBuffer b = _source->read(first_valid_sample, numberOfSamples);
            work = b->getInterval();
            work -= _filter->NeededSamples().inverse();

            if (work.isEmpty()) {
                TIME_CwtFilter Intervals(b->getInterval()).print("CwtFilter unaffected");
                return b;
            }

            // If _source returned some parts that we didn't ask for,
            // try again and explicitly take out those samples

            TIME_CwtFilter Intervals(b->getInterval()).print("FilterOp fixed unaffected");
            // Failed, return the exact samples validated as untouched
            return _source->readFixedLength(first_valid_sample, numberOfSamples);
        }
    }

    // If we've reached this far, the transform will have to be computed
    pChunk c = readChunk(firstSample, numberOfSamples);

    Signal::pBuffer r = transform()->inverse( c );

    _invalid_samples -= r->getInterval();
    return r;
}

} // namespace Tfr
