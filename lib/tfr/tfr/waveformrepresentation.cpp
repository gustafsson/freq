#include "waveformrepresentation.h"
#include "neat_math.h"

namespace Tfr {

TransformDesc::ptr WaveformRepresentationDesc::
        copy() const
{
    return TransformDesc::ptr(new WaveformRepresentationDesc);
}


pTransform WaveformRepresentationDesc::
        createTransform() const
{
    return pTransform(new WaveformRepresentation);
}


float WaveformRepresentationDesc::
        displayedTimeResolution( float FS, float /*hz*/ ) const
{
    return .025f / FS;
}



FreqAxis WaveformRepresentationDesc::
        freqAxis( float ) const
{
    FreqAxis a;
    a.setWaveform (-1, 1, 1<<16);

    return a;
}


unsigned WaveformRepresentationDesc::
        next_good_size( unsigned current_valid_samples_per_chunk, float sample_rate ) const
{
    return DummyTransformDesc::next_good_size (current_valid_samples_per_chunk, sample_rate);
}


unsigned WaveformRepresentationDesc::
        prev_good_size( unsigned current_valid_samples_per_chunk, float sample_rate ) const
{
    return DummyTransformDesc::prev_good_size (current_valid_samples_per_chunk, sample_rate);
}


Signal::Interval WaveformRepresentationDesc::
        requiredInterval( const Signal::Interval& I, Signal::Interval* expectedOutput ) const
{
    // if a sample 'i' is valid it means that the line between i-1 and i is valid
    Signal::Interval J(clamped_sub(I.first, Signal::IntervalType(1)), I.last);

    if (expectedOutput)
        *expectedOutput = J;

    return J;
}


Signal::Interval WaveformRepresentationDesc::
        affectedInterval( const Signal::Interval& I ) const
{
    // if a sample 'i' is valid it means that the line between i-1 and i is valid
    return Signal::Interval(I.first, clamped_add(I.last, Signal::IntervalType(1)));
}


const TransformDesc* WaveformRepresentation::
        transformDesc() const
{
    return &desc;
}

} // namespace Tfr
