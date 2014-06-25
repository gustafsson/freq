#include "waveformrepresentation.h"

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
    Signal::Intervals J = I;
    J = J.enlarge (1);

    if (expectedOutput)
        *expectedOutput = J.spannedInterval ();

    return J.spannedInterval ();
}


Signal::Interval WaveformRepresentationDesc::
        affectedInterval( const Signal::Interval& I ) const
{
    return requiredInterval (I,0);
}


const TransformDesc* WaveformRepresentation::
        transformDesc() const
{
    return &desc;
}

} // namespace Tfr
