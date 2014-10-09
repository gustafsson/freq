#include "waveformrepresentation.h"
#include "neat_math.h"
#include "signal/buffer.h"
#include "tfr/chunk.h"
#include "log.h"

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
    return .0025f / FS;
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
        *expectedOutput = I;

    return J;
}


Signal::Interval WaveformRepresentationDesc::
        affectedInterval( const Signal::Interval& I ) const
{
    // see requiredInterval
    return Signal::Interval(I.first, clamped_add(I.last, Signal::IntervalType(1)));
}


const TransformDesc* WaveformRepresentation::
        transformDesc() const
{
    return &desc;
}


pChunk WaveformRepresentation::
        operator()( Signal::pMonoBuffer b )
{
    pChunk c = DummyTransform::operator() (b);
    c->first_valid_sample++;
    c->n_valid_samples--;
    return c;
}

Signal::pMonoBuffer WaveformRepresentation::
        inverse( pChunk chunk )
{
    EXCEPTION_ASSERTX(false, "waveformrepresentation: apparently missing ChunkFilter::NoInverseTag");
}

} // namespace Tfr
