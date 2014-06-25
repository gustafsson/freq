#ifndef TFR_WAVEFORMREPRESENTATION_H
#define TFR_WAVEFORMREPRESENTATION_H

#include "dummytransform.h"

namespace Tfr {

class WaveformRepresentationDesc: public DummyTransformDesc
{
public:
    TransformDesc::ptr copy() const;
    pTransform createTransform() const;
    float displayedTimeResolution( float FS, float hz ) const;
    FreqAxis freqAxis( float FS ) const;
    unsigned next_good_size( unsigned current_valid_samples_per_chunk, float sample_rate ) const;
    unsigned prev_good_size( unsigned current_valid_samples_per_chunk, float sample_rate ) const;
    Signal::Interval requiredInterval( const Signal::Interval& I, Signal::Interval* expectedOutput ) const;
    Signal::Interval affectedInterval( const Signal::Interval& I ) const;
};

class WaveformRepresentation: public DummyTransform
{
public:
    const TransformDesc* transformDesc() const;

private:
    WaveformRepresentationDesc desc;
};

} // namespace Tfr

#endif // TFR_WAVEFORMREPRESENTATION_H
