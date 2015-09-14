#ifndef TFR_WAVEFORMREPRESENTATION_H
#define TFR_WAVEFORMREPRESENTATION_H

#include "dummytransform.h"

namespace Tfr {

/**
 * @brief The WaveformRepresentationDesc class is used with
 * Heightmap::TfrMappings::WaveformBlockFilterDesc
 */
class WaveformRepresentationDesc: public DummyTransformDesc
{
public:
    TransformDesc::ptr copy() const override;
    pTransform createTransform() const override;
    float displayedTimeResolution( float FS, float hz ) const override;
    FreqAxis freqAxis( float FS ) const override;
    unsigned next_good_size( unsigned current_valid_samples_per_chunk, float sample_rate ) const override;
    unsigned prev_good_size( unsigned current_valid_samples_per_chunk, float sample_rate ) const override;
    Signal::Interval requiredInterval( const Signal::Interval& I, Signal::Interval* expectedOutput ) const override;
    Signal::Interval affectedInterval( const Signal::Interval& I ) const override;
};

class WaveformRepresentation: public DummyTransform
{
public:
    const TransformDesc* transformDesc() const override;
    pChunk operator()( Signal::pMonoBuffer b ) override;
    Signal::pMonoBuffer inverse( pChunk chunk ) override;

private:
    WaveformRepresentationDesc desc;
};

} // namespace Tfr

#endif // TFR_WAVEFORMREPRESENTATION_H
