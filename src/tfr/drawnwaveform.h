#ifndef DRAWNWAVEFORM_H
#define DRAWNWAVEFORM_H

#include "transform.h"
#include "chunk.h"

namespace Tfr {

class DrawnWaveform : public Transform, public TransformParams
{
public:
    DrawnWaveform();

    virtual const TransformParams* transformParams() const { return this; }

    virtual pChunk operator()( Signal::pBuffer b );

    virtual Signal::pBuffer inverse( pChunk chunk );


    virtual pTransform createTransform() const;

    virtual float displayedTimeResolution( float FS, float hz ) const;

    virtual FreqAxis freqAxis( float FS ) const;

    virtual unsigned next_good_size( unsigned current_valid_samples_per_chunk, float sample_rate ) const;

    virtual unsigned prev_good_size( unsigned current_valid_samples_per_chunk, float sample_rate ) const;

    virtual std::string toString() const;

    virtual bool operator==(const TransformParams& b) const;


    float blob(float FS);

    float block_fs;
    unsigned signal_length;
    float maxValue;
};


class DrawnWaveformChunk: public Chunk
{
public:
    DrawnWaveformChunk(float block_fs)
        :
        Chunk(Chunk::Order_row_major),
        block_fs(block_fs)
    {}


    float block_fs;
};

} // namespace Tfr

#endif // DRAWNWAVEFORM_H
