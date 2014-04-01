#ifndef DRAWNWAVEFORM_H
#define DRAWNWAVEFORM_H

#include "transform.h"
#include "chunk.h"

namespace Tfr {

class DrawnWaveform : public Transform, public TransformDesc
{
public:
    DrawnWaveform();

    virtual const TransformDesc* transformDesc() const { return this; }

    virtual pChunk operator()( Signal::pMonoBuffer b );

    virtual Signal::pMonoBuffer inverse( pChunk chunk );



    virtual TransformDesc::ptr copy() const;

    virtual pTransform createTransform() const;

    virtual float displayedTimeResolution( float FS, float hz ) const;

    virtual FreqAxis freqAxis( float FS ) const;

    virtual unsigned next_good_size( unsigned current_valid_samples_per_chunk, float sample_rate ) const;

    virtual unsigned prev_good_size( unsigned current_valid_samples_per_chunk, float sample_rate ) const;

    virtual Signal::Interval requiredInterval( const Signal::Interval& I, Signal::Interval* expectedOutput ) const;

    virtual Signal::Interval affectedInterval( const Signal::Interval& I ) const;

    virtual std::string toString() const;

    virtual bool operator==(const TransformDesc& b) const;


    float blob(float FS);

    float block_fs;
    unsigned signal_length;
    float maxValue;

private:
    void updateMaxValue(Signal::pMonoBuffer b);
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
