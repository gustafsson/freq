#ifndef DRAWNWAVEFORM_H
#define DRAWNWAVEFORM_H

#include "transform.h"
#include "HasSingleton.h"

namespace Tfr {

//    TODO remove HasSingleton
class DrawnWaveform : public Transform
{
public:
    DrawnWaveform();

    virtual pChunk operator()( Signal::pBuffer b );

    virtual Signal::pBuffer inverse( pChunk chunk );

    virtual float displayedTimeResolution( float FS, float hz );

    virtual FreqAxis freqAxis( float FS );

    virtual unsigned next_good_size( unsigned current_valid_samples_per_chunk, float sample_rate );

    virtual unsigned prev_good_size( unsigned current_valid_samples_per_chunk, float sample_rate );

    virtual std::string toString();

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
