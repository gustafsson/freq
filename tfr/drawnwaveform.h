#ifndef DRAWNWAVEFORM_H
#define DRAWNWAVEFORM_H

#include "transform.h"
#include "HasSingleton.h"

namespace Tfr {

class DrawnWaveform : public Transform, public HasSingleton<DrawnWaveform,Transform>
{
public:
    DrawnWaveform();

    virtual pChunk operator()( Signal::pBuffer b );

    virtual Signal::pBuffer inverse( pChunk chunk );

    virtual float displayedTimeResolution( float FS, float hz );

    virtual FreqAxis freqAxis( float FS );

    unsigned blob(float FS);

    float block_fs;
    unsigned signal_length;
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
