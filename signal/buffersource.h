#ifndef SIGNALBUFFERSOURCE_H
#define SIGNALBUFFERSOURCE_H

#include "signal/source.h"

namespace Signal
{

class BufferSource: public Source
{
public:
    BufferSource( pBuffer _waveform = pBuffer() );

    virtual pBuffer read( const Interval& I );
    virtual unsigned sample_rate() { return _waveform->sample_rate; }
    virtual long unsigned number_of_samples() { return _waveform->number_of_samples(); }

    pBuffer getChunkBehind() { return _waveform; }
    void setChunk( pBuffer chunk ) { _waveform = chunk; }

    unsigned read_channel;

private:
    pBuffer getChunk( unsigned firstSample, unsigned numberOfSamples, unsigned channel, Buffer::Interleaved interleaved );
    pSource crop();

    unsigned channel_count() {        return _waveform->waveform_data->getNumberOfElements().height; }

private:

    pBuffer _waveform;
};

} // namespace Signal

#endif // SIGNALBUFFERSOURCE_H
