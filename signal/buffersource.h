#ifndef SIGNALBUFFERSOURCE_H
#define SIGNALBUFFERSOURCE_H

#include "signal/operation.h"

namespace Signal
{

class BufferSource: public FinalSource
{
public:
    BufferSource( pBuffer _waveform = pBuffer() );

    virtual pBuffer read( const Interval& ) { return _waveform; }
    virtual unsigned sample_rate() { return _waveform->sample_rate; }
    virtual long unsigned number_of_samples() { return _waveform->number_of_samples(); }

private:
/*    pBuffer getChunk( unsigned firstSample, unsigned numberOfSamples, unsigned channel, Buffer::Interleaved interleaved );
    pOperation crop();

    unsigned channel_count() {        return _waveform->waveform_data->getNumberOfElements().height; }
*/
protected:
    pBuffer _waveform;

};

} // namespace Signal

#endif // SIGNALBUFFERSOURCE_H
