#ifndef SIGNALBUFFERSOURCE_H
#define SIGNALBUFFERSOURCE_H

#include "signal/operation.h"

namespace Signal
{

class BufferSource: public FinalSource
{
public:
    BufferSource( pBuffer _waveform = pBuffer() );

    void setBuffer( pBuffer _waveform );

    virtual pBuffer read( const Interval& I );
    virtual float sample_rate();
    virtual long unsigned number_of_samples();

    virtual unsigned num_channels();
    virtual void set_channel(unsigned c);
    virtual unsigned get_channel() { return channel; }

protected:
	unsigned channel;
    std::vector<pBuffer> _waveforms;
};

} // namespace Signal

#endif // SIGNALBUFFERSOURCE_H
