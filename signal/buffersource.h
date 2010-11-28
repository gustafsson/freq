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

    unsigned num_channels() { return _waveforms.size(); }
    unsigned channel;

protected:
    std::vector<pBuffer> _waveforms;
};

} // namespace Signal

#endif // SIGNALBUFFERSOURCE_H
