#ifndef SIGNALBUFFERSOURCE_H
#define SIGNALBUFFERSOURCE_H

#include "operation.h"
#include <vector>

namespace Signal
{

class SaweDll BufferSource: public FinalSource
{
public:
    BufferSource( pBuffer waveform = pBuffer() );
    BufferSource( pMonoBuffer waveform );

    void setBuffer( pBuffer waveform );

    virtual pBuffer read( const Interval& I );
    virtual float sample_rate();
    void set_sample_rate( float fs );
    virtual IntervalType number_of_samples();

    virtual unsigned num_channels();

private:
    pBuffer buffer_;
};

} // namespace Signal

#endif // SIGNALBUFFERSOURCE_H
