#ifndef SIGNALBUFFERSOURCE_H
#define SIGNALBUFFERSOURCE_H

#include "signal/operation.h"

namespace Signal
{

class BufferSource: public FinalSource
{
public:
    BufferSource( pBuffer _waveform = pBuffer() );

    virtual pBuffer read( const Interval& I );
    virtual unsigned sample_rate();
    virtual long unsigned number_of_samples();

protected:
    pBuffer _waveform;

};

} // namespace Signal

#endif // SIGNALBUFFERSOURCE_H
