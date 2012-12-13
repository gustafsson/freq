#ifndef SIGNALBUFFERSOURCE_H
#define SIGNALBUFFERSOURCE_H

#include "operation.h"
#include <vector>

namespace Signal
{

class SaweDll BufferSource: public FinalSource, public Operation
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



    virtual Signal::Interval requiredInterval( const Signal::Interval& I ) const {
        return I;
    }
    virtual Signal::pBuffer process(Signal::pBuffer b) {
        Signal::pBuffer r(new Signal::Buffer(b->getInterval (), buffer_->sample_rate (), buffer_->number_of_channels ()));
        *r |= *buffer_;
        return r;
    }

private:
    pBuffer buffer_;
};

} // namespace Signal

#endif // SIGNALBUFFERSOURCE_H
