#ifndef SIGNALBUFFERSOURCE_H
#define SIGNALBUFFERSOURCE_H

#include "operation.h"
#include <vector>

namespace Signal
{

class SaweDll BufferSource: public FinalSource, public OperationSourceDesc
{
public:
    class BufferSourceOperation: public Operation {
    public:
        BufferSourceOperation(pBuffer buffer): buffer_(buffer) {}

        virtual Signal::Interval requiredInterval( Signal::Interval& I ) {
            return I;
        }

        virtual Signal::pBuffer process(Signal::pBuffer b) {
            Signal::pBuffer r(new Signal::Buffer(b->getInterval (), buffer_->sample_rate (), buffer_->number_of_channels ()));
            // Won't actually copy but instead create a reference and use CopyOnWrite
            *r |= *buffer_;
            return r;
        }

    private:
        pBuffer buffer_;
    };

    BufferSource( pBuffer waveform = pBuffer() );
    BufferSource( pMonoBuffer waveform );

    void setBuffer( pBuffer waveform );

    // DeprecatedOperation
    virtual pBuffer read( const Interval& I );
    virtual float sample_rate();
    void set_sample_rate( float fs );
    virtual IntervalType number_of_samples();
    virtual unsigned num_channels();


    // OperationDesc
    virtual OperationDesc::Ptr copy() const;
    virtual Operation::Ptr createOperation(ComputingEngine* engine) const;

    // OperationSourceDesc
    virtual float getSampleRate() const;
    virtual float getNumberOfChannels() const;
    virtual float getNumberOfSamples() const;

    static void test();

private:
    pBuffer buffer_;
};

} // namespace Signal

#endif // SIGNALBUFFERSOURCE_H
