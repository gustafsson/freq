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

        virtual Signal::pBuffer process(Signal::pBuffer b);
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
    virtual Signal::Interval requiredInterval( const Signal::Interval& I, Signal::Interval* expectedOutput ) const;
    virtual Interval affectedInterval( const Interval& I ) const;
    virtual OperationDesc::Ptr copy() const;
    virtual Operation::Ptr createOperation(ComputingEngine* engine) const;
    virtual bool operator==(const OperationDesc& d) const;

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
