#ifndef SIGNALBUFFERSOURCE_H
#define SIGNALBUFFERSOURCE_H

#include "operation.h"
#include "source.h"

namespace Signal
{

class SignalDll BufferSource: public SourceBase, public OperationDesc
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

    // SourceBase
    pBuffer read( const Interval& I );
    float sample_rate();
    IntervalType number_of_samples();
    unsigned num_channels();

    // OperationDesc
    Signal::Interval requiredInterval( const Signal::Interval& I, Signal::Interval* expectedOutput ) const;
    Interval affectedInterval( const Interval& I ) const;
    OperationDesc::ptr copy() const;
    Operation::ptr createOperation(ComputingEngine* engine) const;
    Extent extent() const;
    bool operator==(const OperationDesc& d) const;

    void setBuffer( pBuffer waveform );
    void setSampleRate( float fs );

private:
    pBuffer buffer_;

public:
    static void test();
};

} // namespace Signal

#endif // SIGNALBUFFERSOURCE_H
