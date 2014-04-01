#ifndef COMPUTERMS_H
#define COMPUTERMS_H

#include "signal/operation.h"

namespace Tools {
namespace Support {

class RmsValue
{
public:
    Signal::Intervals rms_I;
    double rms = 0;
};


class ComputeRms : public Signal::Operation
{
public:
    ComputeRms(shared_state<RmsValue> rms);

    Signal::pBuffer process( Signal::pBuffer b);

private:
    shared_state<RmsValue> rms;
};


/**
 * @brief The ComputeRmsDesc class should compute the root mean square value of a signal.
 *
 * It has to start over from the beginning if anything is changed, as indicated by a call
 * to affectedInterval.
 */
class ComputeRmsDesc : public Signal::OperationDesc
{
public:
    ComputeRmsDesc();

    Signal::Interval requiredInterval( const Signal::Interval& I, Signal::Interval* expectedOutput ) const;
    Signal::Interval affectedInterval( const Signal::Interval& I ) const;
    Signal::OperationDesc::Ptr copy() const;
    Signal::Operation::Ptr createOperation(Signal::ComputingEngine* engine=0) const;

    float rms();

private:
    shared_state<RmsValue> rms_;

public:
    static void test();
};
} // namespace Support
} // namespace Tools

#endif // COMPUTERMS_H
