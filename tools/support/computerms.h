#ifndef COMPUTERMS_H
#define COMPUTERMS_H

#include "signal/operation.h"

namespace Tools {
namespace Support {

class ComputeRms : public Signal::Operation
{
public:
    ComputeRms(Signal::pOperation);
    virtual Signal::pBuffer read( const Signal::Interval& I );
    virtual void invalidate_samples(const Signal::Intervals& I);

    Signal::Intervals rms_I;
    double rms;
};

} // namespace Support
} // namespace Tools

#endif // COMPUTERMS_H
