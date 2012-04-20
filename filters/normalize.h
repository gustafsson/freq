#ifndef NORMALIZE_H
#define NORMALIZE_H

#include "signal/operation.h"
namespace Filters {

class Normalize: public Signal::Operation
{
public:
    enum Method
    {
        Method_None,
        Method_Standard,
        Method_TruncatedMean
    };

    Normalize( unsigned normalizationRadius, Signal::pOperation source = Signal::pOperation() );

    virtual std::string name();
    virtual Signal::pBuffer read( const Signal::Interval& I );

private:
    unsigned normalizationRadius;
};

} // namespace Filters
#endif // NORMALIZE_H
