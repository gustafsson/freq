#ifndef FILTERS_ABSOLUTEVALUE_H
#define FILTERS_ABSOLUTEVALUE_H

#include "signal/operation.h"

namespace Filters {

class AbsoluteValue : public Signal::Operation
{
public:
    AbsoluteValue();

    Signal::pBuffer read( const Signal::Interval& I );
};

} // namespace Filters

#endif // FILTERS_ABSOLUTEVALUE_H
