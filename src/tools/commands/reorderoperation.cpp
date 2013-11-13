#include "reorderoperation.h"
#include <boost/foreach.hpp>

#include <sstream>

namespace Tools {
namespace Commands {

ReorderOperation::
        ReorderOperation(Signal::pOperation operation, Signal::pOperation operationTail, Signal::pOperation newSource)
            :
            operation(operation),
            operationTail(operationTail),
            newSource(newSource)
{
}


void ReorderOperation::
        execute()
{
    oldSource = operationTail->source();
    operationTail->source()->invalidate_samples(Signal::DeprecatedOperation::affectedDiff( operation, oldSource ));

    BOOST_FOREACH( Signal::DeprecatedOperation*o, operation->outputs())
    {
        o->source( oldSource );
    }

    // TODO maybe we should just change newSource->outputs()->source for a specific target
    BOOST_FOREACH( Signal::DeprecatedOperation*o, newSource->outputs())
    {
        o->source( operation );
    }

    operationTail->source( newSource );

    operationTail->source()->invalidate_samples(Signal::DeprecatedOperation::affectedDiff( operation, newSource ));
}


void ReorderOperation::
        undo()
{
    operationTail->source()->invalidate_samples(Signal::DeprecatedOperation::affectedDiff( operation, newSource ));

    BOOST_FOREACH( Signal::DeprecatedOperation*o, operation->outputs())
    {
        o->source( newSource );
    }

    // TODO maybe we should just change oldSource->outputs()->source for a specific target
    BOOST_FOREACH( Signal::DeprecatedOperation*o, oldSource->outputs())
    {
        o->source( operation );
    }

    operationTail->source( oldSource );

    operationTail->source()->invalidate_samples(Signal::DeprecatedOperation::affectedDiff( operation, oldSource ));
}


std::string ReorderOperation::
        toString()
{
    std::stringstream ss;
    ss << "Reorder " << operationTail->name();
    return ss.str();
}


} // namespace Commands
} // namespace Tools
