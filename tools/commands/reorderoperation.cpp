#include "reorderoperation.h"
#include <boost/foreach.hpp>

#include <sstream>

namespace Tools {
namespace Commands {

ReorderOperation::
        ReorderOperation(Signal::pOperation operation, Signal::pOperation newSource)
            :
            operation(operation),
            newSource(newSource)
{
}


void ReorderOperation::
        execute()
{
    oldSource = operation->source();
    BOOST_FOREACH( Signal::Operation*o, operation->outputs())
    {
        o->source( oldSource );
    }

    // TODO maybe we should just change newSource->outputs()->source for a specific target
    BOOST_FOREACH( Signal::Operation*o, newSource->outputs())
    {
        o->source( operation );
    }

    operation->source( newSource );
}


void ReorderOperation::
        undo()
{
    BOOST_FOREACH( Signal::Operation*o, operation->outputs())
    {
        o->source( newSource );
    }

    // TODO maybe we should just change oldSource->outputs()->source for a specific target
    BOOST_FOREACH( Signal::Operation*o, oldSource->outputs())
    {
        o->source( operation );
    }

    operation->source( oldSource );
}


std::string ReorderOperation::
        toString()
{
    std::stringstream ss;
    ss << "Reorder " << operation->toString();
    return ss.str();
}


} // namespace Commands
} // namespace Tools
