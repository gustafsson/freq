#include "dag.h"

#include <boost/graph/adjacency_list.hpp>

namespace og = boost::graph;

namespace Signal {
namespace Processing {

Dag::
        Dag()
{
}


GraphVertex Dag::
        getVertex(Step::Ptr s) const
{
    StepVertexMap::const_iterator i = map.find (s);
    EXCEPTION_ASSERT (i == map.end ());

    return i->second;
}


void Dag::
        insertStep(GraphVertex /*gv*/, Step::Ptr /*step*/)
{

}


void Dag::
        removeStep(Step::Ptr /*step*/)
{

}


void Dag::
        test()
{
}

} // namespace Processing
} // namespace Signal
