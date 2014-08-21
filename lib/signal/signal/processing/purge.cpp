#include "purge.h"
#include "log.h"
#include "datastorage.h"

#include <boost/foreach.hpp>

namespace Signal {
namespace Processing {

Purge::
        Purge(Dag::ptr::weak_ptr dag)
    :
      dag(dag)
{
}


size_t recursive_purge(const Graph& g, const GraphVertex& v, Signal::Intervals out_of_date)
{
    size_t purged = 0;
    Step::ptr step = g[v];
    purged += step->purge(out_of_date);

    Signal::Intervals required_input;
    Signal::OperationDesc::ptr operation_desc = step->operation_desc();

    if (operation_desc)
    {
        auto o = operation_desc.read ();
        while (out_of_date)
        {
            Signal::Interval expected_output;
            required_input |= o->requiredInterval (out_of_date.fetchFirstInterval (), &expected_output);
            out_of_date -= expected_output;
        }
    }
    else
    {
        required_input = out_of_date;
    }

    BOOST_FOREACH(GraphEdge e, in_edges(v, g))
        purged += recursive_purge(g, source(e,g), required_input);

    return purged;
}


void Purge::
        purge(TargetNeeds::ptr needs)
{
    Dag::ptr dag = this->dag.lock ();
    Step::ptr step = needs->step ().lock ();
    Signal::Intervals out_of_date = needs->out_of_date();

    if (!dag || !step)
        return;

    auto rdag = dag.read ();

    size_t purged = recursive_purge(rdag->g(), rdag->getVertex(step), out_of_date);
    if (0 < purged)
        Log("Purged %s from the cache") % DataStorageVoid::getMemorySizeText (purged*sizeof(float));
}

} // namespace Processing
} // namespace Signal
