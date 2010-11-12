#include "splinemodel.h"
#include "support/splinefilter.h"

#include <boost/foreach.hpp>

namespace Tools { namespace Selections
{

SplineModel::
        SplineModel( Tfr::FreqAxis const& fa )
            : fa_( fa ),
              drawing ( false )
{
    filter.reset( new Support::SplineFilter( true ) );
    updateFilter();
}


SplineModel::
        ~SplineModel()
{
    TaskTimer(__FUNCTION__).suppressTiming();
}


void SplineModel::
        updateFilter()
{
    Support::SplineFilter* e = dynamic_cast<Support::SplineFilter*>(filter.get());

    if (e->v.size() != v.size())
        e->v.resize( v.size() );

    for (unsigned i=0; i<v.size(); ++i)
    {
        Support::SplineFilter::SplineVertex s;
        s.t = v[i].time;
        s.f = fa_.getFrequency( v[i].scale );
        e->v[i] = s;
    }
}

} } // namespace Tools::Selections
