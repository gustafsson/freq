#include "splinemodel.h"
#include "support/splinefilter.h"
#include "tools/support/operation-composite.h"
 
namespace Tools { namespace Selections
{

SplineModel::
        SplineModel( Tfr::FreqAxis const& fa )
            : drawing ( false ),
            fa_( fa )
{
}


SplineModel::
        ~SplineModel()
{
    TaskTimer(__FUNCTION__).suppressTiming();
}


Signal::pOperation SplineModel::
        updateFilter()
{
    if (v.size() < 3)
    {
        v.clear();
        return Signal::pOperation();
    }

    Support::SplineFilter* e;
    Signal::pOperation filter( e = new Support::SplineFilter( true ) );

    e->v.resize( v.size() );

    for (unsigned i=0; i<v.size(); ++i)
    {
        Support::SplineFilter::SplineVertex s;
        s.t = v[i].time;
        s.f = fa_.getFrequency( v[i].scale );
        e->v[i] = s;
    }

    return filter;
}

} } // namespace Tools::Selections
