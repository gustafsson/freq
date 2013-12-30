#include "splinemodel.h"
#include "support/splinefilter.h"
#include "tools/support/operation-composite.h"
#include "tools/rendermodel.h"

namespace Tools { namespace Selections
{

SplineModel::
        SplineModel( RenderModel* rendermodel )
            : drawing ( false ),
            rendermodel_( rendermodel )
{
}


SplineModel::
        ~SplineModel()
{
    TaskTimer(__FUNCTION__).suppressTiming();
}


Signal::OperationDesc::Ptr SplineModel::
        updateFilter()
{
    if (v.size() < 3)
    {
        v.clear();
        return Signal::OperationDesc::Ptr();
    }

    std::vector<Support::SplineFilter::SplineVertex> ev;

    ev.resize( v.size() );

    for (unsigned i=0; i<v.size(); ++i)
    {
        Support::SplineFilter::SplineVertex s;
        s.t = v[i].time;
        s.f = freqAxis().getFrequency( v[i].scale );
        ev[i] = s;
    }

    Signal::OperationDesc::Ptr filter( new Support::SplineFilterDesc( true, ev ) );
    return filter;
}


Tfr::FreqAxis SplineModel::
        freqAxis()
{
    return rendermodel_->display_scale();
}


} } // namespace Tools::Selections
