#include "ellipsemodel.h"
#include "filters/ellipse.h"
#include "tools/support/operation-composite.h"
#include "tools/rendermodel.h"
#include "tfr/transformoperation.h"

namespace Tools { namespace Selections
{

EllipseModel::
        EllipseModel( RenderModel* rendermodel )
            : rendermodel_(rendermodel)
{
    float l = 8;
    centre.time = l*.5f;
    centre.scale = .85f;
    centrePlusRadius.time = l*sqrt(2.0f);
    centrePlusRadius.scale = 2;

    // no selection
    centre.time = centrePlusRadius.time;
    centre.scale = centrePlusRadius.scale;
}


EllipseModel::
        ~EllipseModel()
{
    TaskTimer(__FUNCTION__).suppressTiming();
}


Signal::OperationDesc::Ptr EllipseModel::
        updateFilter()
{
    if (centre.time == centrePlusRadius.time || centre.scale == centrePlusRadius.scale)
        return Signal::OperationDesc::Ptr();

    Filters::Ellipse* e;
    Tfr::ChunkFilterDesc::Ptr filter( e = new Filters::Ellipse( 0,0,0,0, true ) );

    e->_centre_t = centre.time;
    e->_centre_plus_radius_t = centrePlusRadius.time;
    e->_centre_f = freqAxis().getFrequency( centre.scale );
    e->_centre_plus_radius_f = freqAxis().getFrequency( centrePlusRadius.scale );

    Tfr::TransformDesc::Ptr t = read1(filter)->transformDesc();
    Signal::OperationDesc::Ptr(new Tfr::TransformOperationDesc(e->ChunkFilterDesc::transformDesc (), filter));
    return Signal::OperationDesc::Ptr(new Tfr::TransformOperationDesc(t, filter));
}


void EllipseModel::
        tryFilter(Signal::OperationDesc::Ptr filterp)
{
    Signal::OperationDesc::ReadPtr filter(filterp);
    const Filters::Ellipse* e = dynamic_cast<const Filters::Ellipse*>(filter.get());
    if (!e)
    {
        centrePlusRadius.time = centre.time;
        centrePlusRadius.scale = centre.scale;
        return;
    }

    centre.time = e->_centre_t;
    centrePlusRadius.time = e->_centre_plus_radius_t;
    centre.scale = freqAxis().getFrequencyScalar( e->_centre_f );
    centrePlusRadius.scale = freqAxis().getFrequencyScalar( e->_centre_plus_radius_f );
}


Tfr::FreqAxis EllipseModel::
        freqAxis()
{
    return rendermodel_->display_scale();
}

} } // namespace Tools::Selections
