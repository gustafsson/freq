#include "ellipsemodel.h"
#include "filters/ellipse.h"

namespace Tools { namespace Selections
{

EllipseModel::
        EllipseModel( Tfr::FreqAxis const& fa )
            : fa_(fa)
{
    float l = 8;
    a.time = l*.5f;
    a.scale = .85f;
    b.time = l*sqrt(2.0f);
    b.scale = 2;

    // no selection
    a.time = b.time;
    a.scale = b.scale;

    filter.reset( new Filters::Ellipse( 0,0,0,0, true ) );
    updateFilter();
}


EllipseModel::
        ~EllipseModel()
{
    TaskTimer(__FUNCTION__).suppressTiming();
}


void EllipseModel::
        updateFilter()
{
    Filters::Ellipse* e = dynamic_cast<Filters::Ellipse*>(filter.get());

    e->_t1 = a.time;
    e->_t2 = b.time;
    e->_f1 = fa_.getFrequency( a.scale );
    e->_f2 = fa_.getFrequency( b.scale );
}

} } // namespace Tools::Selections
