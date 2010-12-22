#include "ellipsemodel.h"
#include "filters/ellipse.h"
#include "tools/support/operation-composite.h"

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
}


EllipseModel::
        ~EllipseModel()
{
    TaskTimer(__FUNCTION__).suppressTiming();
}


Signal::pOperation EllipseModel::
        updateFilter()
{
    Signal::pOperation filter( new Filters::Ellipse( 0,0,0,0, true ) );

    Filters::Ellipse* e = dynamic_cast<Filters::Ellipse*>(filter.get());

    e->_t1 = a.time;
    e->_t2 = b.time;
    e->_f1 = fa_.getFrequency( a.scale );
    e->_f2 = fa_.getFrequency( b.scale );

    return filter;
}

} } // namespace Tools::Selections
