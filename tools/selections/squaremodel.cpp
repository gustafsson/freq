#include "squaremodel.h"
#include "filters/rectangle.h"

namespace Tools { namespace Selections
{

SquareModel::
        SquareModel( Tfr::FreqAxis const& fa )
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

    filter.reset( new Filters::Rectangle( 0,0,0,0, true ) );
    updateFilter();
}


SquareModel::
        ~SquareModel()
{
    TaskTimer(__FUNCTION__).suppressTiming();
}


void SquareModel::
        updateFilter()
{
    Filters::Rectangle* e = dynamic_cast<Filters::Rectangle*>(filter.get());

    float
            f1 = fa_.getFrequency( a.scale ),
            f2 = fa_.getFrequency( b.scale );

    e->_t1 = std::min( a.time, b.time );
    e->_t2 = std::max( a.time, b.time );
    e->_f1 = std::min( f1, f2 );
    e->_f2 = std::max( f1, f2 );
}

} } // namespace Tools::Selections
