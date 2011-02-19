#include "rectanglemodel.h"
#include "filters/rectangle.h"
#include "tools/support/operation-composite.h"
#include "sawe/project.h"

#ifdef max
#undef max
#undef min
#endif

namespace Tools { namespace Selections
{

RectangleModel::
        RectangleModel( Tfr::FreqAxis const& fa, Sawe::Project* project )
            :
            fa_(fa),
            project_(project)
{
    // no selection
    a.time = b.time = 0;
    a.scale = b.scale = 0;
}


RectangleModel::
        ~RectangleModel()
{
    TaskTimer(__FUNCTION__).suppressTiming();
}


Signal::pOperation RectangleModel::
        updateFilter()
{
    validate();

    float
            f1 = fa_.getFrequency( a.scale ),
            f2 = fa_.getFrequency( b.scale );

    float FS = project_->head->head_source()->sample_rate();
    Signal::IntervalType
            a_index = std::max(0.f, a.time)*FS,
            b_index = std::max(0.f, b.time)*FS;

    Signal::pOperation filter;

    if (a.scale>=1 || b.scale<=0 || a_index==b_index || a.scale==b.scale)
        ;
    else if (a.scale>0 || b.scale<1)
    {
        filter.reset( new Filters::Rectangle(
                a.time, f1, b.time, f2, true ));
    }
    else
    {
        filter.reset( new Tools::Support::OperationOtherSilent(
                Signal::pOperation(), Signal::Interval( a_index, b_index) ));
    }

    return filter;
}


void RectangleModel::
        validate()
{
    switch (type)
    {
    case RectangleType_RectangleSelection:
        break;

    case RectangleType_FrequencySelection:
        a.time = 0;
        b.time = project_->head->head_source()->length();
        break;

    case RectangleType_TimeSelection:
        a.scale = 0;
        b.scale = 1;
        break;
    }

    float
            t1=a.time,
            t2=b.time,
            f1=a.scale,
            f2=b.scale;

    a.time = std::min( t1, t2 );
    b.time = std::max( t1, t2 );
    a.scale = std::min( f1, f2 );
    b.scale = std::max( f1, f2 );
}


} } // namespace Tools::Selections
