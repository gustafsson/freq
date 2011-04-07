#include "rectanglemodel.h"
#include "filters/rectangle.h"
#include "tools/support/operation-composite.h"
#include "sawe/project.h"
#include "tools/rendermodel.h"

#ifdef max
#undef max
#undef min
#endif

namespace Tools { namespace Selections
{

RectangleModel::
        RectangleModel( RenderModel* rendermodel, Sawe::Project* project )
            :
            rendermodel_(rendermodel),
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
            f1 = freqAxis().getFrequency( a.scale ),
            f2 = freqAxis().getFrequency( b.scale );

    float FS = project_->head->head_source()->sample_rate();
    Signal::IntervalType
            a_index = std::max(0.f, a.time)*FS,
            b_index = std::max(0.f, b.time)*FS;

    Signal::pOperation filter;

    if (a.scale>=1 || b.scale<=0 || a_index==b_index || a.scale==b.scale)
        ;
    else if (a.scale>0 || b.scale<1)
    {
        float bt = b.time;
        if (type == RectangleType_FrequencySelection)
            bt = FLT_MAX;
        filter.reset( new Filters::Rectangle(
                a.time, f1, bt, f2, true ));
    }
    else
    {
        filter.reset( new Tools::Support::OperationOtherSilent(
                FS, Signal::Interval( a_index, b_index) ));
    }

    return filter;
}


bool RectangleModel::
        tryFilter(Signal::pOperation filter)
{
    Filters::Rectangle* e = dynamic_cast<Filters::Rectangle*>(filter.get());
    Tools::Support::OperationOtherSilent* os = dynamic_cast<Tools::Support::OperationOtherSilent*>(filter.get());
    float FS = project_->head->head_source()->sample_rate();
    if (e)
    {
        a.time = e->_t1;
        b.time = e->_t2;
        a.scale = freqAxis().getFrequencyScalar( e->_f1 );
        b.scale = freqAxis().getFrequencyScalar( e->_f2 );
        validate();
        return true;
    }
    else if(os)
    {
        Signal::Interval section = os->section();
        a.time = section.first/FS;
        b.time = section.last/FS;
        a.scale = 0;
        b.scale = 1;
        return true;
    }
    else
    {
        b.time = a.time;
        b.scale = a.scale;
        return false;
    }
}



void RectangleModel::
        validate()
{
    float L = project_->head->head_source()->length();
    switch (type)
    {
    case RectangleType_RectangleSelection:
        break;

    case RectangleType_FrequencySelection:
        a.time = 0;
        b.time = L;
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

    if (a.time<0) a.time = 0;
    if (b.time<0) b.time = 0;
    if (a.time>L) a.time = L;
    if (b.time>L) b.time = L;
    if (a.scale<0) a.scale = 0;
    if (b.scale<0) b.scale = 0;
    if (a.scale>1) a.scale = 1;
    if (b.scale>1) b.scale = 1;
}


Tfr::FreqAxis RectangleModel::
        freqAxis()
{
    return rendermodel_->display_scale();
}


} } // namespace Tools::Selections
