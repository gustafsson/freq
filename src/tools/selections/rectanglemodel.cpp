#include "rectanglemodel.h"
#include "filters/rectangle.h"
#include "filters/bandpass.h"
#include "tools/support/operation-composite.h"
#include "sawe/project.h"
#include "tools/rendermodel.h"
#include "signal/operation-basic.h"

#ifdef max
#undef max
#undef min
#endif

namespace Tools { namespace Selections
{

RectangleModel::
        RectangleModel( RenderModel* rendermodel, Sawe::Project* project )
            :
            type(RectangleType_RectangleSelection),
            rendermodel_(rendermodel),
            project_(project)
{
}


RectangleModel::
        ~RectangleModel()
{
    TaskTimer(__FUNCTION__).suppressTiming();
}


bool RectangleModel::
        replaceFilter( Signal::OperationDesc::Ptr filter )
{
    volatile Signal::OperationSetSilent* oss = dynamic_cast<volatile Signal::OperationSetSilent*>(filter.get());
    if (oss)
        return true;
    return false;
}


Signal::OperationDesc::Ptr RectangleModel::
        updateFilter()
{
    validate();

    float
            f1 = freqAxis().getFrequency( a.scale ),
            f2 = freqAxis().getFrequency( b.scale );

    float FS = project_->extent ().sample_rate.get ();
    Signal::IntervalType L = project_->length ()*FS;
    Signal::IntervalType
            a_index = std::max(0.f, a.time)*FS,
            b_index = std::max(0.f, b.time)*FS;

    Signal::OperationDesc::Ptr filter;

    if (a.scale>=1 || b.scale<=0)
        ;
    else if (a_index>=L || b_index<=0)
        ;
    else if(a_index==b_index)
    {
        if (a.scale==b.scale || (a.scale==0 && b.scale==1))
            filter.reset( new Tools::Support::OperationOtherSilent(
                Signal::Interval( a_index, L) ));
    }
    else if (a.scale>0 || b.scale<1)
    {
        if (type == RectangleType_FrequencySelection)
            filter.reset( new Filters::Bandpass(f1, f2, true ));
        else
            filter.reset( new Filters::Rectangle(
                a.time, f1, b.time, f2, true ));
    }
    else
    {
        filter.reset( new Tools::Support::OperationOtherSilent(
                Signal::Interval( a_index, b_index) ));
    }

    return filter;
}


bool RectangleModel::
        tryFilter(Signal::OperationDesc::Ptr filterp)
{
    Signal::OperationDesc::WritePtr filter(filterp);
    Filters::Rectangle* e = dynamic_cast<Filters::Rectangle*>(filter.get());
    Filters::Bandpass* bp = dynamic_cast<Filters::Bandpass*>(filter.get());
    Tools::Support::OperationOtherSilent* oos = dynamic_cast<Tools::Support::OperationOtherSilent*>(filter.get());
    Signal::OperationSetSilent* oss = dynamic_cast<Signal::OperationSetSilent*>(filter.get());
    float FS = project_->extent ().sample_rate.get ();
    if (e)
    {
        type = RectangleType_RectangleSelection;
        a.time = e->_t1;
        b.time = e->_t2;
        a.scale = freqAxis().getFrequencyScalar( e->_f1 );
        b.scale = freqAxis().getFrequencyScalar( e->_f2 );
        validate();
        return true;
    }
    else if(bp)
    {
        type = RectangleType_FrequencySelection;
        a.time = 0;
        b.time = FLT_MAX;
        a.scale = freqAxis().getFrequencyScalar( bp->_f1 );
        b.scale = freqAxis().getFrequencyScalar( bp->_f2 );
        validate();
        return true;
    }
    else if(oos)
    {
        type = RectangleType_TimeSelection;
        Signal::Interval section = oos->section();
        a.time = section.first/FS;
        b.time = section.last/FS;
        a.scale = 0;
        b.scale = 1;
        return true;
    }
    else if(oss)
    {
        type = RectangleType_TimeSelection;
        Signal::Interval section = oss->section();
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
    float L = project_->length ();
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
