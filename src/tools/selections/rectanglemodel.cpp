#include "rectanglemodel.h"
#include "filters/rectangle.h"
#include "filters/bandpass.h"
#include "filters/timeselection.h"
#include "sawe/project.h"
#include "tools/rendermodel.h"
#include "tfr/transformoperation.h"

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
            select_interior(true),
            rendermodel_(rendermodel),
            project_(project)
{
}


RectangleModel::
        ~RectangleModel()
{
    TaskTimer(__FUNCTION__).suppressTiming();
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
            filter.reset( new Filters::TimeSelection(
                Signal::Interval( a_index, L), select_interior ));
    }
    else if (a.scale>0 || b.scale<1)
    {
        Tfr::ChunkFilterDesc::Ptr cfd;
        if (type == RectangleType_FrequencySelection)
            cfd.reset( new Filters::Bandpass(f1, f2, true ));
        else
            cfd.reset( new Filters::Rectangle(
                a.time, f1, b.time, f2, true ));

        Tfr::TransformDesc::Ptr t = read1(cfd)->transformDesc();
        filter.reset ( new Tfr::TransformOperationDesc(t, cfd));
    }
    else
    {
        filter.reset( new Filters::TimeSelection(
                Signal::Interval( a_index, b_index), select_interior ));
    }

    return filter;
}


bool RectangleModel::
        tryFilter(Signal::OperationDesc::Ptr filterp)
{
    Signal::OperationDesc::WritePtr filter(filterp);
    Tfr::TransformOperationDesc* tod = dynamic_cast<Tfr::TransformOperationDesc*>(&*filter);
    Filters::Rectangle* e = dynamic_cast<Filters::Rectangle*>(tod?&*tod->chunk_filter ():0);
    Filters::Bandpass* bp = dynamic_cast<Filters::Bandpass*>(tod?&*tod->chunk_filter ():0);
    Filters::TimeSelection* ts = dynamic_cast<Filters::TimeSelection*>(&*filter);

    Filters::Selection* s = dynamic_cast<Filters::Selection*>(tod?&*tod->chunk_filter ():0);
    if (!s) s = dynamic_cast<Filters::Selection*>(&*filter);
    if (s)  select_interior = s->isInteriorSelected();

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
    else if(ts)
    {
        type = RectangleType_TimeSelection;
        Signal::Interval section = ts->section();
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
