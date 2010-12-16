#include "squaremodel.h"
#include "filters/rectangle.h"
#include "tools/support/operation-composite.h"
#include "sawe/project.h"

namespace Tools { namespace Selections
{

SquareModel::
        SquareModel( Tfr::FreqAxis const& fa, Sawe::Project* project )
            :
            fa_(fa),
            project_(project)
{
    // no selection
    a.time = b.time = 0;
    a.scale = b.scale = 0;

    Tools::Support::OperationContainer* container =
            new Tools::Support::OperationContainer(Signal::pOperation(), "Rectangle selection");
    filter.reset( container );
}


SquareModel::
        ~SquareModel()
{
    TaskTimer(__FUNCTION__).suppressTiming();
}


void SquareModel::
        updateFilter()
{
    validate();

    float
            f1 = fa_.getFrequency( a.scale ),
            f2 = fa_.getFrequency( b.scale );

    Tools::Support::OperationContainer* container =
            dynamic_cast<Tools::Support::OperationContainer*>(filter.get());

    BOOST_ASSERT(container);

    float FS;
    if (container->source())
        FS = container->sample_rate();
    else
        FS = project_->head_source()->sample_rate();
    Signal::IntervalType
            a_index = std::max(0.f, a.time)*FS,
            b_index = std::max(0.f, b.time)*FS;

    Signal::Operation* op = 0;

    if (a.scale>=1 || b.scale<=0 || a_index==b_index || a.scale==b.scale)
        ;
    else if (a.scale>0 || b.scale<1)
    {
        op = new Filters::Rectangle(
                a.time, f1, b.time, f2, true );
    }
    else
    {
        op = new Tools::Support::OperationOtherSilent(
                Signal::pOperation(), Signal::Interval( a_index, b_index) );
    }

    container->setContent(Signal::pOperation(op));
}


void SquareModel::
        validate()
{
    switch (type)
    {
    case SquareType_SquareSelection:
        break;

    case SquareType_FrequencySelection:
        a.time = 0;
        b.time = project_->head_source()->length();
        break;

    case SquareType_TimeSelection:
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
