#include "renderviewupdateadapter.h"

#include "TaskTimer.h"

//#define UPDATEINFO
#define UPDATEINFO if(0)

using namespace boost;
using namespace Signal;

namespace Tools {
namespace Support {


RenderViewUpdateAdapter::
        RenderViewUpdateAdapter()
{
    qRegisterMetaType<Signal::UnsignedIntervalType>("Signal::UnsignedIntervalType");
}


// overloaded from Support::RenderOperationDesc::RenderTarget
void RenderViewUpdateAdapter::
        refreshSamples(const Intervals& I)
{
    UPDATEINFO TaskInfo(format("refreshSamples %s") % I);

    emit setLastUpdateSize( I.count () );
}


void RenderViewUpdateAdapter::
        processedData(const Interval& input, const Interval& output)
{
    UPDATEINFO TaskInfo(format("processedData %s -> %s") % input % output);

    emit redraw ();
}

} // namespace Support
} // namespace Tools


namespace Tools {
namespace Support {

void RenderViewUpdateAdapterMock::
        redraw()
{
    redraw_count++;
}


void RenderViewUpdateAdapterMock::
        setLastUpdateSize( Signal::UnsignedIntervalType )
{
    setLastUpdateSize_count++;
}


void RenderViewUpdateAdapter::
        test()
{
    // It should translate the Support::RenderOperationDesc::RenderTarget
    // interface to Qt signals/slots that match RenderView.
    {
        RenderViewUpdateAdapter* a;
        RenderTarget::Ptr rt(a = new RenderViewUpdateAdapter);
        RenderViewUpdateAdapterMock mock;

        connect(a, SIGNAL(setLastUpdateSize(Signal::UnsignedIntervalType)), &mock, SLOT(setLastUpdateSize(Signal::UnsignedIntervalType)));
        connect(a, SIGNAL(redraw()), &mock, SLOT(redraw()));

        EXCEPTION_ASSERT_EQUALS(mock.redraw_count, 0);
        EXCEPTION_ASSERT_EQUALS(mock.setLastUpdateSize_count, 0);

        write1(rt)->refreshSamples(Signal::Intervals(1,2));

        EXCEPTION_ASSERT_EQUALS(mock.redraw_count, 0);
        EXCEPTION_ASSERT_EQUALS(mock.setLastUpdateSize_count, 1);

        write1(rt)->processedData(Signal::Interval(1,2), Signal::Interval(3,4));

        EXCEPTION_ASSERT_EQUALS(mock.redraw_count, 1);
        EXCEPTION_ASSERT_EQUALS(mock.setLastUpdateSize_count, 1);
    }

    // It should not rely on a valid instance of RenderView
    {
        // Implemented by not using RenderView at all
    }
}

} // namespace Support
} // namespace Tools
