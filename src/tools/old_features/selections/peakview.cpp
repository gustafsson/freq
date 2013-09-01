#if 0
#include "peakview.h"
#include "peakmodel.h"

#include <gl.h>
#include <TaskTimer.h>
#include <glPushContext.h>

namespace Tools { namespace Selections
{


PeakView::PeakView(PeakModel* model, Signal::Worker* worker)
    :
    visible(true),
    enabled(false),
    spline_view( &model->spline_model, worker ),
    model_(model),
    worker_(worker)
{
}


PeakView::
        ~PeakView()
{
    TaskTimer(__FUNCTION__).suppressTiming();
}


void PeakView::
        draw()
{
    if (visible)
        drawSelectionPeak();
}


void PeakView::
        drawSelectionPeak()
{
    spline_view.enabled = enabled;
    spline_view.drawSelectionSpline();
}


}} // namespace Tools::Selections
#endif
