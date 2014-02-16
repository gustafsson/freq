#include "peakview.h"
#include "peakmodel.h"

#include "gl.h"
#include "tasktimer.h"
#include "glPushContext.h"

namespace Tools { namespace Selections
{


PeakView::PeakView(PeakModel* model)
    :
    visible(true),
    enabled(false),
    spline_view( &model->spline_model ),
    model_(model)
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
