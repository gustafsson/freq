#include "peakview.h"
#include "peakmodel.h"

#include <GL/gl.h>
#include <TaskTimer.h>
#include <glPushContext.h>

namespace Tools { namespace Selections
{


PeakView::PeakView(PeakModel* model, Signal::Worker* worker)
    :
    enabled(false),
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
    drawSelectionPeak();
}


void PeakView::
        drawSelectionPeak()
{

}


}} // namespace Selections::Tools
