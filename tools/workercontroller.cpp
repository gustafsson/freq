#include "workercontroller.h"
#include "renderview.h"
#include "timelineview.h"

namespace Tools {

WorkerController::
        WorkerController(WorkerView* view, RenderView* renderview, TimelineView* timelineview)
    :
    view_(view)
{
    connect( renderview, SIGNAL(painting()), view_, SLOT(draw()) );
    connect( timelineview, SIGNAL(painting()), view_, SLOT(draw()) );
}


} // namespace Tools
