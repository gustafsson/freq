#include "workercontroller.h"
#include "tools/renderview.h"
#include "timelineview.h"
#include "sawe/project.h"
#include "ui/mainwindow.h"

namespace Tools {

WorkerController::
        WorkerController(WorkerView* view, RenderView* renderview, TimelineView* timelineview, Sawe::Project* project)
    :
    view_(view),
    renderview_(renderview),
    timelineview_(timelineview)
{
    Ui::SaweMainWindow* mainwindow = project->mainWindow();
    QAction* a = new QAction(mainwindow);
    a->setShortcut(QKeySequence("Ctrl+Alt+Shift+W"));
    a->setCheckable( true );
    a->setChecked( false );
    connect(a, SIGNAL(toggled(bool)), SLOT(setEnabled(bool)));

    mainwindow->addAction( a );
}


void WorkerController::
        setEnabled( bool enabled )
{
    if (enabled)
    {
        connect( renderview_, SIGNAL(painting()), view_, SLOT(draw()) );
        connect( timelineview_, SIGNAL(painting()), view_, SLOT(draw()) );
    }
    else
    {
        disconnect( renderview_, SIGNAL(painting()), view_, SLOT(draw()) );
        disconnect( timelineview_, SIGNAL(painting()), view_, SLOT(draw()) );
    }
}

} // namespace Tools
