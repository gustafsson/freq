#include "fantrackercontroller.h"
#include "fantrackerview.h"

//#include "sawe/project.h"
#include "ui_mainwindow.h"
#include "ui/mainwindow.h"

namespace Tools {

    FanTrackerController::FanTrackerController(FanTrackerView* view, RenderView* render_view)
{
    render_view_ = render_view;
    view_ = view;
    setupGui();
}


void FanTrackerController::
        setupGui()
{
    ::Ui::MainWindow* ui = render_view_->model->project()->mainWindow()->getItems();

    connect(ui->actionFanTracker, SIGNAL(triggered()), SLOT(receiveFanTracker()));
    connect(render_view_, SIGNAL(painting()), view_, SLOT(draw()));

}


void FanTrackerController::
        receiveFanTracker()
{
    //skapa operation, pOperation
    //lägga till i fantrackermodell,
    //lägga till i projktet, via project()->appendOperation(pOperation)
}

} // namespace Tools
