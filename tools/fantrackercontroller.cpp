#include "fantrackercontroller.h"
#include "fantrackerview.h"

//#include "sawe/project.h"
#include "ui_mainwindow.h"
#include "ui/mainwindow.h"
#include "tools/support/fantrackerfilter.h"
#include "tfr/cepstrum.h"

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

    connect(ui->actionToggleFanTracker, SIGNAL(toggled(bool)), SLOT(receiveToggleFanTracker(bool)));

    connect(render_view_, SIGNAL(painting()), view_, SLOT(draw()));

}


void FanTrackerController::
        receiveToggleFanTracker(bool value)
{
    Signal::PostSink* ps = render_view_->model->renderSignalTarget->post_sink();

    if (value)
    {
        Tools::Support::FanTrackerFilter* filter;
        Signal::pOperation pfilter ( filter = new Tools::Support::FanTrackerFilter());

        view_->model_->filter = pfilter;

        ps->filter( pfilter );
    }

    else
    {
        ps->filter( Signal::pOperation() );
        view_->model_->filter.reset();
    }

    ps->invalidate_samples(Signal::Intervals::Intervals_ALL);

    render_view_->userinput_update();

}

} // namespace Tools
