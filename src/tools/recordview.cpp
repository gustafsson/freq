#include "recordview.h"

#include "recordmodel.h"
#include "tools/renderview.h"
#include "ui/mainwindow.h"
#include "ui_mainwindow.h"

#include "adapters/microphonerecorder.h"
#include "tfr/cwt.h"
#include "sawe/project.h"

#include <QErrorMessage>

namespace Tools
{

RecordView::
        RecordView(RecordModel* model)
            :
            enabled_(false),
            model_(model),
            prev_limit_(0)
{
    prev_limit_ = model_->project->length ();
}


RecordView::
        ~RecordView()
{

}


void RecordView::
        setEnabled(bool v)
{
    enabled_ = v;
}


void RecordView::
        prePaint()
{
    if (enabled_)
    {
//Use Signal::Processing namespace
/*
#ifndef SAWE_NO_MUTEX
        if (!model_->project->worker.isRunning())
        {
            Ui::MainWindow* ui = model_->project->mainWindow()->getItems();
            ui->actionRecord->setChecked(false);
        }
#endif
*/
//        float limit = model_->project->worker.length();

        double limit = model_->project->length ();
        limit -= 1/model_->render_view->model->camera.xscale;
        if (limit<0) limit = 0;

        if (model_->render_view->model->camera.q[0] >= prev_limit_) {
            // -- Following Record Marker --
            // Snap just before end so that project->worker.center starts working on
            // data that has been fetched. If center=length worker will start
            // at the very end and have to assume that the signal is abruptly
            // set to zero after the end. This abrupt change creates a false
            // dirac peek in the transform (false because it will soon be
            // invalid by newly recorded data).
            model_->render_view->model->camera.q[0] = std::max(model_->render_view->model->camera.q[0], limit);

            bool ismicrophonerecorder = (0 != dynamic_cast<Adapters::MicrophoneRecorder*>(model_->recording.raw ()));
            if ( ismicrophonerecorder && model_->recording.write ()->time_since_last_update() > 5 )
            {
                QErrorMessage::qtHandler()->showMessage(
                    "It looks like your recording device doesn't report any "
                    "data, so the recording has been stopped.\n"
                    "Restarting the recording might help temporarily. "
                    "You can also try another recording device in the "
                    "settings dialog.", "No data from recording device");
                emit gotNoData();
            }
        }
        prev_limit_ = limit;

        model_->project->setModified();
    }
}

} // namespace Tools
