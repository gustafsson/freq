#include "recordview.h"

#include "recordmodel.h"
#include "renderview.h"
#include "adapters/microphonerecorder.h"
#include "ui/mainwindow.h"
#include "ui_mainwindow.h"

#include "tfr/cwt.h"
#include "sawe/project.h"

#include <QErrorMessage>

namespace Tools
{

RecordView::
        RecordView(RecordModel* model)
            :
            enabled(false),
            model_(model),
            prev_limit_(0)
{
    float l = model->project->worker.source()->length();
    prev_limit_ = l;
}


RecordView::
        ~RecordView()
{

}


void RecordView::
        prePaint()
{
    if (enabled)
    {
        float fs = model_->project->worker.source()->sample_rate();
        Tfr::Cwt& cwt = Tfr::Cwt::Singleton();
        float time_support = cwt.wavelet_time_support();
        cwt.wavelet_fast_time_support( cwt.wavelet_default_time_support() );
        double limit = std::max(0.f, model_->recording->time() - 2*cwt.wavelet_time_support_samples(fs)/fs);
        cwt.wavelet_fast_time_support( time_support );
        limit = std::min(limit, (double)model_->project->worker.length());

        if (model_->render_view->model->_qx >= prev_limit_) {
            // -- Following Record Marker --
            // Snap just before end so that project->worker.center starts working on
            // data that has been fetched. If center=length worker will start
            // at the very end and have to assume that the signal is abruptly
            // set to zero after the end. This abrupt change creates a false
            // dirac peek in the transform (false because it will soon be
            // invalid by newly recorded data).
            model_->render_view->model->_qx = std::max(model_->render_view->model->_qx, limit);

            if ( model_->recording->time_since_last_update() > 5 )
            {
                QErrorMessage::qtHandler()->showMessage(
                    "It looks like your recording device doesn't report any "
                    "data, so the recording has been stopped.\n"
                    "Restarting the recording might help temporarily. "
                    "You can also try another recording device by the command "
                    "line argument '--record-device=\"number\"'. "
                    "Available devices are listed in 'sonicawe.log'.", "No data from recording device");
                Ui::MainWindow* ui = model_->project->mainWindow()->getItems();
                ui->actionRecord->setChecked(false);
            }
        }
        prev_limit_ = limit;

        model_->render_view->userinput_update();
        model_->project->setModified();
    }
}


} // namespace Tools
