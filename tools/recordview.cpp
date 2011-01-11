#include "recordview.h"

#include "recordmodel.h"
#include "renderview.h"
#include "adapters/microphonerecorder.h"

#include "tfr/cwt.h"
#include "sawe/project.h"

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
        model_->project->worker.requested_fps( 60 );
        double limit = std::max(0.f, model_->recording->time() - 2*Tfr::Cwt::Singleton().wavelet_time_support_samples(fs)/fs);

        if (model_->render_view->model->_qx >= prev_limit_) {
            // -- Following Record Marker --
            // Snap just before end so that project->worker.center starts working on
            // data that has been fetched. If center=length worker will start
            // at the very end and have to assume that the signal is abruptly
            // set to zero after the end. This abrupt change creates a false
            // dirac peek in the transform (false because it will soon be
            // invalid by newly recorded data).
            model_->render_view->model->_qx = std::max(model_->render_view->model->_qx, limit);
        }
        prev_limit_ = limit;

        model_->render_view->userinput_update();
    }
}


} // namespace Tools
