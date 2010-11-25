#include "recordcontroller.h"

#include "recordmodel.h"
#include "renderview.h"
#include "ui_mainwindow.h"

#include "sawe/project.h"
#include "ui/mainwindow.h"
#include "adapters/microphonerecorder.h"
#include "support/sinksignalproxy.h"
#include "tfr/cwt.h"

#include <TaskTimer.h>
#include <demangle.h>
#include <Statistics.h>

namespace Tools
{

RecordController::
        RecordController( RecordView* view, RenderView* render_view )
            :   view_ ( view ),
                render_view_ ( render_view )
{
    setupGui();
}


RecordController::
        ~RecordController()
{
    stopRecording();
}


void RecordController::
        stopRecording()
{
    receiveRecord(false);
}

void RecordController::
        receiveRecord(bool active)
{
    Adapters::MicrophoneRecorder* r = model()->recording;
    if (active)
    {
        Support::SinkSignalProxy* proxy;
        Signal::pOperation proxy_operation( proxy = new Support::SinkSignalProxy() );

        std::vector<Signal::pOperation> record_sinks;
        record_sinks.push_back( proxy_operation );
        r->getPostSink()->sinks( record_sinks );

        connect(proxy,
                SIGNAL(recievedInvalidSamples( Signal::Intervals )),
                SLOT(recievedInvalidSamples( Signal::Intervals )) );

        r->startRecording();
    }
    else
    {
        if (!r->isStopped())
            r->stopRecording();
    }
}


void RecordController::
        recievedInvalidSamples( Signal::Intervals I )
{
    TaskTimer tt("RecordController::recievedBuffer( %s )", I.toString().c_str());

    float fs = model()->project->head_source()->sample_rate();
    Signal::IntervalType s = Tfr::Cwt::Singleton().wavelet_time_support_samples( fs );

    Signal::Intervals v = ((I << s) | (I >> s)).coveredInterval();

    model()->project->worker.postSink()->invalidate_samples( v );

    // TODO invalidate collection samples through worker
    render_view_->model->collection->invalidate_samples( v );

    render_view_->userinput_update();
}


void RecordController::
        setupGui()
{
    Ui::MainWindow* ui = model()->project->mainWindow()->getItems();

    connect(ui->actionRecord, SIGNAL(triggered(bool)), SLOT(receiveRecord(bool)));

    //connect(render_view_, SIGNAL(destroying()), SLOT(close()));
    connect(render_view_, SIGNAL(destroying()), SLOT(stopRecording()));

    if (dynamic_cast<Adapters::MicrophoneRecorder*>(model()->project->head_source()->root()))
    {
        ui->actionRecord->setEnabled( true );
    }
}


} // namespace Tools
