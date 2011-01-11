#include "recordcontroller.h"

#include "recordmodel.h"
#include "renderview.h"
#include "ui_mainwindow.h"

#include "sawe/project.h"
#include "ui/mainwindow.h"
#include "adapters/microphonerecorder.h"
#include "support/sinksignalproxy.h"
#include "tfr/cwt.h"
#include "heightmap/collection.h"

#include <TaskTimer.h>
#include <demangle.h>
#include <Statistics.h>

namespace Tools
{

RecordController::
        RecordController( RecordView* view )
            :   view_ ( view ),
                destroyed_ ( false )
{
    setupGui();
}


RecordController::
        ~RecordController()
{
    TaskInfo("~RecordController");
    destroying();
}


void RecordController::
        destroying()
{
    TaskInfo("RecordController::destroying()");
    if (destroyed_)
        return;

    receiveRecord(false);
    destroyed_ = true;
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
                SLOT(recievedInvalidSamples( Signal::Intervals )),
                Qt::QueuedConnection );

        r->startRecording();
    }
    else
    {
        if (!r->isStopped())
            r->stopRecording();
    }

    view_->enabled = active;
}


void RecordController::
        recievedInvalidSamples( Signal::Intervals I )
{
    if ( destroyed_ )
        return;

    //TaskTimer tt("RecordController::recievedBuffer( %s )", I.toString().c_str());

    float fs = model()->project->head_source()->sample_rate();
    Signal::IntervalType s = Tfr::Cwt::Singleton().wavelet_time_support_samples( fs );

    Signal::Intervals v = ((I << s) | (I >> s)).coveredInterval();

    model()->project->worker.invalidate_post_sink( v );

    model()->render_view->userinput_update();
}


void RecordController::
        setupGui()
{
    Ui::MainWindow* ui = model()->project->mainWindow()->getItems();

    connect(ui->actionRecord, SIGNAL(triggered(bool)), SLOT(receiveRecord(bool)));

    connect(model()->render_view, SIGNAL(destroying()), SLOT(destroying()));
    connect(model()->render_view, SIGNAL(prePaint()), view_, SLOT(prePaint()));

    if (dynamic_cast<Adapters::MicrophoneRecorder*>(model()->project->head_source()->root()))
    {
        ui->actionRecord->setEnabled( true );
    }
}


} // namespace Tools
