#include "recordcontroller.h"

#include "recordmodel.h"
#include "renderview.h"
#include "ui_mainwindow.h"

#include "sawe/project.h"
#include "ui/mainwindow.h"
#include "adapters/microphonerecorder.h"
#include "support/sinksignalproxy.h"

#include <TaskTimer.h>
#include <demangle.h>

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
                SIGNAL(recievedBuffer(Signal::Buffer*)),
                SLOT(recievedBuffer(Signal::Buffer*)), Qt::DirectConnection );

        r->startRecording();
    }
    else
    {
        if (!r->isStopped())
            r->stopRecording();
    }
}


void RecordController::
        recievedBuffer(Signal::Buffer* b)
{
    render_view_->userinput_update();
    model()->project->worker.postSink()->invalidate_samples( b->getInterval() );
}


void RecordController::
        setupGui()
{
    Ui::MainWindow* ui = model()->project->mainWindow()->getItems();

    connect(ui->actionRecord, SIGNAL(triggered(bool)), SLOT(receiveRecord(bool)));

    if (dynamic_cast<Adapters::MicrophoneRecorder*>(model()->project->head_source()->root()))
    {
        ui->actionRecord->setEnabled( true );
    }
}


} // namespace Tools
