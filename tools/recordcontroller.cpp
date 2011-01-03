#include "recordcontroller.h"

#include "adapters/microphonerecorder.h"
#include "signal/worker.h"
#include "ui_mainwindow.h"
#include "support/sinksignalproxy.h"

namespace Tools
{

RecordController::
        RecordController( Signal::Worker* worker, Ui::MainWindow* actions )
            :
            _worker(worker)
{
    setupGui( actions );
}


RecordController::
        ~RecordController()
{
    Adapters::MicrophoneRecorder* r =
            dynamic_cast<Adapters::MicrophoneRecorder*>(_record_model.get() );

    if (r && !r->isStopped())
        r->stopRecording();
}

void RecordController::
        receiveRecord(bool active)
{
    if (!active)
    {
        Adapters::MicrophoneRecorder* r =
                dynamic_cast<Adapters::MicrophoneRecorder*>(_record_model.get() );

        if (r)
        {
            if (!r->isStopped())
                r->stopRecording();
            _record_model.reset();
        }
    }
    else
    {
        Adapters::MicrophoneRecorder* r;
        _record_model.reset( r = new Adapters::MicrophoneRecorder( 0 ) );
        r->startRecording();
        _record_model->source( _worker->source() );
        _worker->source( _record_model );

        Support::SinkSignalProxy* proxy;
        Signal::pOperation proxy_operation( proxy = new Support::SinkSignalProxy());

        std::vector<Signal::pOperation> record_sinks;
        record_sinks.push_back( proxy_operation );
        r->getPostSink()->sinks( record_sinks );

        connect(proxy, SIGNAL(recievedBuffer(Signal::pBuffer)), SLOT(recievedBuffer(Signal::pBuffer)) );
    }
}


void RecordController::
        recievedBuffer(Signal::pBuffer b)
{
    _worker->postSink()->invalidate_samples( b->getInterval() );
}


void RecordController::
        setupGui( Ui::MainWindow* ui )
{
    connect(ui->actionRecord, SIGNAL(triggered(bool)), SLOT(receiveRecord(bool)));
}


} // namespace Tools
