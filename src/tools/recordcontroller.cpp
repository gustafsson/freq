#include "recordcontroller.h"

#include "recordmodel.h"
#include "renderview.h"
#include "ui_mainwindow.h"

#include "sawe/project.h"
#include "ui/mainwindow.h"
#include "adapters/recorder.h"
#include "support/sinksignalproxy.h"
#include "tfr/cwt.h"
#include "heightmap/collection.h"
#include "tools/commands/recordedcommand.h"

#include "TaskTimer.h"
#include "demangle.h"
#include "Statistics.h"

namespace Tools
{

RecordController::
        RecordController( RecordView* view )
            :   view_ ( view ),
                destroyed_ ( false ),
                prev_length_( 0 )
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
    Adapters::Recorder* r = model()->recording;
    if (active)
    {
/*
//If the recording is stopped because of an error, the button in the user interface should notice somehow ...
        Support::SinkSignalProxy* proxy;
        Signal::pOperation proxy_operation( proxy = new Support::SinkSignalProxy() );

        std::vector<Signal::pOperation> record_sinks;
        record_sinks.push_back( proxy_operation );
        r->getPostSink()->sinks( record_sinks );

        connect(proxy,
                SIGNAL(recievedInvalidSamples( Signal::Intervals )),
                SLOT(recievedInvalidSamples( Signal::Intervals )),
                Qt::QueuedConnection );
*/
        prev_length_ = r->number_of_samples();
        r->startRecording();

        if (!r->canRecord())
            model()->project->mainWindow()->getItems()->actionRecord->setChecked( false );
    }
    else
    {
        if (!r->isStopped())
        {
            r->stopRecording();

            if (model()->recording->number_of_samples() > prev_length_)
            {
// TODO this command really should be invoked when the recording is started.
                Tools::Commands::pCommand cmd( new Tools::Commands::RecordedCommand( model()->recording, prev_length_, model()->render_view->model ));
                model()->project->commandInvoker()->invokeCommand(  cmd );
            }
        }
    }

    view_->enabled = active;
}


void RecordController::
        receiveStop()
{
    Ui::MainWindow* ui = model()->project->mainWindow()->getItems();
    ui->actionRecord->setChecked(false);
}


void RecordController::
        recievedInvalidSamples( Signal::Intervals I )
{
    if ( destroyed_ )
        return;

    model()->recording->invalidate_samples( I );

    if (model()->recording->isStopped())
        receiveStop();
}


void RecordController::
        setupGui()
{
    Ui::MainWindow* ui = model()->project->mainWindow()->getItems();

    connect(ui->actionRecord, SIGNAL(toggled(bool)), SLOT(receiveRecord(bool)));
    connect(ui->actionStopPlayBack, SIGNAL(triggered()), SLOT(receiveStop()));

    connect(model()->render_view, SIGNAL(destroying()), SLOT(destroying()));
    connect(model()->render_view, SIGNAL(prePaint()), view_, SLOT(prePaint()));

//    Adapters::Recorder* r = dynamic_cast<Adapters::Recorder*>(model()->project->head->head_source()->root());
    Adapters::Recorder* r = model()->recording;
    if (r)
    {
        ui->actionRecord->setVisible (true);
        if (r->canRecord())
            ui->actionRecord->setEnabled( true );
        else
            ui->actionRecord->setToolTip("Can't record, no record devices found");

        #if defined(TARGET_hast)
            ui->actionSave_project->setVisible( false );
            ui->actionSave_project_as->setVisible( false );
            ui->actionExport_audio->setVisible( false );
            ui->actionExport_selection->setVisible( false );
        #endif
    }
    else
        ui->actionRecord->setToolTip("Can only record on recordings");
}


} // namespace Tools
