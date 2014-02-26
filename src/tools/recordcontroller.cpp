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

#include "tasktimer.h"
#include "demangle.h"
#include "Statistics.h"

namespace Tools
{

RecordController::
        RecordController( RecordView* view, QAction* actionRecord )
            :   view_ ( view ),
                ui( new Actions() ),
                destroyed_ ( false ),
                prev_length_( 0 )
{
    qRegisterMetaType<Signal::Interval>("Signal::Interval");

    ui->actionRecord = actionRecord;

    setupGui();

    connect(view_, SIGNAL(gotNoData()), SLOT(receiveStop()));
    connect(model(), SIGNAL(markNewlyRecordedData(Signal::Interval)), SLOT(redraw(Signal::Interval)));
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

    Adapters::Recorder::WritePtr r(model()->recording);
    if (!r->isStopped())
        r->stopRecording();

    destroyed_ = true;
}

void RecordController::
        receiveRecord(bool active)
{
    Adapters::Recorder::WritePtr r(model()->recording);

    if (active)
    {
        prev_length_ = r->number_of_samples();
        r->startRecording();

        if (!r->canRecord()) {
            TaskInfo("can't record :(");
            ui->actionRecord->setChecked( false );
        }
    }
    else
    {
        if (!r->isStopped())
        {
            r->stopRecording();

            if (r->number_of_samples() > prev_length_)
            {
                r.unlock ();

                // TODO this command really should be invoked when the
                // recording is started rather than when it is finished. As it
                // would be less surprising to the user when looking at the undo
                // menu.
                Tools::Commands::pCommand cmd( new Tools::Commands::RecordedCommand(
                                                   model()->recording,
                                                   prev_length_,
                                                   model()->render_view->model,
                                                   model()->invalidator ));
                model()->project->commandInvoker()->invokeCommand(  cmd );
            }
        }
    }

    view_->setEnabled( active );
}


void RecordController::
        receiveStop()
{
    ui->actionRecord->setChecked(false);
}


void RecordController::
        redraw(Signal::Interval)
{
    model()->render_view->redraw ();
}


void RecordController::
        setupGui()
{
    connect(ui->actionRecord, SIGNAL(toggled(bool)), SLOT(receiveRecord(bool)));
//    connect(ui->actionStopPlayBack, SIGNAL(triggered()), SLOT(receiveStop()));

    connect(model()->render_view, SIGNAL(destroying()), SLOT(destroying()));
    connect(model()->render_view, SIGNAL(prePaint()), view_, SLOT(prePaint()));

    if (model()->recording)
    {
        ui->actionRecord->setVisible (true);
        if (write1(model()->recording) -> canRecord())
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
