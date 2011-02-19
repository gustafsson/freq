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
        setupGui()
{
    Ui::MainWindow* ui = model()->project->mainWindow()->getItems();

    connect(ui->actionRecord, SIGNAL(toggled(bool)), SLOT(receiveRecord(bool)));

    connect(model()->render_view, SIGNAL(destroying()), SLOT(destroying()));
    connect(model()->render_view, SIGNAL(prePaint()), view_, SLOT(prePaint()));

    if (dynamic_cast<Adapters::MicrophoneRecorder*>(model()->project->head->head_source()->root()))
    {
        ui->actionRecord->setEnabled( true );
    }
}


} // namespace Tools
