#include "exportaudiodialog.h"
#include "ui_exportaudiodialog.h"

#include "ui_mainwindow.h"
#include "ui/mainwindow.h"

#include "renderview.h"

#include "sawe/project.h"
#include "adapters/writewav.h"

#include <QFileDialog>

namespace Tools {

ExportAudioDialog::ExportAudioDialog(
        Sawe::Project* project,
        SelectionModel* selection_model,
        RenderView* render_view
        )
            :
    QDialog(project->mainWindow()),
    ui(new Ui::ExportAudioDialog),
    project(project),
    selection_model(selection_model),
    render_view(render_view),
    total(0)
{
    ui->setupUi(this);

    update_timer.setSingleShot( true );
    update_timer.setInterval( 1000 );
    connect( &update_timer, SIGNAL( timeout() ), SLOT( update() ) );

    setupGui();
}


ExportAudioDialog::~ExportAudioDialog()
{
    delete ui;
}


void ExportAudioDialog::
        exportEntireFile()
{
    start(Signal::pOperation());
}


void ExportAudioDialog::
        exportSelection()
{
    if (0 == project->tools().selection_model.current_selection().get())
    {
        TaskInfo("ExportAudio::exportSelection without selection");
    }

    Signal::pOperation selection = project->tools().selection_model.current_selection_copy( SelectionModel::SaveInside_TRUE );

    start(selection);
}


void ExportAudioDialog::
        abortExport()
{
    exportTarget.reset();
}


void ExportAudioDialog::
        selectionChanged()
{
    bool has_selection = false;
    has_selection = 0 != selection_model->current_selection();

    project->mainWindow()->getItems()->actionExport_selection->setEnabled( has_selection );
}


void ExportAudioDialog::
        populateTodoList()
{
    if (!project->worker.fetch_todo_list().empty() || !exportTarget)
        return;

    project->worker.center = 0;
    project->worker.target(exportTarget);
}


void ExportAudioDialog::
        paintEvent ( QPaintEvent * event )
{
    QDialog::paintEvent(event);

    if (!exportTarget)
        return;

    Signal::PostSink* postsink = exportTarget->post_sink();
    Signal::IntervalType missing = postsink->invalid_samples().count();
    float finished = 1.f - missing/(double)total;

    unsigned percent = finished*100;
    ui->progressBar->setValue( percent );

    bool isFinished = 0 == missing;

    if (isFinished)
    {
        ui->buttonBoxAbort->setEnabled( !isFinished );
        ui->buttonBoxOk->setEnabled( isFinished );

        float L = total/postsink->sample_rate();
        ui->labelExporting->setText(QString(
                "Exported %1 to %2")
                    .arg( Signal::SourceBase::lengthLongFormat(L).c_str() )
                    .arg( filemame ));
        exportTarget.reset();
    }
    else
    {
        update_timer.start();
        render_view->userinput_update( false );
    }
}


void ExportAudioDialog::
        setupGui()
{
    connect(project->mainWindow()->getItems()->actionExport_audio, SIGNAL(triggered()), SLOT(exportEntireFile()));
    connect(project->mainWindow()->getItems()->actionExport_selection, SIGNAL(triggered()), SLOT(exportSelection()));
    connect(selection_model, SIGNAL(selectionChanged()), SLOT(selectionChanged()));
    connect(render_view, SIGNAL(populateTodoList()), SLOT(populateTodoList()));
    connect(this, SIGNAL(rejected()), SLOT(abortExport()));
    connect(ui->buttonBoxAbort, SIGNAL(rejected()), SLOT(abortExport()));
    connect(ui->buttonBoxOk, SIGNAL(accepted()), SLOT(close()));

    selectionChanged();
}


void ExportAudioDialog::
        start(Signal::pOperation filter)
{
    filemame = QFileDialog::getSaveFileName(project->mainWindow(), "Export audio", "", "Wav audio (*.wav)");
    if (0 == filemame.length()) {
        // User pressed cancel
        return;
    }

    QString extension = ".wav";
    if (filemame.length() < extension.length())
        filemame += extension;
    if (0 != QString::compare(filemame.mid(filemame.length() - extension.length()), extension, Qt::CaseInsensitive))
        filemame += extension;

    exportTarget.reset(new Signal::Target(&project->layers));
    Signal::PostSink* postsink = exportTarget->post_sink();

    postsink->filter( filter );
    std::vector<Signal::pOperation> sinks;
    sinks.push_back( Signal::pOperation( new Adapters::WriteWav( filemame.toStdString() )) );
    postsink->sinks(sinks);

    Signal::Intervals expected_data;
    if (filter)
        expected_data = ~filter->zeroed_samples_recursive();
    else
        expected_data = Signal::Intervals::Intervals_ALL;

    expected_data &= Signal::Interval(0, project->worker.source()->number_of_samples());
    postsink->invalidate_samples( expected_data );
    total = expected_data.count();

    bool isFinished = false;
    ui->buttonBoxAbort->setEnabled( !isFinished );
    ui->buttonBoxOk->setEnabled( isFinished );

    float L = total/postsink->sample_rate();
    ui->labelExporting->setText(QString("Exporting %1").arg( Signal::SourceBase::lengthLongFormat(L).c_str()));

    if (filter)
        setWindowTitle("Exporting selection");
    else
        setWindowTitle("Exporting entire signal");

    hide();
    setWindowModality( Qt::WindowModal );
    show();
}


} // namespace Tools
