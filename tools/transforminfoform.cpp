#include "transforminfoform.h"
#include "ui_transforminfoform.h"
#include "renderview.h"

#include "ui/mainwindow.h"
#include "ui_mainwindow.h"

#include "heightmap/collection.h"
#include "heightmap/blockfilter.h"
#include "heightmap/renderer.h"
#include "tfr/cwt.h"
#include "tfr/stft.h"
#include "tfr/cepstrum.h"

#include <QTimer>

#define LOG_TRANSFORM_INFO
//#define LOG_TRANSFORM_INFO if(0)

namespace Tools
{

TransformInfoForm::TransformInfoForm(Sawe::Project* project, RenderView* renderview) :
    ui(new Ui::TransformInfoForm),
    project(project),
    renderview(renderview)
{
    ui->setupUi(this);

    Ui::SaweMainWindow* MainWindow = project->mainWindow();
    dock = new QDockWidget(MainWindow);
    dock->setObjectName(QString::fromUtf8("dockWidgetTransformInfoForm"));
    dock->setMinimumSize(QSize(42, 79));
    dock->setMaximumSize(QSize(524287, 524287));
    dock->setContextMenuPolicy(Qt::NoContextMenu);
    //dock->setFeatures(QDockWidget::DockWidgetFeatureMask);
    dock->setFeatures(QDockWidget::AllDockWidgetFeatures);
    dock->setAllowedAreas(Qt::AllDockWidgetAreas);
    dock->setEnabled(true);
    dock->setWidget(this);
    dock->setWindowTitle("Transform info");
    dock->hide();

    MainWindow->addDockWidget(Qt::RightDockWidgetArea, dock);
    //MainWindow->tabifyDockWidget(MainWindow->getItems()->operationsWindow, dock);

    connect(MainWindow->getItems()->actionTransform_info, SIGNAL(toggled(bool)), dock, SLOT(setVisible(bool)));
    connect(MainWindow->getItems()->actionTransform_info, SIGNAL(triggered()), dock, SLOT(raise()));
    connect(dock, SIGNAL(visibilityChanged(bool)), SLOT(checkVisibility(bool)));
    MainWindow->getItems()->actionTransform_info->setChecked( false );
    dock->setVisible(false);

    connect(renderview, SIGNAL(transformChanged()), SLOT(transformChanged()), Qt::QueuedConnection);

    timer.setSingleShot( true );
    timer.setInterval( 500 );
    connect(&timer, SIGNAL(timeout()), SLOT(transformChanged()));

    transformChanged();
}



TransformInfoForm::~TransformInfoForm()
{
    delete ui;
}


void TransformInfoForm::
        checkVisibility(bool visible)
{
    Ui::SaweMainWindow* MainWindow = project->mainWindow();
    visible |= !MainWindow->tabifiedDockWidgets( dock ).empty();
    visible |= dock->isVisibleTo( dock->parentWidget() );
    MainWindow->getItems()->actionTransform_info->setChecked(visible);
}


void TransformInfoForm::
        transformChanged()
{
    LOG_TRANSFORM_INFO TaskInfo ti("TransformInfoForm::transformChanged()");
    ui->tableWidget->clear();
    ui->tableWidget->setRowCount(0);
    ui->tableWidget->setColumnCount(2);
    QStringList header;
    header.push_back("Name");
    header.push_back("Value");
    ui->tableWidget->setHorizontalHeaderLabels( header );
    ui->tableWidget->verticalHeader()->hide();

    QTableWidgetItem*prototype = new QTableWidgetItem;
    prototype->setFlags( prototype->flags() & ~Qt::ItemIsEditable);
    ui->tableWidget->setItemPrototype( prototype );

    if (renderview->model->collections.empty())
        return;

    Tfr::Filter* f = renderview->model->block_filter();
    Tfr::Cwt* cwt = dynamic_cast<Tfr::Cwt*>(!f?0:f->transform().get());
    Tfr::Stft* stft = dynamic_cast<Tfr::Stft*>(!f?0:f->transform().get());
    Tfr::Cepstrum* cepstrum = dynamic_cast<Tfr::Cepstrum*>(!f?0:f->transform().get());

    float fs = project->head->head_source()->sample_rate();

    addRow("Sample rate", QString("%1").arg(fs));

    if (cwt)
    {
        addRow("Type", "Gabor wavelet");
        if (renderview->model->renderSignalTarget->post_sink()->filter())
            addRow("Filter", vartype(*renderview->model->renderSignalTarget->post_sink()->filter()).c_str());
        addRow("T/F resolution", QString("%1").arg(cwt->tf_resolution()));
        addRow("Time support", QString("%1").arg(cwt->wavelet_time_support_samples( fs )/fs));
        addRow("Scales", QString("%1").arg(cwt->nScales(fs)));
        addRow("Scales per octave", QString("%1").arg(cwt->scales_per_octave()));
        addRow("Sigma", QString("%1").arg(cwt->sigma()));
        addRow("Bins", QString("%1").arg(cwt->find_bin( cwt->scales_per_octave())));
        addRow("Max hz", QString("%1").arg(cwt->get_max_hz(fs)));
        addRow("Min hz", QString("%1").arg(cwt->get_min_hz(fs)));
        addRow("Amplification factor", QString("%1").arg(renderview->model->renderer->y_scale));
    }
    else if (stft)
    {
        addRow("Type", "Short time fourier");
        if (renderview->model->renderSignalTarget->post_sink()->filter())
            addRow("Filter", vartype(*renderview->model->renderSignalTarget->post_sink()->filter()).c_str());
        addRow("Window type", "Regular");
        addRow("Window size", QString("%1").arg(stft->chunk_size()));
        addRow("Overlap", "0");
        addRow("Amplification factor", QString("%1").arg(renderview->model->renderer->y_scale));
    }
    else if (cepstrum)
    {
        addRow("Type", "Cepstrum");
        if (renderview->model->renderSignalTarget->post_sink()->filter())
            addRow("Filter", vartype(*renderview->model->renderSignalTarget->post_sink()->filter()).c_str());
        addRow("Window type", "Regular");
        addRow("Window size", QString("%1").arg(cepstrum->chunk_size()));
        addRow("Overlap", "0");
        addRow("Amplification factor", QString("%1").arg(renderview->model->renderer->y_scale));
        addRow("Lowest fundamental", QString("%1").arg( 2*fs / cepstrum->chunk_size()));
    }
    else
    {
        addRow("Type", "Unknown");
        addRow("Error", "Doesn't recognize transform");
    }

    if (project->areToolsInitialized())
    {
        Signal::Intervals I = project->worker.todo_list();

        if (I.count())
        {
            addRow("Invalid heightmap", QString("%1 s").arg(I.count()/fs));
            timer.start();
        }
    }
}


void TransformInfoForm::
        addRow(QString name, QString value)
{
    class QTableReadOnlyText: public QTableWidgetItem
    {
    public:
        QTableReadOnlyText(QString text): QTableWidgetItem(text)
        {
            setFlags( flags() & ~Qt::ItemIsEditable);
        }

    };

    ui->tableWidget->insertRow( ui->tableWidget->rowCount());
    ui->tableWidget->setItem(ui->tableWidget->rowCount()-1, 0, new QTableReadOnlyText (name));
    ui->tableWidget->setItem(ui->tableWidget->rowCount()-1, 1, new QTableReadOnlyText (value));

    LOG_TRANSFORM_INFO TaskInfo("%s = %s", name.toStdString().c_str(), value.toStdString().c_str() );
}

} // namespace Tools
