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

    connect(ui->lineEdit, SIGNAL(textEdited(QString)), SLOT(binResolutionChanged()));

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
        ui->label->setVisible(false);
        ui->lineEdit->setVisible(false);
    }
    else if (stft)
    {
        addRow("Type", "Short time fourier");
        if (renderview->model->renderSignalTarget->post_sink()->filter())
            addRow("Filter", vartype(*renderview->model->renderSignalTarget->post_sink()->filter()).c_str());
        addRow("Window type", "Regular");
        addRow("Window size", QString("%1").arg(stft->chunk_size()));
        addRow("Overlap", "0");
        addRow("Max hz", QString("%1").arg(fs/2));
        addRow("Min hz", QString("%1").arg(0));
        //addRow("Hz/bin", QString("%1").arg(fs/stft->chunk_size()));
        addRow("Rendered height", QString("%1 px").arg(renderview->height()));
        addRow("Rendered width", QString("%1 px").arg(renderview->width()));
        addRow("Amplification factor", QString("%1").arg(renderview->model->renderer->y_scale));
        ui->label->setVisible(true);
        ui->lineEdit->setVisible(true);
        ui->label->setText("Hz/bin");
        QString lineEditText = QString("%1").arg(fs/stft->chunk_size(),0,'f',2);
        if (ui->lineEdit->text() != lineEditText)
            ui->lineEdit->setText(lineEditText);
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
        ui->label->setVisible(false);
        ui->lineEdit->setVisible(false);
    }
    else
    {
        addRow("Type", "Unknown");
        addRow("Error", "Doesn't recognize transform");
        ui->label->setVisible(false);
        ui->lineEdit->setVisible(false);
    }

    size_t free=0, total=0;
    cudaMemGetInfo(&free, &total);
    addRow("Free GPU mem", GpuCpuVoidData::getMemorySizeText( free ).c_str());
    addRow("Total GPU mem", GpuCpuVoidData::getMemorySizeText( total ).c_str());

    size_t cacheByteSize=0;
    foreach( boost::shared_ptr<Heightmap::Collection> h, renderview->model->collections)
    {
        cacheByteSize += h->cacheByteSize();
    }
    addRow("Sonic AWE caches", GpuCpuVoidData::getMemorySizeText( cacheByteSize ).c_str());

    if (project->areToolsInitialized())
    {
        Signal::Intervals I = project->worker.todo_list();

        if (I.count())
        {
            addRow("Invalid heightmap", QString("%1 s").arg(I.count()/fs, 0, 'f', 1));
            timer.start();
        }
    }
}


void TransformInfoForm::
        binResolutionChanged()
{
    float fs = project->head->head_source()->sample_rate();
    float newValue = ui->lineEdit->text().toFloat();
    if (newValue<0.1)
        newValue=0.1;
    if (newValue>fs/4)
        newValue=fs/4;

    Tfr::Filter* f = renderview->model->block_filter();
    Tfr::Stft* stft = dynamic_cast<Tfr::Stft*>(!f?0:f->transform().get());
    if (0==stft)
        return;

    unsigned new_chunk_size = fs/newValue;
    if (new_chunk_size != stft->chunk_size())
    {
        project->head->head_source()->invalidate_samples(Signal::Intervals::Intervals_ALL);
        stft->set_exact_chunk_size( new_chunk_size );
        transformChanged();
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
