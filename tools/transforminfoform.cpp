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
#include "tfr/drawnwaveform.h"
#include "adapters/csvtimeseries.h"

#ifdef USE_CUDA
#include <cuda_runtime.h>
#endif

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
    connect(&timer, SIGNAL(timeout()), SLOT(transformChanged()), Qt::QueuedConnection);

    connect(ui->minHzEdit, SIGNAL(textChanged(QString)), SLOT(minHzChanged()));
    //connect(ui->maxHzEdit, SIGNAL(textEdited(QString)), SLOT(maxHzChanged()));
    connect(ui->binResolutionEdit, SIGNAL(textEdited(QString)), SLOT(binResolutionChanged()));
    connect(ui->sampleRateEdit, SIGNAL(textEdited(QString)), SLOT(sampleRateChanged()));

    timer.start(); // call transformChanged once
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
    Tfr::DrawnWaveform* waveform = dynamic_cast<Tfr::DrawnWaveform*>(!f?0:f->transform().get());

    Signal::pOperation head = project->head->head_source();
    float fs = head->sample_rate();

    addRow("Length", QString("%1").arg(head->lengthLongFormat().c_str()));
    bool adjustable_sample_rate = 0 != dynamic_cast<Adapters::CsvTimeseries*>(project->head->head_source()->root());
    ui->sampleRateEdit->setVisible(adjustable_sample_rate);
    ui->sampleRateLabel->setVisible(adjustable_sample_rate);
    if (adjustable_sample_rate)
    {
        QString sampleRateText = QString("%1").arg(fs);
        if (ui->sampleRateEdit->text() != sampleRateText)
            ui->sampleRateEdit->setText(sampleRateText);
    }
    else
    {
        addRow("Sample rate", QString("%1").arg(fs));
    }
    addRow("Number of samples", QString("%1").arg(head->number_of_samples()));

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
        addRow("Bins", QString("%1").arg(cwt->nBins(fs)));
        addRow("Max hz", QString("%1").arg(cwt->get_max_hz(fs)));
        addRow("Actual min hz", QString("%1").arg(cwt->get_min_hz(fs)));
        addRow("Amplification factor", QString("%1").arg(renderview->model->renderer->y_scale));
        ui->minHzLabel->setVisible(true);
        ui->minHzEdit->setVisible(true);
        ui->maxHzLabel->setVisible(false);
        ui->maxHzEdit->setVisible(false);
        ui->binResolutionLabel->setVisible(false);
        ui->binResolutionEdit->setVisible(false);
        QString minHzText = QString("%1").arg(cwt->wanted_min_hz());
        if (ui->minHzEdit->text() != minHzText)
            ui->minHzEdit->setText(minHzText);
        //QString maxHzText = QString("%1").arg(cwt->get_max_hz(fs));
        //if (ui->maxHzEdit->text() != maxHzText)
        //    ui->maxHzEdit->setText(maxHzText);
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
        ui->minHzLabel->setVisible(false);
        ui->minHzEdit->setVisible(false);
        ui->maxHzLabel->setVisible(false);
        ui->maxHzEdit->setVisible(false);
        ui->binResolutionLabel->setVisible(true);
        ui->binResolutionEdit->setVisible(true);
        QString binResolutionText = QString("%1").arg(fs/stft->chunk_size(),0,'f',2);
        if (ui->binResolutionEdit->text() != binResolutionText)
            ui->binResolutionEdit->setText(binResolutionText);
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
        ui->minHzLabel->setVisible(false);
        ui->minHzEdit->setVisible(false);
        ui->maxHzLabel->setVisible(false);
        ui->maxHzEdit->setVisible(false);
        ui->binResolutionLabel->setVisible(false);
        ui->binResolutionEdit->setVisible(false);
    }
    else if (waveform)
    {
        addRow("Type", "Waveform");
        ui->minHzLabel->setVisible(false);
        ui->minHzEdit->setVisible(false);
        ui->maxHzLabel->setVisible(false);
        ui->maxHzEdit->setVisible(false);
        ui->binResolutionLabel->setVisible(false);
        ui->binResolutionEdit->setVisible(false);
    }
    else
    {
        addRow("Type", "Unknown");
        addRow("Error", "Doesn't recognize transform");
        ui->minHzLabel->setVisible(false);
        ui->minHzEdit->setVisible(false);
        ui->maxHzLabel->setVisible(false);
        ui->maxHzEdit->setVisible(false);
        ui->binResolutionLabel->setVisible(false);
        ui->binResolutionEdit->setVisible(false);
    }

#ifdef USE_CUDA
    size_t free=0, total=0;
    cudaMemGetInfo(&free, &total);
    addRow("Free GPU mem", DataStorageVoid::getMemorySizeText( free ).c_str());
    addRow("Total GPU mem", DataStorageVoid::getMemorySizeText( total ).c_str());
#endif

    size_t cacheByteSize=0;
    foreach( boost::shared_ptr<Heightmap::Collection> h, renderview->model->collections)
    {
        cacheByteSize += h->cacheByteSize();
    }
    addRow("Sonic AWE caches", DataStorageVoid::getMemorySizeText( cacheByteSize ).c_str());

    BOOST_ASSERT(project->areToolsInitialized());

    Signal::Intervals I = project->worker.todo_list();

    if (I.count())
    {
        addRow("Invalid heightmap", QString("%1 s").arg(I.count()/fs, 0, 'f', 1));
        timer.start();
    }
}


void TransformInfoForm::
        minHzChanged()
{
    float fs = project->head->head_source()->sample_rate();
    float newValue = ui->minHzEdit->text().toFloat();
    if (newValue<fs/100000)
        newValue=fs/100000;
    if (newValue>fs/2)
        newValue=fs/2;

    Tfr::Cwt* cwt = &Tfr::Cwt::Singleton();

    if (cwt->wanted_min_hz() != newValue)
    {
        project->head->head_source()->invalidate_samples(Signal::Intervals::Intervals_ALL);
        cwt->set_wanted_min_hz(newValue);
        renderview->emitTransformChanged();
    }
}


void TransformInfoForm::
        binResolutionChanged()
{
    float fs = project->head->head_source()->sample_rate();
    float newValue = ui->binResolutionEdit->text().toFloat();
    if (newValue<0.1)
        newValue=0.1;
    if (newValue>fs/4)
        newValue=fs/4;

    Tfr::Stft* stft = &Tfr::Stft::Singleton();

    unsigned new_chunk_size = fs/newValue;
    if (new_chunk_size != stft->chunk_size())
    {
        project->head->head_source()->invalidate_samples(Signal::Intervals::Intervals_ALL);
        stft->set_exact_chunk_size( new_chunk_size );
        renderview->emitTransformChanged();
    }
}


void TransformInfoForm::
        sampleRateChanged()
{
    float newValue = ui->sampleRateEdit->text().toFloat();
    if (newValue<0.01)
        newValue=0.01;
    float minHz = ui->minHzEdit->text().toFloat();
    float orgMinHz = minHz;
    if (minHz<newValue/100000)
        minHz=newValue/100000;
    if (minHz>newValue/2)
        minHz=newValue/2;
    if (orgMinHz != minHz)
        Tfr::Cwt::Singleton().set_wanted_min_hz(minHz);

    Signal::BufferSource* bs = dynamic_cast<Signal::BufferSource*>(project->head->head_source()->root());
    if (bs && (bs->sample_rate() != newValue || orgMinHz != minHz))
    {
        bs->set_sample_rate( newValue );

        project->head->head_source()->invalidate_samples(Signal::Intervals::Intervals_ALL);
        renderview->emitTransformChanged();
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
