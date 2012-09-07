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
#include "filters/normalize.h"

#ifdef USE_CUDA
#include <cuda_runtime.h>
#endif

#include <QTimer>

//#define LOG_TRANSFORM_INFO
#define LOG_TRANSFORM_INFO if(0)

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
    connect(renderview, SIGNAL(finishedWorkSection()), SLOT(transformChanged()), Qt::QueuedConnection);

    timer.setSingleShot( true );
    timer.setInterval( 500 );
    connect(&timer, SIGNAL(timeout()), SLOT(transformChanged()), Qt::QueuedConnection);

    for (int i=0;i<Tfr::StftParams::WindowType_NumberOfWindowTypes; ++i)
    {
        ui->windowTypeComboBox->addItem(Tfr::StftParams::windowTypeName((Tfr::StftParams::WindowType)i).c_str(), i);
    }

    ui->normalizationComboBox->addItem("Select normalization to apply", -1.f);
    ui->normalizationComboBox->addItem("0 s normalization", 0.f);
    ui->normalizationComboBox->addItem("1 s normalization", 1.f);
    ui->normalizationComboBox->addItem("10 s normalization", 10.f);
    ui->normalizationComboBox->addItem("100 s normalization", 100.f);

    connect(ui->minHzEdit, SIGNAL(editingFinished()), SLOT(minHzChanged()));
    connect(ui->binResolutionEdit, SIGNAL(editingFinished()), SLOT(binResolutionChanged()));
    connect(ui->windowSizeEdit, SIGNAL(editingFinished()), SLOT(windowSizeChanged()));
    connect(ui->sampleRateEdit, SIGNAL(editingFinished()), SLOT(sampleRateChanged()));
    connect(ui->overlapEdit, SIGNAL(editingFinished()), SLOT(overlapChanged()));
    connect(ui->averagingEdit, SIGNAL(editingFinished()), SLOT(averagingChanged()));
    connect(ui->normalizationComboBox, SIGNAL(currentIndexChanged(int)), SLOT(normalizationChanged()));
    connect(ui->windowTypeComboBox, SIGNAL(currentIndexChanged(int)), SLOT(windowTypeChanged()));
    //connect(ui->maxHzEdit, SIGNAL(textEdited(QString)), SLOT(maxHzChanged()));
    //connect(ui->binResolutionEdit, SIGNAL(textEdited(QString)), SLOT(binResolutionChanged()));
    //connect(ui->sampleRateEdit, SIGNAL(textEdited(QString)), SLOT(sampleRateChanged()));

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

    const Tfr::Filter* f = renderview->model->block_filter();
    const Tfr::Cwt* cwt = dynamic_cast<const Tfr::Cwt*>(!f?0:f->transform()->transformParams());
    const Tfr::StftParams* stft = dynamic_cast<const Tfr::StftParams*>(!f?0:f->transform()->transformParams());
    const Tfr::CepstrumParams* cepstrum = dynamic_cast<const Tfr::CepstrumParams*>(!f?0:f->transform()->transformParams());
    const Tfr::DrawnWaveform* waveform = dynamic_cast<const Tfr::DrawnWaveform*>(!f?0:f->transform()->transformParams());
    if (cepstrum) stft = 0; // CepstrumParams inherits StftParams

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

    if (-1.f != ui->normalizationComboBox->itemData(ui->normalizationComboBox->currentIndex()).toFloat() && !ui->normalizationComboBox->hasFocus())
        ui->normalizationComboBox->setCurrentIndex(ui->normalizationComboBox->findData(-1.f));

#ifdef TARGET_hast
    {
        bool cwt = false, stft = false, cepstrum = false;
#endif
    ui->minHzLabel->setVisible(cwt);
    ui->minHzEdit->setVisible(cwt);
    ui->maxHzLabel->setVisible(false);
    ui->maxHzEdit->setVisible(false);
    ui->binResolutionLabel->setVisible(stft);
    ui->binResolutionEdit->setVisible(stft);
    ui->averagingLabel->setVisible(stft);
    ui->averagingEdit->setVisible(stft);
    ui->windowSizeLabel->setVisible(stft || cepstrum);
    ui->windowSizeEdit->setVisible(stft || cepstrum);
    ui->windowTypeLabel->setVisible(stft || cepstrum);
    ui->windowTypeComboBox->setVisible(stft || cepstrum);
    ui->overlapLabel->setVisible(stft || cepstrum);
    ui->overlapEdit->setVisible(stft || cepstrum);
#ifdef TARGET_hast
    }
#endif

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
        setEditText( ui->minHzEdit, QString("%1").arg(cwt->wanted_min_hz()) );
        //setEditText( ui->maxHzEdit, QString("%1").arg(cwt->get_max_hz(fs)) );
    }
    else if (stft)
    {
        addRow("Type", "Short time fourier");
        if (renderview->model->renderSignalTarget->post_sink()->filter())
            addRow("Filter", vartype(*renderview->model->renderSignalTarget->post_sink()->filter()).c_str());
        addRow("Window type", "Regular");
        addRow("Overlap", "0");
        addRow("Max hz", QString("%1").arg(fs/2));
        addRow("Min hz", QString("%1").arg(0));
        //addRow("Hz/bin", QString("%1").arg(fs/stft->chunk_size()));
        addRow("Rendered height", QString("%1 px").arg(renderview->height()));
        addRow("Rendered width", QString("%1 px").arg(renderview->width()));
        addRow("Amplification factor", QString("%1").arg(renderview->model->renderer->y_scale));
        setEditText( ui->binResolutionEdit, QString("%1").arg(fs/stft->chunk_size(),0,'f',2) );
        setEditText( ui->windowSizeEdit, QString("%1").arg(stft->chunk_size()) );
        setEditText( ui->overlapEdit, QString("%1").arg(stft->overlap()) );
        setEditText( ui->averagingEdit, QString("%1").arg(stft->averaging()) );
        Tfr::StftParams::WindowType windowtype = stft->windowType();
        if (windowtype != ui->windowTypeComboBox->itemData(ui->windowTypeComboBox->currentIndex()).toInt() && !ui->windowTypeComboBox->hasFocus())
            ui->windowTypeComboBox->setCurrentIndex(ui->windowTypeComboBox->findData((int)windowtype));
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

        setEditText( ui->binResolutionEdit, QString("%1").arg(fs/cepstrum->chunk_size(),0,'f',2) );
        setEditText( ui->windowSizeEdit, QString("%1").arg(cepstrum->chunk_size()) );
        setEditText( ui->overlapEdit, QString("%1").arg(cepstrum->overlap()) );
        Tfr::StftParams::WindowType windowtype = cepstrum->windowType();
        if (windowtype != ui->windowTypeComboBox->itemData(ui->windowTypeComboBox->currentIndex()).toInt() && !ui->windowTypeComboBox->hasFocus())
            ui->windowTypeComboBox->setCurrentIndex(ui->windowTypeComboBox->findData((int)windowtype));
    }
    else if (waveform)
    {
        addRow("Type", "Waveform");
    }
    else
    {
        addRow("Type", "Unknown");
        addRow("Error", "Doesn't recognize transform");
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

    Tfr::Cwt* cwt = project->tools().render_model.getParam<Tfr::Cwt>();

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

    Tfr::StftParams* stft = renderview->model->getParam<Tfr::StftParams>();
    Signal::IntervalType new_chunk_size = fs/newValue;

    if (new_chunk_size != stft->chunk_size())
    {
        project->head->head_source()->invalidate_samples(Signal::Intervals::Intervals_ALL);
        stft->set_exact_chunk_size( new_chunk_size );
        renderview->emitTransformChanged();
    }
}


void TransformInfoForm::
        windowSizeChanged()
{
    int newValue = ui->windowSizeEdit->text().toInt();
    Signal::IntervalType N = project->head->head_source()->number_of_samples();
    if (newValue<1)
        newValue=1;
    if ((unsigned)newValue>N*2)
        newValue=N*2;

    Tfr::StftParams* stft = renderview->model->getParam<Tfr::StftParams>();

    if (newValue != stft->chunk_size())
    {
        project->head->head_source()->invalidate_samples(Signal::Intervals::Intervals_ALL);
        stft->set_approximate_chunk_size( newValue );
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
        project->tools().render_model.getParam<Tfr::Cwt>()->set_wanted_min_hz(minHz);

    Signal::BufferSource* bs = dynamic_cast<Signal::BufferSource*>(project->head->head_source()->root());
    if (bs && (bs->sample_rate() != newValue || orgMinHz != minHz))
    {
        bs->set_sample_rate( newValue );

        project->head->head_source()->invalidate_samples(Signal::Intervals::Intervals_ALL);
        renderview->emitTransformChanged();
    }
}


void TransformInfoForm::
        windowTypeChanged()
{
    int windowtype = ui->windowTypeComboBox->itemData(ui->windowTypeComboBox->currentIndex()).toInt();

    Tfr::StftParams* stft = renderview->model->getParam<Tfr::StftParams>();
    if (stft->windowType() != windowtype)
    {
        float overlap = stft->overlap();
        if (stft->windowType() == Tfr::StftParams::WindowType_Rectangular && overlap == 0.f)
            overlap = 0.5f;

        stft->setWindow( (Tfr::StftParams::WindowType)windowtype, overlap );

        renderview->model->renderSignalTarget->post_sink()->invalidate_samples( Signal::Intervals::Intervals_ALL );
        renderview->emitTransformChanged();
    }
}


void TransformInfoForm::
        overlapChanged()
{
    float newValue = ui->overlapEdit->text().toFloat();

    // Tfr::Stft::setWindow validates value range

    Tfr::StftParams* stft = renderview->model->getParam<Tfr::StftParams>();
    if (stft->overlap() != newValue)
    {
        Tfr::StftParams::WindowType windowtype = stft->windowType();
        stft->setWindow( windowtype, newValue );

        renderview->model->renderSignalTarget->post_sink()->invalidate_samples( Signal::Intervals::Intervals_ALL );
        renderview->emitTransformChanged();
    }
}


void TransformInfoForm::
        averagingChanged()
{
    float newValue = ui->averagingEdit->text().toFloat();

    Tfr::StftParams* stft = renderview->model->getParam<Tfr::StftParams>();
    if (stft->averaging() != newValue)
    {
        stft->averaging( newValue );

        renderview->model->renderSignalTarget->post_sink()->invalidate_samples( Signal::Intervals::Intervals_ALL );
        renderview->emitTransformChanged();
    }
}


void TransformInfoForm::
        normalizationChanged()
{
    float newValue = ui->normalizationComboBox->itemData(ui->normalizationComboBox->currentIndex()).toFloat();

    if (0.f <= newValue)
    {
        float fs = project->head->head_source()->sample_rate();
        project->appendOperation(Signal::pOperation(
                new Filters::Normalize(newValue*fs)));
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


void TransformInfoForm::
        setEditText(QLineEdit* edit, QString value)
{
    if (edit->text() != value && !edit->hasFocus())
        edit->setText(value);
}

} // namespace Tools
