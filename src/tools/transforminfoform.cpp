#include "transforminfoform.h"
#include "ui_transforminfoform.h"
#include "renderview.h"

#include "ui/mainwindow.h"
#include "ui_mainwindow.h"

#include "heightmap/collection.h"
#include "heightmap/renderer.h"
#include "tfr/cwt.h"
#include "tfr/stft.h"
#include "tfr/cepstrum.h"
#include "tfr/drawnwaveform.h"
#include "adapters/csvtimeseries.h"
#include "filters/normalize.h"
#include "filters/normalizespectra.h"
#include "widgets/valueslider.h"
#include "tools/support/chaininfo.h"

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
    dock = new QDockWidget("Transform", MainWindow);
    dock->setObjectName(QString::fromUtf8("dockWidgetTransformInfoForm"));
    dock->setWidget(this);

    MainWindow->addDockWidget(Qt::RightDockWidgetArea, dock);

    MainWindow->getItems ()->actionTransform_info->setVisible(false);
    MainWindow->getItems ()->menu_Windows->insertAction (0, dock->toggleViewAction ());
    dock->toggleViewAction ()->setText ("Transform");

    connect(renderview, SIGNAL(transformChanged()), SLOT(transformChanged()), Qt::QueuedConnection);
    connect(renderview, SIGNAL(finishedWorkSection()), SLOT(transformChanged()), Qt::QueuedConnection);

    timer.setSingleShot( true );
    timer.setInterval( 500 );
    connect(&timer, SIGNAL(timeout()), SLOT(transformChanged()), Qt::QueuedConnection);
    QTimer::singleShot (0,this,SLOT(hidedock()));

    for (int i=0;i<Tfr::StftDesc::WindowType_NumberOfWindowTypes; ++i)
    {
        ui->windowTypeComboBox->addItem(Tfr::StftDesc::windowTypeName((Tfr::StftDesc::WindowType)i).c_str(), i);
    }

    {   ui->timeNormalizationSlider->setOrientation( Qt::Horizontal );
        ui->timeNormalizationSlider->setRange (0.0, 100, Widgets::ValueSlider::Quadratic );
        ui->timeNormalizationSlider->setValue ( 0 );
        ui->timeNormalizationSlider->setDecimals (1);
        ui->timeNormalizationSlider->setToolTip( "Normalization along time axis" );
        ui->timeNormalizationSlider->setSliderSize ( 300 );
        ui->timeNormalizationSlider->setUnit ("s");

        connect(ui->timeNormalizationSlider, SIGNAL(valueChanged(qreal)), SLOT(timeNormalizationChanged(qreal)));

        ui->timeNormalizationSlider->setVisible (false);
        ui->normalizationLabel->setVisible (false);
    }

    {   ui->freqNormalizationSlider->setOrientation( Qt::Horizontal );
        ui->freqNormalizationSlider->setRange (0.0, 1500, Widgets::ValueSlider::Quadratic );
        ui->freqNormalizationSlider->setValue ( 0 );
        ui->freqNormalizationSlider->setDecimals (1);
        ui->freqNormalizationSlider->setToolTip( "Normalization width along frequency axis in Hz" );
        ui->freqNormalizationSlider->setSliderSize ( 300 );
        ui->freqNormalizationSlider->setUnit ("Hz");

        connect(ui->freqNormalizationSlider, SIGNAL(valueChanged(qreal)), SLOT(freqNormalizationChanged(qreal)));
    }

    {   ui->freqNormalizationSliderPercent->setOrientation( Qt::Horizontal );
        ui->freqNormalizationSliderPercent->setRange (0.0, 100.0, Widgets::ValueSlider::Linear );
        ui->freqNormalizationSliderPercent->setValue ( 0 );
        ui->freqNormalizationSliderPercent->setDecimals (1);
        ui->freqNormalizationSliderPercent->setToolTip( "Normalization width along frequency axis in fraction of octave" );
        ui->freqNormalizationSliderPercent->setSliderSize ( 300 );
        ui->freqNormalizationSliderPercent->setUnit ("%");

        connect(ui->freqNormalizationSliderPercent, SIGNAL(valueChanged(qreal)), SLOT(freqNormalizationPercentChanged(qreal)));
    }

    connect(ui->minHzEdit, SIGNAL(editingFinished()), SLOT(minHzChanged()));
    connect(ui->binResolutionEdit, SIGNAL(editingFinished()), SLOT(binResolutionChanged()));
    connect(ui->windowSizeEdit, SIGNAL(editingFinished()), SLOT(windowSizeChanged()));
    //connect(ui->sampleRateEdit, SIGNAL(editingFinished()), SLOT(sampleRateChanged()));
    connect(ui->overlapEdit, SIGNAL(editingFinished()), SLOT(overlapChanged()));
    connect(ui->averagingEdit, SIGNAL(editingFinished()), SLOT(averagingChanged()));
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

    if (renderview->model->collections().empty())
        return;

    Tfr::TransformDesc::ptr f = renderview->model->transform_desc();
    const Tfr::Cwt* cwt = dynamic_cast<const Tfr::Cwt*>(f.get ());
    const Tfr::StftDesc* stft = dynamic_cast<const Tfr::StftDesc*>(f.get ());
    const Tfr::CepstrumDesc* cepstrum = dynamic_cast<const Tfr::CepstrumDesc*>(f.get ());
    const Tfr::DrawnWaveform* waveform = dynamic_cast<const Tfr::DrawnWaveform*>(f.get ());
    if (cepstrum) stft = 0; // CepstrumParams inherits StftDesc

    Signal::OperationDesc::Extent x = project->extent ();
    float fs = x.sample_rate.get ();
    float L = x.interval.get ().count() / fs;

    addRow("Length", QString("%1").arg( Signal::SourceBase::lengthLongFormat ( L ).c_str ()));
    //bool adjustable_sample_rate = 0 != dynamic_cast<Adapters::CsvTimeseries*>(project->head->head_source()->root());
    bool adjustable_sample_rate = false;
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
    addRow("Number of samples", QString("%1").arg(x.interval.get ().count()));

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
    ui->freqNormalizationSlider->setVisible(stft);
    ui->freqNormalizationSliderPercent->setVisible(stft);
#ifdef TARGET_hast
    }
#endif

    if (cwt)
    {
        addRow("Type", "Gabor wavelet");

        addRow("T/F resolution", QString("%1").arg(cwt->tf_resolution()));
        addRow("Time support", QString("%1").arg(cwt->wavelet_time_support_samples()/fs));
        addRow("Scales", QString("%1").arg(cwt->nScales()));
        addRow("Scales per octave", QString("%1").arg(cwt->scales_per_octave()));
        addRow("Sigma", QString("%1").arg(cwt->sigma()));
        addRow("Bins", QString("%1").arg(cwt->nBins()));
        addRow("Max hz", QString("%1").arg(cwt->get_max_hz(fs)));
        addRow("Actual min hz", QString("%1").arg(cwt->get_min_hz(fs)));
        addRow("Amplification factor", QString("%1").arg(renderview->model->renderer->render_settings.y_scale));
        setEditText( ui->minHzEdit, QString("%1").arg(cwt->get_wanted_min_hz(fs)) );
        //setEditText( ui->maxHzEdit, QString("%1").arg(cwt->get_max_hz(fs)) );
    }
    else if (stft)
    {
        addRow("Type", "Short time fourier");

        addRow("Max hz", QString("%1").arg(fs/2));
        addRow("Min hz", QString("%1").arg(0));
        //addRow("Hz/bin", QString("%1").arg(fs/stft->chunk_size()));
        addRow("Rendered height", QString("%1 px").arg(renderview->height()));
        addRow("Rendered width", QString("%1 px").arg(renderview->width()));
        addRow("Amplification factor", QString("%1").arg(renderview->model->renderer->render_settings.y_scale));
        setEditText( ui->binResolutionEdit, QString("%1").arg(fs/stft->chunk_size(),0,'f',2) );
        setEditText( ui->windowSizeEdit, QString("%1").arg(stft->chunk_size()) );
        setEditText( ui->overlapEdit, QString("%1").arg(stft->overlap()) );
        setEditText( ui->averagingEdit, QString("%1").arg(stft->averaging()) );
        Tfr::StftDesc::WindowType windowtype = stft->windowType();
        if (windowtype != ui->windowTypeComboBox->itemData(ui->windowTypeComboBox->currentIndex()).toInt() && !ui->windowTypeComboBox->hasFocus())
            ui->windowTypeComboBox->setCurrentIndex(ui->windowTypeComboBox->findData((int)windowtype));
    }
    else if (cepstrum)
    {
        addRow("Type", "Cepstrum");

        addRow("Window size", QString("%1").arg(cepstrum->chunk_size()));
        addRow("Amplification factor", QString("%1").arg(renderview->model->renderer->render_settings.y_scale));
        addRow("Lowest fundamental", QString("%1").arg( 2*fs / cepstrum->chunk_size()));

        setEditText( ui->binResolutionEdit, QString("%1").arg(fs/cepstrum->chunk_size(),0,'f',2) );
        setEditText( ui->windowSizeEdit, QString("%1").arg(cepstrum->chunk_size()) );
        setEditText( ui->overlapEdit, QString("%1").arg(cepstrum->overlap()) );
        Tfr::StftDesc::WindowType windowtype = cepstrum->windowType();
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
    foreach( const Heightmap::Collection::ptr& h, renderview->model->collections())
    {
        cacheByteSize += h.write ()->cacheByteSize();
    }

    addRow("Sonic AWE caches", DataStorageVoid::getMemorySizeText( cacheByteSize ).c_str());

    EXCEPTION_ASSERT(project->areToolsInitialized());

//    Tools::Support::ChainInfo ci(project->processing_chain ());
//    if (ci.hasWork ())
//    {
//        addRow("Invalid heightmap", QString("%1 s").arg(ci.out_of_date_sum()/fs, 0, 'f', 1));
//        timer.start();
//    }
}


void TransformInfoForm::
        minHzChanged()
{
    Signal::OperationDesc::Extent x = project->extent ();
    float fs = x.sample_rate.get ();
    float newValue = ui->minHzEdit->text().toFloat();
    if (newValue<fs/100000)
        newValue=fs/100000;
    if (newValue>fs/2)
        newValue=fs/2;

    {
        auto td = renderview->model->transform_descs ().write ();
        Tfr::Cwt& cwt = td->getParam<Tfr::Cwt>();

        if (cwt.get_wanted_min_hz (fs) == newValue)
            return;

        cwt.set_wanted_min_hz(newValue, fs);
    }

    deprecateAll ();
}


void TransformInfoForm::
        binResolutionChanged()
{
    float fs = project->extent().sample_rate.get ();
    float newValue = ui->binResolutionEdit->text().toFloat();
    if (newValue<0.1)
        newValue=0.1;
    if (newValue>fs/4)
        newValue=fs/4;
    Signal::IntervalType new_chunk_size = fs/newValue;

    {
        auto td = renderview->model->transform_descs ().write ();
        Tfr::StftDesc& stft = td->getParam<Tfr::StftDesc>();

        if (new_chunk_size == stft.chunk_size())
            return;

        stft.set_exact_chunk_size( new_chunk_size );
    }

    deprecateAll ();
}


void TransformInfoForm::
        windowSizeChanged()
{
    int newValue = ui->windowSizeEdit->text().toInt();
    Signal::IntervalType N = project->extent().interval.get ().count();
    if (newValue<1)
        newValue=1;
    if ((unsigned)newValue>N*2)
        newValue=N*2;

    {
        auto td = renderview->model->transform_descs ().write ();
        Tfr::StftDesc& stft = td->getParam<Tfr::StftDesc>();

        if (newValue == stft.chunk_size())
            return;

        stft.set_approximate_chunk_size( newValue );
    }

    renderview->emitTransformChanged();
}


//void TransformInfoForm::
//        sampleRateChanged()
//{
//    csv_source = ...

//    float newValue = ui->sampleRateEdit->text().toFloat();
//    if (newValue<0.01)
//        newValue=0.01;
//    float minHz = ui->minHzEdit->text().toFloat();
//    float orgMinHz = minHz;
//    if (minHz<newValue/100000)
//        minHz=newValue/100000;
//    if (minHz>newValue/2)
//        minHz=newValue/2;
//    if (orgMinHz != minHz)
//        renderview->model->transform_descs (.write ())->getParam<Tfr::Cwt>().set_wanted_min_hz(minHz);

//    Signal::BufferSource* bs = dynamic_cast<Signal::BufferSource*>(project->head->head_source()->root());
//    if (bs && (bs->sample_rate() != newValue || orgMinHz != minHz))
//    {
//        bs->set_sample_rate( newValue );

//        // TODO invalidate samples
//    }
//}


void TransformInfoForm::
        windowTypeChanged()
{
    int windowtype = ui->windowTypeComboBox->itemData(ui->windowTypeComboBox->currentIndex()).toInt();

    {
        auto td = renderview->model->transform_descs ().write ();
        Tfr::StftDesc& stft = td->getParam<Tfr::StftDesc>();
        if (stft.windowType() == windowtype)
            return;

        float overlap = stft.overlap();
        if (stft.windowType() == Tfr::StftDesc::WindowType_Rectangular && overlap == 0.f)
            overlap = 0.5f;

        stft.setWindow( (Tfr::StftDesc::WindowType)windowtype, overlap );
    }

    deprecateAll();
}


void TransformInfoForm::
        overlapChanged()
{
    float newValue = ui->overlapEdit->text().toFloat();

    // Tfr::Stft::setWindow validates value range

    {
        auto td = renderview->model->transform_descs ().write ();
        Tfr::StftDesc& stft = td->getParam<Tfr::StftDesc>();
        if (stft.overlap() == newValue)
            return;

        Tfr::StftDesc::WindowType windowtype = stft.windowType();
        stft.setWindow( windowtype, newValue );
    }

    deprecateAll();
}


void TransformInfoForm::
        averagingChanged()
{
    float newValue = ui->averagingEdit->text().toFloat();

    {
        auto td = renderview->model->transform_descs ().write ();
        Tfr::StftDesc& stft = td->getParam<Tfr::StftDesc>();
        if (stft.averaging() == newValue)
            return;

        stft.averaging( newValue );
    }

    deprecateAll();
}


void TransformInfoForm::
        timeNormalizationChanged(qreal /*newValue*/)
{
    EXCEPTION_ASSERTX(false, "Use Signal::Processing namespace");
/*
    Signal::PostSink* ps = project->tools ().render_model.renderSignalTarget->post_sink ();
    float fs = ps->sample_rate ();
    if (0.f < newValue)
    {
        Signal::pOperation timeNormalization(
                        new Filters::Normalize(newValue*fs));
        ps->filter ( timeNormalization );
    }
    else
        ps->filter( Signal::pOperation() );
*/
}


void TransformInfoForm::
        freqNormalizationChanged(qreal newValue)
{
    Heightmap::TfrMappings::StftBlockFilterParams::ptr stft_params =
            project->tools ().render_model.get_stft_block_filter_params ();
    EXCEPTION_ASSERT( stft_params );

    if (0.f < newValue)
    {
        ui->freqNormalizationSliderPercent->setValue ( 0 );
        stft_params.write ()->freq_normalization = Tfr::pChunkFilter(
                    new Filters::NormalizeSpectra(newValue));
        //project->tools ().render_model.amplitude_axis (Heightmap::AmplitudeAxis_Real);
    }
    else
    {
        stft_params.write ()->freq_normalization.reset();
        //project->tools ().render_model.amplitude_axis (Heightmap::AmplitudeAxis_Linear);
    }

    deprecateAll();
}


void TransformInfoForm::
        freqNormalizationPercentChanged(qreal newValue)
{
    Heightmap::TfrMappings::StftBlockFilterParams::ptr stft_params =
            project->tools ().render_model.get_stft_block_filter_params ();
    EXCEPTION_ASSERT( stft_params );

    if (0.f < newValue)
    {
        ui->freqNormalizationSlider->setValue ( 0 );
        TaskInfo("new stuff: %f", -newValue/100.0f);

        stft_params.write ()->freq_normalization = Tfr::pChunkFilter(
                    new Filters::NormalizeSpectra(-newValue/100.0f));
        //project->tools ().render_model.amplitude_axis (Heightmap::AmplitudeAxis_Real);
    }
    else
    {
        stft_params.write ()->freq_normalization.reset();
        //project->tools ().render_model.amplitude_axis (Heightmap::AmplitudeAxis_Linear);
    }

    deprecateAll();
}


void TransformInfoForm::
        hidedock()
{
    dock->close ();
}


void TransformInfoForm::
        deprecateAll()
{
    renderview->model->target_marker()->target_needs ().write ()->deprecateCache( Signal::Intervals::Intervals_ALL );
    renderview->emitTransformChanged();
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
