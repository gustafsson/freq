#include "rendercontroller.h"

// tools
#include "support/sinksignalproxy.h"
#include "support/toolbar.h"

// ui
#include "ui/comboboxaction.h"
#include "ui_mainwindow.h" // Locate actions for binding
#include "ui/mainwindow.h"

// Sonic AWE, Setting different transforms for rendering
#include "filters/reassign.h"
#include "filters/ridge.h"
#include "heightmap/blockfilter.h"
#include "heightmap/renderer.h"
#include "signal/postsink.h"
#include "tfr/cwt.h"
#include "tfr/cepstrum.h"
#include "graphicsview.h"
#include "sawe/application.h"
#include "signal/buffersource.h"
#include "signal/worker.h"
#include "signal/reroutechannels.h"

// gpumisc
#include <demangle.h>

// Qt
#include <QSlider>
#include <QGraphicsView>
#include <QResizeEvent>
#include <QMetaClassInfo>
#include <QGLContext>
#include <QSettings>

// todo remove
#include "navigationcontroller.h"
#include <QTimer>

using namespace Ui;

#ifdef min
#undef min
#endif

namespace Tools
{

class BlockFilterSink: public Signal::Sink
{
public:
    BlockFilterSink
        (
            Signal::pOperation o,
            RenderModel* model,
            RenderView* view,
            RenderController* controller
        )
        :
            model_(model),
            view_(view),
            controller_(controller),
            prevSignal( o->getInterval() )
    {
        BOOST_ASSERT( o );
        Operation::source(o);
    }


    virtual void source(Signal::pOperation v) { Operation::source()->source(v); }
    virtual bool deleteMe() { return false; } // Never delete this sink

    virtual Signal::pBuffer read(const Signal::Interval& I) {
        Signal::pBuffer r = Signal::Operation::read( I );
        return r;
    }

    virtual void invalidate_samples(const Signal::Intervals& I)
    {
        validateSize();

        // If BlockFilter is a CwtFilter wavelet time support has already been included in I

        foreach(boost::shared_ptr<Heightmap::Collection> c, model_->collections)
        {
            c->invalidate_samples( I );
        }

        Operation::invalidate_samples( I );
    }


    virtual Signal::Intervals invalid_samples()
    {
        Signal::Intervals I;
        foreach ( boost::shared_ptr<Heightmap::Collection> c, model_->collections)
        {
            Signal::Intervals inv_coll = c->invalid_samples();
            I |= inv_coll;
        }

        return I;
    }


    void validateSize()
    {
        unsigned N = num_channels();
        if ( N != model_->collections.size())
        {
            N = num_channels();

            model_->collections.resize(N);
            for (unsigned c=0; c<N; ++c)
            {
                if (!model_->collections[c])
                    model_->collections[c].reset( new Heightmap::Collection(model_->renderSignalTarget->source()));
            }

            view_->emitTransformChanged();
        }

        foreach (boost::shared_ptr<Heightmap::Collection> c, model_->collections)
        {
            c->block_filter( Operation::source() );
        }

        Signal::Interval currentInterval = getInterval();
        if (prevSignal != currentInterval)
        {
            foreach (boost::shared_ptr<Heightmap::Collection> c, model_->collections)
            {
                c->discardOutside( currentInterval );
            }

            if (currentInterval.last < prevSignal.last)
            {
                if (view_->model->_qx > currentInterval.last/sample_rate())
                    view_->model->_qx = currentInterval.last/sample_rate();
            }
        }
        prevSignal = currentInterval;
    }


private:
    RenderModel* model_;
    RenderView* view_;
    RenderController* controller_;

    Signal::Interval prevSignal;
};


RenderController::
        RenderController( QPointer<RenderView> view )
            :
            view(view),
            toolbar_render(0),
            hz_scale(0),
            linearScale(0),
            logScale(0),
            cepstraScale(0),
            amplitude_scale(0),
            hzmarker(0),
            color(0),
            transform(0)
{
    setupGui();

    // Default values for rendermodel are set in rendermodel constructor

    {
        // Default values for rendercontroller
        Ui::SaweMainWindow* main = dynamic_cast<Ui::SaweMainWindow*>(model()->project()->mainWindow());
        Ui::MainWindow* ui = main->getItems();
#ifdef TARGET_sss
        tf_resolution->setValue( 10 );
        ui->actionToggleOrientation->setChecked(true);
        transform->actions().at(0)->trigger();
#else
        transform->actions().at(1)->trigger();
#endif
        ui->actionSet_colorscale->trigger();

#ifdef TARGET_reader
        ui->actionTransform_Stft->trigger();
        linearScale->trigger();
#else
        ui->actionTransform_Cwt->trigger();
        logScale->trigger();
#endif
        main->getItems()->actionToggleTransformToolBox->setChecked( true );
    }
}


RenderController::
        ~RenderController()
{
    if (QGLContext::currentContext())
        clearCachedHeightmap();
}


void RenderController::
        stateChanged()
{
    // Don't lock the UI, instead wait a moment before any change is made
    view->userinput_update();

    model()->project()->setModified();
}


void RenderController::
        receiveSetRainbowColors()
{
    model()->renderer->color_mode = Heightmap::Renderer::ColorMode_Rainbow;
    stateChanged();
}


void RenderController::
        receiveSetGrayscaleColors()
{
    model()->renderer->color_mode = Heightmap::Renderer::ColorMode_Grayscale;
    stateChanged();
}


void RenderController::
        receiveSetColorscaleColors()
{
    model()->renderer->color_mode = Heightmap::Renderer::ColorMode_FixedColor;
    stateChanged();
}


void RenderController::
        receiveToogleHeightlines(bool value)
{
    model()->renderer->draw_height_lines = value;
    stateChanged();
}


void RenderController::
        receiveToggleOrientation(bool value)
{
    model()->renderer->left_handed_axes = !value;

    view->graphicsview->setLayoutDirection( value
                                            ? QBoxLayout::RightToLeft
                                            : QBoxLayout::TopToBottom );

    stateChanged();
}


void RenderController::
        receiveTogglePiano(bool value)
{
    model()->renderer->draw_piano = value;

    Ui::SaweMainWindow* main = dynamic_cast<Ui::SaweMainWindow*>(model()->project()->mainWindow());
    Ui::MainWindow* ui = main->getItems();

    ui->actionToggle_cursor_marker->setEnabled( ui->actionToggle_t_grid->isChecked() || ui->actionToggle_hz_grid->isChecked() || ui->actionToggle_piano_grid->isChecked() );

    stateChanged();
}


void RenderController::
        receiveToggleHz(bool value)
{
    model()->renderer->draw_hz = value;

    Ui::SaweMainWindow* main = dynamic_cast<Ui::SaweMainWindow*>(model()->project()->mainWindow());
    Ui::MainWindow* ui = main->getItems();

    ui->actionToggle_cursor_marker->setEnabled( ui->actionToggle_t_grid->isChecked() || ui->actionToggle_hz_grid->isChecked() || ui->actionToggle_piano_grid->isChecked() );

    stateChanged();
}


void RenderController::
        receiveToggleTAxis(bool value)
{
    model()->renderer->draw_t = value;

    Ui::SaweMainWindow* main = dynamic_cast<Ui::SaweMainWindow*>(model()->project()->mainWindow());
    Ui::MainWindow* ui = main->getItems();

    ui->actionToggle_cursor_marker->setEnabled( ui->actionToggle_t_grid->isChecked() || ui->actionToggle_hz_grid->isChecked() || ui->actionToggle_piano_grid->isChecked() );

    stateChanged();
}


void RenderController::
    receiveToggleCursorMarker(bool value)
{
    model()->renderer->draw_cursor_marker = value;
    stateChanged();
}


void RenderController::
        transformChanged()
{
    Tfr::Cwt& c = Tfr::Cwt::Singleton();

    // keep in sync with receiveSetTimeFrequencyResolution
    float f = log(c.scales_per_octave()/2.L)/8;
    int value = f * tf_resolution->maximum() + .5;

    this->tf_resolution->setValue( value );


    // keep in sync with receiveSetYScale
    f = log(model()->renderer->y_scale)/4.f;
    f = (f > 0 ? 1 : -1) * sqrtf( f > 0 ? f : -f );
    value = (f + 1)/2 * yscale->maximum() + .5;

    this->yscale->setValue( value );


    // keep buttons in sync
    Ui::SaweMainWindow* main = dynamic_cast<Ui::SaweMainWindow*>(model()->project()->mainWindow());
    Ui::MainWindow* ui = main->getItems();
    if (model()->renderer->draw_piano)  hzmarker->setCheckedAction( ui->actionToggle_piano_grid );
    if (model()->renderer->draw_hz)  hzmarker->setCheckedAction( ui->actionToggle_hz_grid );
    switch( model()->renderer->color_mode )
    {
    case Heightmap::Renderer::ColorMode_Rainbow: color->setCheckedAction(ui->actionSet_colorscale); break;
    case Heightmap::Renderer::ColorMode_Grayscale: color->setCheckedAction(ui->actionSet_grayscale); break;
    case Heightmap::Renderer::ColorMode_FixedColor: color->setCheckedAction(ui->actionSet_colorscale); break;
    }
    ui->actionSet_heightlines->setChecked(model()->renderer->draw_height_lines);
    ui->actionToggleOrientation->setChecked(!model()->renderer->left_handed_axes);

    // clear worker assumptions of target
    model()->project()->worker.target(model()->renderSignalTarget);
}


void RenderController::
        receiveSetYScale( int value )
{
    // Keep in sync with transformChanged()
    float f = 2.f * value / yscale->maximum() - 1.f;
    model()->renderer->y_scale = exp( 4.f*f*f * (f>0?1:-1));

    stateChanged();

    yscale->setToolTip(QString("Intensity level %1").arg(model()->renderer->y_scale));
}


void RenderController::
        receiveSetTimeFrequencyResolution( int value )
{
    float FS = model()->project()->worker.source()->sample_rate();

    Tfr::Cwt& c = Tfr::Cwt::Singleton();

    // Keep in sync with transformChanged()
    float f = value / (float)tf_resolution->maximum();
    c.scales_per_octave( 2*exp( 8.L*f ), FS ); // scales_per_octave >= 2

    float wavelet_default_time_support = c.wavelet_default_time_support();
    float wavelet_fast_time_support = c.wavelet_time_support();
    c.wavelet_time_support(wavelet_default_time_support);

    Tfr::Stft& s = Tfr::Stft::Singleton();
    s.set_approximate_chunk_size( 0.25f*c.wavelet_time_support_samples(FS)/c.wavelet_time_support() );

    c.wavelet_fast_time_support( wavelet_fast_time_support );


    model()->renderSignalTarget->post_sink()->invalidate_samples( Signal::Intervals::Intervals_ALL );

    stateChanged();

    view->emitTransformChanged();

    tf_resolution->setToolTip(QString("Time/frequency resolution\nMorlet std: %1 (%2 scales/octave)\nSTFT window: %3 samples").arg(c.sigma(), 0, 'f', 1).arg(c.scales_per_octave(), 0, 'f', 1).arg(s.chunk_size()));
}


void RenderController::yscaleIncrease()
{    yscale->triggerAction( QAbstractSlider::SliderPageStepAdd ); }
void RenderController::yscaleDecrease()
{    yscale->triggerAction( QAbstractSlider::SliderPageStepSub ); }
void RenderController::tfresolutionIncrease()
{    tf_resolution->triggerAction( QAbstractSlider::SliderPageStepAdd ); }
void RenderController::tfresolutionDecrease()
{    tf_resolution->triggerAction( QAbstractSlider::SliderPageStepSub ); }


Signal::PostSink* RenderController::
        setBlockFilter(Signal::Operation* blockfilter)
{
    BlockFilterSink* bfs;
    Signal::pOperation blockop( blockfilter );
    Signal::pOperation channelop( bfs = new BlockFilterSink(blockop, model(), view, this));

    std::vector<Signal::pOperation> v;
    v.push_back( channelop );
    Signal::PostSink* ps = model()->renderSignalTarget->post_sink();
    ps->sinks(v);
    bfs->validateSize();
    bfs->invalidate_samples( Signal::Intervals::Intervals_ALL );

    stateChanged();

    view->emitTransformChanged();

    Ui::SaweMainWindow* main = dynamic_cast<Ui::SaweMainWindow*>(model()->project()->mainWindow());
    Ui::MainWindow* ui = main->getItems();

    ui->actionToggle_piano_grid->setVisible( true );
    hz_scale->setEnabled( true );

    return ps;
}


Tfr::Transform* RenderController::
        currentTransform()
{
    Signal::PostSink* ps = model()->renderSignalTarget->post_sink();
    if (ps->sinks().empty())
        return 0;
    BlockFilterSink* bfs = dynamic_cast<BlockFilterSink*>(ps->sinks()[0].get());
    BOOST_ASSERT( bfs != 0 );
    Tfr::Filter* filter = dynamic_cast<Tfr::Filter*>(bfs->Operation::source().get());
    return filter->transform().get();
}


void RenderController::
        receiveSetTransform_Cwt()
{
    Heightmap::CwtToBlock* cwtblock = new Heightmap::CwtToBlock(&model()->collections);
    cwtblock->complex_info = Heightmap::ComplexInfo_Amplitude_Non_Weighted;

    setBlockFilter( cwtblock );
}


void RenderController::
        receiveSetTransform_Stft()
{
    Heightmap::StftToBlock* stftblock = new Heightmap::StftToBlock(&model()->collections);

    setBlockFilter( stftblock );
}


void RenderController::
        receiveSetTransform_Cwt_phase()
{
    Heightmap::CwtToBlock* cwtblock = new Heightmap::CwtToBlock(&model()->collections);
    cwtblock->complex_info = Heightmap::ComplexInfo_Phase;

    setBlockFilter( cwtblock );
}


#ifdef USE_CUDA
void RenderController::
        receiveSetTransform_Cwt_reassign()
{
    Heightmap::CwtToBlock* cwtblock = new Heightmap::CwtToBlock(&model()->collections);
    cwtblock->complex_info = Heightmap::ComplexInfo_Amplitude_Weighted;

    Signal::PostSink* ps = setBlockFilter( cwtblock );

    ps->filter( Signal::pOperation(new Filters::Reassign()));
}
#endif


void RenderController::
        receiveSetTransform_Cwt_ridge()
{
    Heightmap::CwtToBlock* cwtblock = new Heightmap::CwtToBlock(&model()->collections);
    cwtblock->complex_info = Heightmap::ComplexInfo_Amplitude_Non_Weighted;

    Signal::PostSink* ps = setBlockFilter( cwtblock );

    ps->filter( Signal::pOperation(new Filters::Ridge()));
}


void RenderController::
        receiveSetTransform_Cwt_weight()
{
    Heightmap::CwtToBlock* cwtblock = new Heightmap::CwtToBlock(&model()->collections);
    cwtblock->complex_info = Heightmap::ComplexInfo_Amplitude_Weighted;

    setBlockFilter( cwtblock );
}


void RenderController::
        receiveSetTransform_Cepstrum()
{
    Heightmap::CepstrumToBlock* cepstrumblock = new Heightmap::CepstrumToBlock(&model()->collections);

    setBlockFilter( cepstrumblock );
}


void RenderController::
        receiveSetTransform_DrawnWaveform()
{
    Heightmap::DrawnWaveformToBlock* drawnwaveformblock = new Heightmap::DrawnWaveformToBlock(&model()->collections);

    setBlockFilter( drawnwaveformblock );

    Ui::SaweMainWindow* main = dynamic_cast<Ui::SaweMainWindow*>(model()->project()->mainWindow());
    Ui::MainWindow* ui = main->getItems();

    hz_scale->setEnabled( false );
    if (ui->actionToggle_piano_grid->isChecked())
        hzmarker->setChecked( false );
    ui->actionToggle_piano_grid->setVisible( false );

    // blockfilter sets the proper "frequency" axis
    linearScale->trigger();
}


void RenderController::
        receiveLinearScale()
{
    float fs = model()->project()->head->head_source()->sample_rate();

    Tfr::FreqAxis fa;
    fa.setLinear( fs );

    if (currentTransform() && fa.min_hz < currentTransform()->freqAxis(fs).min_hz)
    {
        fa.min_hz = currentTransform()->freqAxis(fs).min_hz;
        fa.f_step = fs/2 - fa.min_hz;
    }

    model()->display_scale( fa );

    view->emitAxisChanged();
    stateChanged();
}


void RenderController::
        receiveLogScale()
{
    float fs = model()->project()->head->head_source()->sample_rate();

    Tfr::FreqAxis fa;
    fa.setLogarithmic(
            Tfr::Cwt::Singleton().wanted_min_hz(),
            Tfr::Cwt::Singleton().get_max_hz(fs) );

    if (currentTransform() && fa.min_hz < currentTransform()->freqAxis(fs).min_hz)
    {
        // Happens typically when currentTransform is a cepstrum transform with a short window size
        fa.setLogarithmic(
                currentTransform()->freqAxis(fs).min_hz,
                Tfr::Cwt::Singleton().get_max_hz(fs) );
    }

    model()->display_scale( fa );

    view->emitAxisChanged();
    stateChanged();
}


void RenderController::
        receiveCepstraScale()
{
    float fs = model()->project()->head->head_source()->sample_rate();

    Tfr::FreqAxis fa;
    fa.setQuefrencyNormalized( fs, Tfr::Cepstrum::Singleton().chunk_size() );

    if (currentTransform() && fa.min_hz < currentTransform()->freqAxis(fs).min_hz)
    {
        // min_hz = 2*fs/window_size;
        // window_size = 2*fs/min_hz;
        fa.setQuefrencyNormalized( fs, 2*fs/currentTransform()->freqAxis(fs).min_hz );
    }

    model()->display_scale( fa );

    view->emitAxisChanged();
    stateChanged();
}


void RenderController::
        receiveLinearAmplitude()
{
    model()->amplitude_axis( Heightmap::AmplitudeAxis_Linear );
    view->emitAxisChanged();
    stateChanged();
}


void RenderController::
        receiveLogAmplitude()
{
    model()->amplitude_axis( Heightmap::AmplitudeAxis_Logarithmic );
    view->emitAxisChanged();
    stateChanged();
}


void RenderController::
        receiveFifthAmplitude()
{
    model()->amplitude_axis( Heightmap::AmplitudeAxis_5thRoot );
    view->emitAxisChanged();
    stateChanged();
}


RenderModel *RenderController::
        model()
{
    BOOST_ASSERT( view );
    return view->model;
}


void RenderController::
        setupGui()
{
    Ui::SaweMainWindow* main = dynamic_cast<Ui::SaweMainWindow*>(model()->project()->mainWindow());
    toolbar_render = new Support::ToolBar(main);
    toolbar_render->setObjectName(QString::fromUtf8("toolBarRenderController"));
    toolbar_render->setWindowTitle(QApplication::translate("MainWindow", "toolBar", 0, QApplication::UnicodeUTF8));
    toolbar_render->setEnabled(true);
    toolbar_render->setContextMenuPolicy(Qt::NoContextMenu);
    toolbar_render->setToolButtonStyle(Qt::ToolButtonIconOnly);
    main->addToolBar(Qt::BottomToolBarArea, toolbar_render);

    connect(main->getItems()->actionToggleTransformToolBox, SIGNAL(toggled(bool)), toolbar_render, SLOT(setVisible(bool)));
    connect((Support::ToolBar*)toolbar_render, SIGNAL(visibleChanged(bool)), main->getItems()->actionToggleTransformToolBox, SLOT(setChecked(bool)));

    main->installEventFilter( this );

    // Find Qt Creator managed actions
    Ui::MainWindow* ui = main->getItems();


    // ComboBoxAction* hzmarker
    {   hzmarker = new ComboBoxAction(toolbar_render);
        hzmarker->setObjectName("hzmarker");
        hzmarker->addActionItem( ui->actionToggle_hz_grid );
        hzmarker->addActionItem( ui->actionToggle_piano_grid );
        toolbar_render->addWidget( hzmarker );
        toolbar_render->addAction( ui->actionToggle_t_grid );
        toolbar_render->addAction( ui->actionToggle_cursor_marker );

        connect(ui->actionToggle_piano_grid, SIGNAL(toggled(bool)), SLOT(receiveTogglePiano(bool)));
        connect(ui->actionToggle_hz_grid, SIGNAL(toggled(bool)), SLOT(receiveToggleHz(bool)));
        connect(ui->actionToggle_t_grid, SIGNAL(toggled(bool)), SLOT(receiveToggleTAxis(bool)));
        connect(ui->actionToggle_cursor_marker, SIGNAL(toggled(bool)), SLOT(receiveToggleCursorMarker(bool)));
    }

    connect(ui->actionResetGraphics, SIGNAL(triggered()), view, SLOT(clearCaches()));


    // ComboBoxAction* color
    {   color = new ComboBoxAction(toolbar_render);
        color->setObjectName("ComboBoxActioncolor");
        color->decheckable( false );
        color->addActionItem( ui->actionSet_rainbow_colors );
        color->addActionItem( ui->actionSet_grayscale );
        color->addActionItem( ui->actionSet_colorscale );
        toolbar_render->addWidget( color );

        connect(ui->actionSet_rainbow_colors, SIGNAL(triggered()), SLOT(receiveSetRainbowColors()));
        connect(ui->actionSet_grayscale, SIGNAL(triggered()), SLOT(receiveSetGrayscaleColors()));
        connect(ui->actionSet_colorscale, SIGNAL(triggered()), SLOT(receiveSetColorscaleColors()));
        color->setCheckedAction(ui->actionSet_colorscale);
    }

    // ComboBoxAction* channels
    {   channelselector = new QToolButton(toolbar_render);
        channelselector->setObjectName("channelselector");
        channelselector->setCheckable( false );
        channelselector->setText("Channels");
        channelselector->setContextMenuPolicy( Qt::ActionsContextMenu );
        channelselector->setToolTip("Press to get a list of channels (or right click)");
        toolbar_render->addWidget( channelselector );
    }

    // QAction *actionSet_heightlines
    toolbar_render->addAction(ui->actionSet_heightlines);
    connect(ui->actionSet_heightlines, SIGNAL(toggled(bool)), SLOT(receiveToogleHeightlines(bool)));

    toolbar_render->addAction(ui->actionToggleOrientation);
    connect(ui->actionToggleOrientation, SIGNAL(toggled(bool)), SLOT(receiveToggleOrientation(bool)));

    // ComboBoxAction* transform
    {   connect(ui->actionTransform_Cwt, SIGNAL(triggered()), SLOT(receiveSetTransform_Cwt()));
        connect(ui->actionTransform_Stft, SIGNAL(triggered()), SLOT(receiveSetTransform_Stft()));
//        connect(ui->actionTransform_Cwt_phase, SIGNAL(triggered()), SLOT(receiveSetTransform_Cwt_phase()));
//        connect(ui->actionTransform_Cwt_reassign, SIGNAL(triggered()), SLOT(receiveSetTransform_Cwt_reassign()));
//        connect(ui->actionTransform_Cwt_ridge, SIGNAL(triggered()), SLOT(receiveSetTransform_Cwt_ridge()));
//        connect(ui->actionTransform_Cwt_weight, SIGNAL(triggered()), SLOT(receiveSetTransform_Cwt_weight()));
        connect(ui->actionTransform_Cepstrum, SIGNAL(triggered()), SLOT(receiveSetTransform_Cepstrum()));
        connect(ui->actionTransform_Waveform, SIGNAL(triggered()), SLOT(receiveSetTransform_DrawnWaveform()));

        transform = new ComboBoxAction(toolbar_render);
        transform->setObjectName("ComboBoxActiontransform");
        transform->addActionItem( ui->actionTransform_Stft );
        transform->addActionItem( ui->actionTransform_Cwt );
        transform->addActionItem( ui->actionTransform_Cepstrum );
        transform->addActionItem( ui->actionTransform_Waveform );
//        transform->addActionItem( ui->actionTransform_Cwt_phase );
//        transform->addActionItem( ui->actionTransform_Cwt_reassign );
//        transform->addActionItem( ui->actionTransform_Cwt_ridge );
//        transform->addActionItem( ui->actionTransform_Cwt_weight );
        transform->decheckable( false );
        toolbar_render->addWidget( transform );

        unsigned k=0;
        foreach( QAction* a, transform->actions())
        {
            a->setShortcut('1' + k++);
        }
    }


    // ComboBoxAction* hz-scale
    {   linearScale = new QAction( toolbar_render );
        logScale = new QAction( toolbar_render );
        cepstraScale = new QAction( toolbar_render );

        linearScale->setText("Linear scale");
        logScale->setText("Logarithmic scale");
        cepstraScale->setText("Cepstra scale");

        // for serialization
        linearScale->setObjectName("linearScale");
        logScale->setObjectName("logScale");
        cepstraScale->setObjectName("cepstraScale");

        linearScale->setCheckable( true );
        logScale->setCheckable( true );
        cepstraScale->setCheckable( true );

        connect(linearScale, SIGNAL(triggered()), SLOT(receiveLinearScale()));
        connect(logScale, SIGNAL(triggered()), SLOT(receiveLogScale()));
        connect(cepstraScale, SIGNAL(triggered()), SLOT(receiveCepstraScale()));

        hz_scale = new ComboBoxAction();
        hz_scale->setObjectName("hz_scale");
        hz_scale->addActionItem( linearScale );
        hz_scale->addActionItem( logScale );
        hz_scale->addActionItem( cepstraScale );
        hz_scale->decheckable( false );
        toolbar_render->addWidget( hz_scale );

        unsigned k=0;
        foreach( QAction* a, hz_scale->actions())
        {
            a->setShortcut(QString("Ctrl+") + ('1' + k++));
        }
        logScale->trigger();
    }


    // ComboBoxAction* amplitude-scale
    {   QAction* linearAmplitude = new QAction( toolbar_render );
        QAction* logAmpltidue = new QAction( toolbar_render );
        QAction* fifthAmpltidue = new QAction( toolbar_render );

        linearAmplitude->setText("Linear amplitude");
        logAmpltidue->setText("Logarithmic amplitude");
        fifthAmpltidue->setText("|A|^(1/5) amplitude");

        // for serialization
        linearAmplitude->setObjectName("linearAmplitude");
        logAmpltidue->setObjectName("logAmpltidue");
        fifthAmpltidue->setObjectName("fifthAmpltidue");

        linearAmplitude->setCheckable( true );
        logAmpltidue->setCheckable( true );
        fifthAmpltidue->setCheckable( true );

        connect(linearAmplitude, SIGNAL(triggered()), SLOT(receiveLinearAmplitude()));
        connect(logAmpltidue, SIGNAL(triggered()), SLOT(receiveLogAmplitude()));
        connect(fifthAmpltidue, SIGNAL(triggered()), SLOT(receiveFifthAmplitude()));

        amplitude_scale = new ComboBoxAction();
        amplitude_scale->addActionItem( linearAmplitude );
        amplitude_scale->addActionItem( logAmpltidue );
        amplitude_scale->addActionItem( fifthAmpltidue );
        amplitude_scale->decheckable( false );
        toolbar_render->addWidget( amplitude_scale );

        unsigned k=0;
        foreach( QAction* a, amplitude_scale->actions())
        {
            a->setShortcut(QString("Alt+") + ('1' + k++));
        }
        linearAmplitude->trigger();
    }


    // QSlider * yscale
    {   yscale = new QSlider( toolbar_render );
        yscale->setObjectName("yscale");
        yscale->setOrientation( Qt::Horizontal );
        yscale->setMaximum( 10000 );
        yscale->setValue( 5000 );
        yscale->setToolTip( "Intensity level" );
        yscale->setPageStep( 100 );
        toolbar_render->addWidget( yscale );

        connect(yscale, SIGNAL(valueChanged(int)), SLOT(receiveSetYScale(int)));
        receiveSetYScale(yscale->value());

        QAction* yscaleIncrease = new QAction(yscale);
        QAction* yscaleDecrease = new QAction(yscale);

        yscaleIncrease->setShortcut(QString("Alt+Up"));
        yscaleDecrease->setShortcut(QString("Alt+Down"));

        connect(yscaleIncrease, SIGNAL(triggered()), SLOT(yscaleIncrease()));
        connect(yscaleDecrease, SIGNAL(triggered()), SLOT(yscaleDecrease()));

        yscale->addAction( yscaleIncrease );
        yscale->addAction( yscaleDecrease );
    }


    // QSlider * tf_resolution
    {   tf_resolution = new QSlider( toolbar_render );
        tf_resolution->setObjectName("tf_resolution");
        tf_resolution->setOrientation( Qt::Horizontal );
        tf_resolution->setMaximum( 10000 );
        tf_resolution->setValue( 3000 );
        tf_resolution->setToolTip( "Time/frequency resolution." );
        tf_resolution->setPageStep( 100 );
        toolbar_render->addWidget( tf_resolution );

        connect(tf_resolution, SIGNAL(valueChanged(int)), SLOT(receiveSetTimeFrequencyResolution(int)));
        receiveSetTimeFrequencyResolution(tf_resolution->value());

        QAction* tfresolutionIncrease = new QAction(yscale);
        QAction* tfresolutionDecrease = new QAction(yscale);

        tfresolutionIncrease->setShortcut(QString("Alt+Right"));
        tfresolutionDecrease->setShortcut(QString("Alt+Left"));

        connect(tfresolutionIncrease, SIGNAL(triggered()), SLOT(tfresolutionIncrease()));
        connect(tfresolutionDecrease, SIGNAL(triggered()), SLOT(tfresolutionDecrease()));

        tf_resolution->addAction( tfresolutionIncrease );
        tf_resolution->addAction( tfresolutionDecrease );
    }

    connect(this->view.data(), SIGNAL(transformChanged()), SLOT(updateFreqAxis()));
    connect(this->view.data(), SIGNAL(transformChanged()), SLOT(updateChannels()));
    connect(this->view.data(), SIGNAL(transformChanged()), SLOT(transformChanged()));

    // Release cuda buffers and disconnect them from OpenGL before destroying
    // OpenGL rendering context. Just good housekeeping.
    connect(view, SIGNAL(destroying()), SLOT(clearCachedHeightmap()));

    // Create the OpenGL rendering context early. Because we want to create the
    // cuda context (in main.cpp) and bind it to an OpenGL context before the
    // context is required to be created by lazy initialization when painting
    // the widget
    view->glwidget = new QGLWidget( 0, Sawe::Application::shared_glwidget(), Qt::WindowFlags(0) );
    view->glwidget->makeCurrent();

    view->graphicsview = new GraphicsView(view);
    view->graphicsview->setViewport(view->glwidget);
    view->glwidget->makeCurrent(); // setViewport makes the glwidget loose context, take it back
    view->tool_selector = view->graphicsview->toolSelector(0, model()->project()->commandInvoker());

    main->centralWidget()->layout()->setMargin(0);
    main->centralWidget()->layout()->addWidget(view->graphicsview);

    view->emitTransformChanged();
}


void RenderController::
        clearCachedHeightmap()
{
    // Stop worker from producing any more heightmaps by disconnecting
    // the collection callback from worker.
    if (model()->renderSignalTarget == model()->project()->worker.target())
        model()->project()->worker.target(Signal::pTarget());
    model()->renderSignalTarget.reset();


    // Assuming calling thread is the GUI thread.

    // Clear all cached blocks and release cuda memory befure destroying cuda
    // context
    foreach( const boost::shared_ptr<Heightmap::Collection>& collection, model()->collections )
        collection->reset();
}


void RenderController::
        updateFreqAxis()
{
    switch(model()->display_scale().axis_scale)
    {
    case Tfr::AxisScale_Linear:
        receiveLinearScale();
        break;

    case Tfr::AxisScale_Logarithmic:
        receiveLogScale();
        break;

    case Tfr::AxisScale_Quefrency:
        receiveCepstraScale();
        break;

    default:
        break;
    }
}


void RenderController::
        updateChannels()
{
    Signal::RerouteChannels* channels = model()->renderSignalTarget->channels();
    unsigned N = channels->source()->num_channels();
    for (unsigned i=0; i<N; ++i)
    {
        foreach (QAction* o, channelselector->actions())
        {
            QAction* a = dynamic_cast<QAction*>(o);
            if (!a)
                continue;
            if (a->data().toUInt() >= N)
                delete a;
        }

        for (unsigned i = channelselector->actions().size(); i<N; ++i)
        {
            QAction* a = new QAction(channelselector);
            a->setText(QString("Channel %1").arg(i));
            a->setData( i );
            a->setCheckable( true );
            a->setChecked( false );
            connect(a, SIGNAL(toggled(bool)), SLOT( reroute() ));
            channelselector->addAction( a );

            a->setChecked( true ); // invokes reroute
        }
    }
}


void RenderController::
        reroute()
{
    Signal::RerouteChannels* channels = model()->renderSignalTarget->channels();
    foreach (QAction* o, channelselector->actions())
    {
        unsigned c = o->data().toUInt();
        channels->map(c, o->isChecked() ? c : Signal::RerouteChannels::NOTHING );
    }
}


bool RenderController::
        eventFilter(QObject *o, QEvent *e)
{
    //eventFilter is called a lot, do the most simple test possible to
    // determine if we're interested or not
    if (e->type() == QEvent::FocusIn || e->type() == QEvent::FocusOut)
    {
        if (model()->project()->mainWindow() == o)
        {
            if (e->type() == QEvent::FocusIn)
                windowGotFocus();
            else
                windowLostFocus();
        }
    }

    // this eventFilter doesn't block any events
    return false;
}


void RenderController::
        windowLostFocus()
{
    model()->project()->worker.min_fps( 20 );
}


void RenderController::
        windowGotFocus()
{
    model()->project()->worker.min_fps( 2 );
}


} // namespace Tools
